"""
recommender.py
--------------
Contextual well-being recommendation generation via Ollama.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Design
------
  Primary:   Ollama (Llama 3.1 8B) — single-shot prompt engineered for
             a short, friendly, context-aware recommendation.
             2-second timeout; falls back to template if exceeded.
  Secondary: Ollama (Qwen 3.5 4B) — runs in parallel when
             LLM_COMPARISON_MODE = True; result stored for thesis
             evaluation only and never delays the notification.
  Fallback:  Template-based messages selected by trigger type.
             Activates when Ollama is not running or exceeds the timeout.

Output
------
  Always a non-empty string ≤ 200 characters suitable for a Windows
  desktop notification (driven by the primary model).

Usage
-----
  rec = WellbeingRecommender()
  message = rec.generate({
      "app_category":         "Social Media",
      "active_window":        "Facebook – Google Chrome",
      "stress_index":         0.72,
      "valence":              -0.45,
      "arousal":              0.60,
      "session_minutes":      85,
      "minutes_since_break":  55,
      "blink_rate":           9.2,
  })
"""

import os
import sys
import time
import concurrent.futures

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.config import (
    OLLAMA_MODEL,
    OLLAMA_MODEL_PRIMARY,
    OLLAMA_MODEL_SECONDARY,
    OLLAMA_TIMEOUT,
    OLLAMA_SECONDARY_TIMEOUT,
    LLM_COMPARISON_MODE,
    DB_PATH,
)

# ── Template fallbacks, keyed by trigger type ─────────────────────────────────
# Chosen to be specific and actionable, not generic.
_TEMPLATES: dict[str, list[str]] = {
    "high_stress": [
        "Stress signals detected. Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s.",
        "Your stress indicators are elevated. A 5-minute walk away from the screen can help.",
        "Tension detected. Roll your shoulders back, take three slow breaths, then continue.",
    ],
    "disengagement": [
        "You seem disengaged. Identify the next one task and work on it for just 10 minutes.",
        "Low engagement detected. Try the Pomodoro technique: 25 min focused, 5 min break.",
        "Feeling flat? A short walk — even just to another room — can reset your energy.",
    ],
    "negative_affect": [
        "You seem tense. Try box breathing: inhale 4 seconds, hold 4, exhale 4, hold 4.",
        "Tension signals detected. A 2-minute breathing exercise can reset your focus.",
    ],
    "eye_strain": [
        "Blink rate is low — apply the 20-20-20 rule: look 20 feet away for 20 seconds now.",
        "Eye strain detected. Close your eyes for 20 seconds and let them rest.",
    ],
    "prolonged_idle": [
        "You've been idle for a while. Check in: are you stuck? Breaking the task down may help.",
        "Long idle period. Even a small action — writing a note, drafting a line — builds momentum.",
    ],
    "positive_flow": [
        "You are in great flow right now. Keep it up and remember to take a short break soon!",
        "Strong focus and positive signals — excellent work. Stay hydrated and keep going!",
    ],
    "negative_valence": [
        "You seem frustrated. Step away for 5 minutes — returning with fresh eyes often helps.",
        "Low mood detected. A brief change of scenery or a glass of water can shift perspective.",
    ],
    "default": [
        "You've been working a while. A 5-minute break will help you sustain focus longer.",
        "Time for a short pause. Stand up, stretch, and return refreshed.",
    ],
}

# ── Per-trigger guidance injected into the LLM system prompt ──────────────────
_TRIGGER_GUIDANCE: dict[str, str] = {
    "high_stress":     "Suggest a break, breathing exercise, or physical movement.",
    "disengagement":   "Suggest task switching, a short walk, or a refocusing technique.",
    "negative_affect": "Suggest a breathing exercise or brief mindfulness activity.",
    "eye_strain":      "Suggest the 20-20-20 rule: look 20 feet away for 20 seconds.",
    "prolonged_idle":  "Gently nudge the user to re-engage with their work.",
    "positive_flow":   "Give genuine brief encouragement and acknowledge the good work.",
    "default":         "Be specific and actionable.",
}

_TEMPLATE_INDEX: dict[str, int] = {}   # rotating pointer per trigger type


class WellbeingRecommender:
    """
    Generates contextual well-being recommendations.

    Primary path:   Ollama (Llama 3.1 8B) with 2-second timeout.
    Comparison path: When LLM_COMPARISON_MODE is True, Qwen 3.5 4B runs in
                    parallel in a background thread; the primary result is
                    returned immediately so the notification is never delayed.
                    Both responses are saved to llm_comparisons for thesis
                    evaluation after both models have finished.
    Fallback path:  Rotating template messages when Ollama is unavailable.

    The same _ollama_ok probe strategy as WindowClassifier is used:
      - None  = not yet probed
      - True  = Ollama available (continues trying)
      - False = disabled for this session (avoids repeated latency)
    """

    def __init__(
        self,
        model:   str   = OLLAMA_MODEL,
        timeout: float = OLLAMA_TIMEOUT,
        db_path: str   = DB_PATH,
    ):
        self._model    = model
        self._timeout  = timeout
        self._db_path  = db_path
        self._ollama_ok: bool | None = None
        # Primary executor — used for the notification-driving LLM call.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ollama-rec-primary"
        )
        # Secondary executor — used only in comparison mode; never blocks
        # the notification because its result is collected in a daemon thread.
        self._executor_secondary = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ollama-rec-secondary"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, context: dict) -> str:
        """
        Return a well-being recommendation string (≤ 200 characters).

        When LLM_COMPARISON_MODE is True, both models are invoked in parallel.
        The primary model (Llama) drives the returned message; the secondary
        model (Qwen) result is persisted asynchronously to llm_comparisons.

        Parameters
        ----------
        context : dict
            app_category       : str   — dominant app category
            active_window      : str   — current window title (optional)
            stress_index       : float — fusion stress score 0-1
            valence            : float | None  — avg valence from face module
            arousal            : float | None  — avg arousal from face module
            session_minutes    : int   — minutes since session started
            minutes_since_break: int | None  — minutes since last break
            blink_rate         : float | None — blinks per minute
        """
        trigger = _infer_trigger(context)

        if self._should_try_ollama():
            if LLM_COMPARISON_MODE:
                msg = self._generate_with_comparison(context)
            else:
                msg = self._generate_ollama(context)
            if msg:
                return msg[:200]

        return _template_message(trigger)

    def close(self) -> None:
        """Shut down both thread pools cleanly."""
        self._executor.shutdown(wait=False)
        self._executor_secondary.shutdown(wait=False)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _should_try_ollama(self) -> bool:
        if self._ollama_ok is not None:
            return self._ollama_ok

        try:
            import ollama  # noqa: F401
            self._ollama_ok = True
        except ImportError:
            self._ollama_ok = False

        return self._ollama_ok

    def _generate_ollama(self, context: dict) -> str | None:
        """Call the primary model with a timeout. Return None on failure."""
        prompt = _build_prompt(context)
        try:
            future   = self._executor.submit(self._timed_ollama_call, OLLAMA_MODEL_PRIMARY, prompt)
            raw, _   = future.result(timeout=self._timeout)
            return _clean_response(raw)

        except concurrent.futures.TimeoutError:
            return None

        except Exception as exc:
            print(f"[Recommender] Ollama error: {exc}. Switching to template fallback.")
            self._ollama_ok = False
            return None

    def _generate_with_comparison(self, context: dict) -> str | None:
        """
        Run Llama first (notification-critical), then Qwen sequentially in a
        background thread once Llama has finished and released GPU memory.

        Sequential execution avoids VRAM contention that caused Qwen to return
        empty responses when both models loaded simultaneously.  The 2-second
        sleep between calls gives the GPU allocator time to reclaim Llama's
        memory before Qwen starts loading.

        The notification is dispatched with Llama's response immediately after
        Llama completes; Qwen's result is saved to llm_comparisons once it
        finishes without affecting the notification latency.
        """
        import threading

        prompt = _build_prompt(context)

        # ── Step 1: Run Llama (primary, notification-critical) ────────────────
        llama_text, llama_ms = None, None
        try:
            f_llama = self._executor.submit(
                self._timed_ollama_call, OLLAMA_MODEL_PRIMARY, prompt
            )
            raw_llama, llama_ms = f_llama.result(timeout=self._timeout)
            llama_text = _clean_response(raw_llama)
        except concurrent.futures.TimeoutError:
            llama_ms = round(self._timeout * 1000, 1)
            print(f"[Recommender] Primary model timed out after {llama_ms:.0f} ms")
        except Exception as exc:
            print(f"[Recommender] Primary model error: {exc}. Falling back to template.")
            self._ollama_ok = False

        print(f"[Recommender] Llama response: {llama_text[:50] if llama_text else 'None'}")

        # ── Step 2: Run Qwen in background after Llama finishes ───────────────
        def _run_qwen_and_save() -> None:
            # Allow GPU memory to settle after Llama unloads before Qwen starts.
            time.sleep(2)

            qwen_text, qwen_ms = None, None
            try:
                f_qwen = self._executor_secondary.submit(
                    self._timed_ollama_call, OLLAMA_MODEL_SECONDARY, prompt
                )
                raw_qwen, qwen_ms = f_qwen.result(timeout=OLLAMA_SECONDARY_TIMEOUT)
                qwen_text = _clean_response(raw_qwen)
            except concurrent.futures.TimeoutError:
                qwen_ms = round(OLLAMA_SECONDARY_TIMEOUT * 1000, 1)
                print(f"[Recommender] Secondary model timed out after {qwen_ms:.0f} ms")
            except Exception as exc:
                qwen_ms = round(OLLAMA_SECONDARY_TIMEOUT * 1000, 1)
                print(f"[Recommender] Secondary model ({OLLAMA_MODEL_SECONDARY}) error: {exc}")

            print(f"[Recommender] Qwen response: {qwen_text[:50] if qwen_text else 'None'}")
            try:
                _save_comparison_row(
                    self._db_path, context,
                    llama_text, qwen_text,
                    llama_ms,   qwen_ms,
                )
            except Exception as exc:
                print(f"[Recommender] Comparison DB save error: {exc}")

        threading.Thread(target=_run_qwen_and_save, daemon=True).start()

        return llama_text

    def _timed_ollama_call(
        self,
        model:        str,
        prompt:       str,
        use_thinking: bool = True,
    ) -> tuple[str, float]:
        """Blocking Ollama call with wall-clock timing.

        For Qwen 3.5, the chat endpoint is used with ``think=False`` to
        suppress chain-of-thought reasoning at the API level.  For all other
        models the generate endpoint is used without the think option, which
        Llama 3.1 does not support.

        Parameters
        ----------
        model        : Ollama model identifier
        prompt       : complete prompt string
        use_thinking : explicit override — False forces think suppression even
                       on non-Qwen models (default True)

        Returns
        -------
        tuple of (response_text, latency_ms)
        """
        import time
        import ollama

        is_qwen = "qwen" in model.lower()
        suppress_thinking = is_qwen or not use_thinking

        start = time.monotonic()

        if suppress_thinking:
            if is_qwen:
                # Triple suppression for Qwen: API option + system role instruction
                # + /no_think text suffix.  The options-only approach fails because
                # thinking tokens consume num_predict before any content is emitted.
                messages = [
                    {
                        "role":    "system",
                        "content": (
                            "You are a well-being assistant. Respond directly and "
                            "concisely. Do not show your thinking process. Output "
                            "ONLY the final recommendation, nothing else."
                        ),
                    },
                    {"role": "user", "content": prompt + " /no_think"},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            response = ollama.chat(
                model    = model,
                messages = messages,
                options  = {
                    "think":       False,
                    "num_predict": 150,
                    "num_ctx":     2048,
                    "temperature": 0.7,
                },
            )
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            if isinstance(response, dict):
                msg_obj       = response.get("message", {})
                text          = msg_obj.get("content", "") or ""
                thinking_text = msg_obj.get("thinking", "") or ""
            else:
                msg_obj       = getattr(response, "message", None)
                text          = (msg_obj.content if msg_obj and hasattr(msg_obj, "content") else "") or ""
                thinking_text = (getattr(msg_obj, "thinking", "") if msg_obj else "") or ""
        else:
            response = ollama.generate(
                model   = model,
                prompt  = call_prompt,
                options = {"num_predict": 150, "temperature": 0.7},
            )
            latency_ms    = round((time.monotonic() - start) * 1000, 1)
            thinking_text = ""
            if isinstance(response, dict):
                text = response.get("response", "") or ""
            else:
                text = getattr(response, "response", "") or ""

        if text:
            print(f"[Recommender] {model} raw text ({latency_ms:.0f} ms): {repr(text[:80])}")
        else:
            print(f"[Recommender] {model} EMPTY ({latency_ms:.0f} ms) — full response: {response!r}")
            # Last resort: extract a usable sentence from the thinking field if
            # done_reason='length' consumed all tokens before producing content.
            text = _extract_from_thinking(thinking_text)
            if text:
                print(f"[Recommender] {model} salvaged from thinking field: {repr(text[:80])}")

        return text, latency_ms

    def _ollama_call(self, prompt: str) -> str:
        """Blocking Ollama call — kept for backwards compatibility."""
        raw, _ = self._timed_ollama_call(self._model, prompt)
        return raw


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT + RESPONSE HELPERS (module-level, no state)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_from_thinking(thinking: str) -> str:
    """Pull the final recommendation out of a Qwen thinking block.

    Called only when content is empty (done_reason='length' consumed all
    num_predict tokens on internal reasoning).

    Strategy:
    1. Split on common final-answer markers; take text AFTER the last one found.
    2. Strip markdown bold/italic and whitespace.
    3. Accept only if 20–500 characters.
    4. Return empty string if nothing valid found (caller uses template fallback).
    """
    import re

    if not thinking:
        return ""

    markers = [
        "Final Decision:",
        "Final Polish:",
        "Final Answer:",
        "Revised:",
        "Refining",
        "Let's make it clearer:",
        "6.", "7.", "8.",           # last numbered steps in thinking chain
    ]

    best_pos   = -1
    best_marker = ""
    for marker in markers:
        pos = thinking.rfind(marker)          # last occurrence
        if pos > best_pos:
            best_pos    = pos
            best_marker = marker

    candidate = ""
    if best_pos >= 0:
        after = thinking[best_pos + len(best_marker):].strip()
        # Keep only the first paragraph (stop at a blank line or section header)
        first_para = re.split(r"\n\s*\n|\n(?=[A-Z*#])", after)[0]
        candidate = first_para.strip()

    # Remove markdown bold/italic markers
    candidate = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", candidate)
    candidate = candidate.strip()

    if 20 <= len(candidate) <= 500:
        return candidate

    return ""


def _build_prompt(ctx: dict) -> str:
    """Build the Ollama prompt for the given context.

    The same prompt is used for all models. Thinking suppression for models
    that support it (e.g. Qwen 3.5) is handled at the API level via
    ``options={"think": False}`` inside ``_timed_ollama_call``, not via a
    text suffix which proved unreliable in practice.

    Parameters
    ----------
    ctx : dict — recommendation context (see WellbeingRecommender.generate)
    """
    stress   = ctx.get("stress_index", 0.0) or 0.0
    valence  = ctx.get("valence")
    arousal  = ctx.get("arousal")
    cat      = ctx.get("app_category", "Unknown")
    window   = ctx.get("active_window", "")
    session  = ctx.get("session_minutes", 0) or 0
    last_brk = ctx.get("minutes_since_break")
    blinks   = ctx.get("blink_rate")
    trigger  = ctx.get("trigger_reason", "default")

    v_str     = f"{valence:+.2f}" if valence is not None else "N/A"
    a_str     = f"{arousal:+.2f}" if arousal is not None else "N/A"
    b_str     = f"{blinks:.1f}/min" if blinks is not None else "N/A"
    lb_str    = f"{last_brk} min ago" if last_brk is not None else "unknown"
    emo_label = ctx.get("emotion_label")
    guidance  = _TRIGGER_GUIDANCE.get(trigger, _TRIGGER_GUIDANCE["default"])

    if emo_label is not None:
        emo_line = f"- Emotional state: {emo_label} (valence={v_str}, arousal={a_str})\n"
    else:
        emo_line = f"- Emotional state: valence={v_str}, arousal={a_str}\n"

    return (
        "System: You are a well-being assistant for a knowledge worker.\n"
        "Generate ONE short, friendly recommendation (max 2 sentences).\n"
        f"{guidance}\n"
        "Never mention medical advice.\n\n"
        "Context:\n"
        f"- Trigger: {trigger}\n"
        f"- Current activity: {cat} in {window}\n"
        f"- Stress index: {stress:.2f}/1.0\n"
        + emo_line
        + f"- Session duration: {session} minutes\n"
        f"- Last break: {lb_str}\n"
        f"- Blink rate: {b_str} (normal: 15-20)\n\n"
        "Generate a recommendation:"
    )


def _strip_thinking_artifacts(text: str) -> str:
    """Remove chain-of-thought reasoning artifacts from a model response.

    Acts as a safety net in case ``options={"think": False}`` is not fully
    honoured by the installed Ollama version or model variant.

    Strips
    ------
    - Any ``<think>…</think>`` block, including the tags themselves
    - Lines that begin with "Thinking..." or "Thinking Process:"
    - Leading/trailing whitespace after removal
    """
    import re

    # Remove <think>…</think> blocks (multiline, non-greedy).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Drop entire lines whose content is a thinking prefix.
    lines = text.splitlines()
    filtered = [
        line for line in lines
        if not re.match(r"^\s*(Thinking\.{0,3}|Thinking Process:)", line, re.IGNORECASE)
    ]

    return "\n".join(filtered).strip()


def _clean_response(raw: str) -> str | None:
    """Strip thinking artifacts, whitespace, and unwanted prefixes.

    Returns None when the cleaned text is shorter than 10 characters
    (indicates the model produced an empty or degenerate response).
    """
    # First remove any reasoning/thinking blocks the model may have emitted.
    text = _strip_thinking_artifacts(raw)

    # Remove common LLM-generated prefixes.
    for prefix in ("Recommendation:", "Here's a recommendation:", "Sure!"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    if len(text) < 10:
        return None
    return text[:200]


def _infer_trigger(ctx: dict) -> str:
    """Choose the most relevant fallback template category from context.

    Prefers the explicit trigger_reason supplied by the aggregator.
    Falls back to signal-based inference when no reason is provided.
    """
    trigger_reason = ctx.get("trigger_reason")
    if trigger_reason and trigger_reason in _TEMPLATES:
        return trigger_reason

    # Signal-based inference as last resort
    stress   = ctx.get("stress_index", 0.0) or 0.0
    valence  = ctx.get("valence")
    arousal  = ctx.get("arousal")
    blinks   = ctx.get("blink_rate")
    activity = ctx.get("avg_activity_pct")

    if blinks is not None and blinks < 8:
        return "eye_strain"
    if arousal is not None and arousal < 0.2:
        return "disengagement"
    if activity is not None and activity < 20:
        return "prolonged_idle"
    if valence is not None and valence < -0.4:
        return "negative_valence"
    if stress >= 0.65:
        return "high_stress"
    return "default"


def _save_comparison_row(
    db_path:    str,
    context:    dict,
    llama_text: str | None,
    qwen_text:  str | None,
    llama_ms:   float | None,
    qwen_ms:    float | None,
) -> None:
    """Persist one LLM comparison row to the database.

    Called from a daemon background thread so it never blocks the notification.
    The llm_comparisons table is created on first write if it does not yet exist.
    """
    import json
    from datetime import datetime
    from src.utils.db import open_db, ensure_llm_comparisons_table

    conn = open_db(db_path)
    ensure_llm_comparisons_table(conn)

    # Serialise a small, analysis-relevant subset of the context.
    ctx_small = {k: context.get(k) for k in (
        "app_category", "active_window", "trigger_reason",
        "stress_index", "valence", "arousal",
    )}
    conn.execute(
        """
        INSERT INTO llm_comparisons
            (timestamp, trigger_reason, context_json,
             llama_response, qwen_response,
             llama_latency_ms, qwen_latency_ms, stress_index)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            context.get("trigger_reason"),
            json.dumps(ctx_small),
            llama_text,
            qwen_text,
            llama_ms,
            qwen_ms,
            context.get("stress_index"),
        ),
    )
    conn.commit()
    conn.close()
    print(
        f"[Recommender] Comparison saved — "
        f"llama={llama_ms} ms, qwen={qwen_ms} ms"
    )


def _template_message(trigger: str) -> str:
    """Return the next template message for the given trigger (rotating)."""
    templates = _TEMPLATES.get(trigger, _TEMPLATES["default"])
    idx = _TEMPLATE_INDEX.get(trigger, 0)
    msg = templates[idx % len(templates)]
    _TEMPLATE_INDEX[trigger] = (idx + 1) % len(templates)
    return msg


# ── Convenience module-level function ────────────────────────────────────────

_default_recommender: WellbeingRecommender | None = None


def generate(context: dict) -> str:
    """Module-level shortcut — creates the shared instance on first call."""
    global _default_recommender
    if _default_recommender is None:
        _default_recommender = WellbeingRecommender()
    return _default_recommender.generate(context)


# ── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    rec = WellbeingRecommender()

    contexts = [
        {
            "app_category": "Social Media", "active_window": "Facebook",
            "stress_index": 0.72, "valence": -0.45, "arousal": 0.60,
            "session_minutes": 85, "minutes_since_break": 55, "blink_rate": 9.2,
        },
        {
            "app_category": "Software Development", "active_window": "VS Code",
            "stress_index": 0.35, "valence": 0.2, "arousal": 0.3,
            "session_minutes": 120, "minutes_since_break": 80, "blink_rate": 14.0,
        },
        {
            "app_category": "Academic Work", "active_window": "Overleaf",
            "stress_index": 0.55, "valence": -0.20, "arousal": 0.15,
            "session_minutes": 200, "minutes_since_break": None, "blink_rate": 7.5,
        },
    ]

    print(f"Recommender mode: {'LLM (Ollama)' if rec._should_try_ollama() else 'template fallback'}\n")

    for i, ctx in enumerate(contexts, 1):
        msg = rec.generate(ctx)
        print(f"--- Context {i}: {ctx['app_category']} | stress={ctx['stress_index']:.2f}")
        print(f"    {msg}\n")

    rec.close()
