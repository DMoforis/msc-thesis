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

# Per-trigger category lists; the recommender rotates through these so that
# consecutive notifications of the same type suggest different strategies.
_VARIETY_CATEGORIES: dict[str, list[str]] = {
    "high_stress": [
        "physical (stretching, movement, brief walk)",
        "breathing or mindfulness exercise",
        "cognitive (task switch, prioritise, simplify)",
        "social (call a colleague, talk to someone)",
        "environmental (change location, get fresh air, open a window)",
    ],
    "disengagement": [
        "task re-engagement (identify the next single action)",
        "physical movement (walk, stretch, change posture)",
        "time-boxing technique (Pomodoro, 10-min sprint)",
        "environmental change (different room, natural light)",
    ],
    "negative_affect": [
        "breathing or grounding exercise",
        "brief physical reset (walk, stretch)",
        "cognitive reframing (write one thing going well)",
        "sensory break (close eyes, listen to calming audio)",
    ],
    "eye_strain": [
        "20-20-20 rule (look 20 ft away for 20 s)",
        "full eye rest (close eyes 30–60 s)",
        "screen distance and brightness adjustment",
        "blinking exercise and eye massage",
    ],
    "prolonged_idle": [
        "micro re-engagement (write one sentence or bullet)",
        "task decomposition (break the blocker into smaller steps)",
        "brief physical movement then return",
        "change of medium (voice memo, whiteboard sketch)",
    ],
    "positive_flow": [
        "brief encouragement and hydration reminder",
        "acknowledge focus streak and suggest a coming break",
        "positive reinforcement and posture check",
    ],
    "default": [
        "physical movement or stretch",
        "brief mindfulness or breathing",
        "short screen break",
        "hydration and posture reset",
    ],
}


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
        # Variety state — prevent repetitive recommendations across calls.
        self._recent_responses: list[str] = []   # last 3 delivered messages
        self._category_idx: dict[str, int] = {}  # rotating category pointer per trigger
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

        # Attach variety hints so _build_prompt can inject them into the LLM prompt.
        categories = _VARIETY_CATEGORIES.get(trigger, _VARIETY_CATEGORIES["default"])
        idx        = self._category_idx.get(trigger, 0)
        context    = {
            **context,
            "_variety_category": categories[idx % len(categories)],
            "_recent_responses": list(self._recent_responses),
        }
        self._category_idx[trigger] = idx + 1

        if self._should_try_ollama():
            if LLM_COMPARISON_MODE:
                msg = self._generate_with_comparison(context)
            else:
                msg = self._generate_ollama(context)
            if msg:
                msg = msg[:200]
                self._recent_responses = (self._recent_responses + [msg])[-3:]
                return msg

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
        import random
        import ollama

        temperature = round(random.uniform(0.6, 0.9), 2)

        is_qwen = "qwen" in model.lower()

        start = time.monotonic()

        if is_qwen:
            # Qwen 3.5 thinking suppression: /no_think prefix at message start
            # (documented Qwen format) + think:false API option as belt-and-suspenders.
            # num_predict=512 is necessary: even when thinking is partially suppressed
            # the model still emits some thinking tokens; 120 was exhausted before
            # any content was produced (eval_count=120, content='').
            response = ollama.chat(
                model    = model,
                messages = [{"role": "user", "content": "/no_think\n\n" + prompt}],
                options  = {
                    "think":       False,
                    "num_predict": 512,
                    "num_ctx":     4096,
                    "temperature": temperature,
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
        elif not use_thinking:
            # Non-Qwen model with thinking explicitly suppressed via chat endpoint.
            response = ollama.chat(
                model    = model,
                messages = [{"role": "user", "content": prompt}],
                options  = {"think": False, "num_predict": 150, "temperature": temperature},
            )
            latency_ms    = round((time.monotonic() - start) * 1000, 1)
            thinking_text = ""
            if isinstance(response, dict):
                text = response.get("message", {}).get("content", "") or ""
            else:
                msg_obj = getattr(response, "message", None)
                text    = (msg_obj.content if msg_obj and hasattr(msg_obj, "content") else "") or ""
        else:
            # Llama and other standard models — generate endpoint, no think option.
            response = ollama.generate(
                model   = model,
                prompt  = prompt,
                options = {"num_predict": 150, "temperature": temperature},
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
    """Pull a usable recommendation out of a Qwen thinking block.

    Qwen 3.5:4b in Ollama (this build) always runs its thinking process and
    exhausts num_predict before emitting content.  This function is therefore
    the primary path for all Qwen responses, not just an edge-case fallback.

    Extraction priority (highest → lowest):
    1. Last '*Draft N:*' block that ends with '- Good' or '(2 sentences)'
    2. Last explicit final-answer marker (Final Decision: / Final Answer: etc.)
    3. Last numbered list item with ≥ 20 chars of actual text
    4. Last sentence of ≥ 20 chars

    Cleans bold/italic markdown before returning. Returns '' if nothing ≥ 20 chars.
    """
    import re

    if not thinking:
        return ""

    # Phrases that indicate the text is still in Qwen's analysis section,
    # not in the final recommendation.  Any candidate containing one of these
    # is rejected so the caller falls through to the template fallback.
    _ANALYSIS_MARKERS = (
        "* Constraint", "* Task", "* Role",
        "Analyze the Request", "**",
    )

    def _clean(s: str) -> str:
        s = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", s)   # **bold** / *italic*
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _is_valid_candidate(text: str) -> bool:
        """Return False if text looks like analysis/constraint prose, not a recommendation."""
        if len(text) < 30:
            return False
        if text.lstrip().startswith(("*", "-")):
            return False
        for marker in _ANALYSIS_MARKERS:
            if marker in text:
                return False
        return True

    # ── Priority 1: Named candidate blocks ───────────────────────────────────
    # Qwen uses varying labels: *Draft N:*, *Idea N:*, *Option N:*, etc.
    # Split on any such marker and treat each segment as a candidate.
    candidate_segments = re.split(
        r"\*(?:Draft|Idea|Option|Attempt|Version|Suggestion)\s*\d*:\*\s*",
        thinking, flags=re.IGNORECASE,
    )
    complete_candidates: list[str] = []
    for segment in candidate_segments[1:]:      # [0] is pre-candidate preamble
        # Cut at next named-candidate marker or critique/meta section.
        # Use [\*\s]* to handle both "*Critique" and "*   *Critique" list formats.
        cut = re.search(
            r"\n\s*[\*\s]*(?:Critique|Note|Draft|Idea|Option|Attempt|Assess|Refin|Final)",
            segment, re.IGNORECASE,
        )
        if cut:
            segment = segment[: cut.start()]
        # Strip trailing markdown list artifacts (e.g. "\n    *   " left by splitter)
        segment = re.sub(r"[\s*]+$", "", segment)
        # Strip trailing parenthetical meta-commentary: "(Too vague?)", "- Good", etc.
        # The trailing punctuation may fall outside the closing paren, hence [.!?]?
        segment = re.sub(r"\s*\([^)]{0,80}\)[.!?]?\s*$", "", segment)
        segment = re.sub(
            r"\s*-?\s*(?:Good|OK|Great|Too generic|Too vague)[.!]?\s*$",
            "", segment, flags=re.IGNORECASE,
        )
        candidate = _clean(segment)
        # Only accept segments that end with sentence-closing punctuation
        # and do not contain analysis/constraint prose.
        if re.search(r"[.!?]$", candidate) and _is_valid_candidate(candidate):
            complete_candidates.append(candidate)
    # Prefer the last complete candidate (most refined in Qwen's thinking chain)
    for candidate in reversed(complete_candidates):
        if len(candidate) <= 500:
            return candidate

    # ── Priority 2: Explicit final-answer markers ─────────────────────────────
    final_markers = [
        "Final Polish:", "Final Decision:", "Final Answer:",
        "Revised:", "Final Recommendation:",
    ]
    best_pos, best_len = -1, 0
    for marker in final_markers:
        pos = thinking.rfind(marker)
        if pos > best_pos:
            best_pos, best_len = pos, len(marker)

    if best_pos >= 0:
        after = thinking[best_pos + best_len:].strip()
        first_para = re.split(r"\n\s*\n|\n(?=[A-Z0-9*#])", after)[0]
        candidate = _clean(first_para)
        if _is_valid_candidate(candidate) and len(candidate) <= 500:
            return candidate

    # ── Priority 3: Last numbered list item that ends with sentence punctuation ──
    items = re.findall(r"\d+\.\s+([A-Z].{19,})", thinking)
    for raw in reversed(items):
        candidate = _clean(raw.split("\n")[0])  # first line only
        if re.search(r"[.!?]$", candidate) and _is_valid_candidate(candidate) and len(candidate) <= 500:
            return candidate

    # ── Priority 4: Last complete sentence (ends with punctuation) ───────────
    sentences = [s.strip() for s in re.split(r"[.!?]+", thinking) if s.strip()]
    for sentence in reversed(sentences):
        candidate = _clean(sentence)
        original_end = thinking[thinking.rfind(sentence) + len(sentence):]
        if (re.match(r"^[.!?]", original_end)
                and _is_valid_candidate(candidate)
                and len(candidate) <= 500):
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

    variety_category = ctx.get("_variety_category", "")
    recent           = ctx.get("_recent_responses", [])

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

    variety_line = (
        f"Focus your recommendation on: {variety_category} strategies.\n"
        if variety_category else ""
    )
    avoid_line = ""
    if recent:
        recent_str = " | ".join(recent)[:200]
        avoid_line = f"IMPORTANT: Do NOT repeat or closely paraphrase these recent recommendations: {recent_str}\n"

    return (
        "System: You are a well-being assistant for a knowledge worker.\n"
        "Generate ONE short, friendly recommendation (max 2 sentences).\n"
        f"{guidance}\n"
        f"{variety_line}"
        f"{avoid_line}"
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
    """Strip thinking artifacts, preamble phrases, and surrounding quotes.

    Returns None when the cleaned text is shorter than 10 characters
    (indicates the model produced an empty or degenerate response).
    """
    if not raw:
        return None

    # Remove any reasoning/thinking blocks the model may have emitted.
    text = _strip_thinking_artifacts(raw).strip()

    # Remove common LLM-generated preamble phrases (case-insensitive prefix match).
    _PREAMBLES = [
        "Here's a friendly recommendation:",
        "Here is a friendly recommendation:",
        "Here's my recommendation:",
        "Here's a recommendation:",
        "Here is a recommendation:",
        "Here's a suggestion:",
        "Here is a suggestion:",
        "Recommendation:",
        "Sure!",
        "Sure,",
    ]
    for preamble in _PREAMBLES:
        if text.lower().startswith(preamble.lower()):
            text = text[len(preamble):].strip()
            break  # only strip one preamble

    # Remove surrounding quotation marks added by some models.
    if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        text = text[1:-1].strip()
    # Also strip a leading quote with no matching closing quote (partial wrap).
    elif text.startswith('"') or text.startswith("'"):
        text = text[1:].strip()

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
