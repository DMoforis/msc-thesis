"""
recommender.py
--------------
Contextual well-being recommendation generation via Ollama.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Design
------
  Primary:  Ollama (Llama 3.1 8B) — single-shot prompt engineered for
            a short, friendly, context-aware recommendation.
            2-second timeout (same as classifier).
  Fallback: Template-based messages selected by trigger type.
            Activates when Ollama is not running or exceeds the timeout.

Output
------
  Always a non-empty string ≤ 200 characters suitable for a Windows
  desktop notification.

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
import concurrent.futures

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.config import OLLAMA_MODEL, OLLAMA_TIMEOUT

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

    Primary path:  Ollama (Llama 3.1 8B) with 2-second timeout.
    Fallback path: Rotating template messages.

    The same _ollama_ok probe strategy as WindowClassifier is used:
      - None  = not yet probed
      - True  = Ollama available (continues trying)
      - False = disabled for this session (avoids repeated latency)
    """

    def __init__(
        self,
        model:   str   = OLLAMA_MODEL,
        timeout: float = OLLAMA_TIMEOUT,
    ):
        self._model    = model
        self._timeout  = timeout
        self._ollama_ok: bool | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ollama-rec"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, context: dict) -> str:
        """
        Return a well-being recommendation string (≤ 200 characters).

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
            msg = self._generate_ollama(context)
            if msg:
                return msg[:200]

        return _template_message(trigger)

    def close(self) -> None:
        """Shut down the thread pool cleanly."""
        self._executor.shutdown(wait=False)

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
        """Call Ollama with a timeout. Return None on failure."""
        prompt = _build_prompt(context)
        try:
            future   = self._executor.submit(self._ollama_call, prompt)
            response = future.result(timeout=self._timeout)
            return _clean_response(response)

        except concurrent.futures.TimeoutError:
            return None

        except Exception as exc:
            print(f"[Recommender] Ollama error: {exc}. Switching to template fallback.")
            self._ollama_ok = False
            return None

    def _ollama_call(self, prompt: str) -> str:
        """Blocking Ollama call — runs in thread pool."""
        import ollama
        response = ollama.generate(
            model   = self._model,
            prompt  = prompt,
            options = {"num_predict": 80, "temperature": 0.7},
        )
        if isinstance(response, dict):
            return response.get("response", "")
        return getattr(response, "response", "")


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT + RESPONSE HELPERS (module-level, no state)
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(ctx: dict) -> str:
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


def _clean_response(raw: str) -> str | None:
    """Strip leading/trailing whitespace and unwanted prefixes."""
    text = raw.strip()
    # Remove common prefixes the LLM sometimes adds
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
