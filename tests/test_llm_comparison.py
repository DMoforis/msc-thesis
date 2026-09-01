"""
test_llm_comparison.py
----------------------
Tests for the dual-model LLM comparison feature.

Coverage
--------
  1. test_both_models_called_in_comparison_mode
     When LLM_COMPARISON_MODE is True, both Llama and Qwen are called
     for the same trigger context.

  2. test_single_model_used_when_comparison_off
     When LLM_COMPARISON_MODE is False, only the primary model is called.

  3. test_failed_secondary_model_does_not_block_primary
     A Qwen failure (exception) does not prevent the primary recommendation
     from being returned.

  4. test_comparison_saved_to_db
     After a dual-model call the llm_comparisons table contains one row
     with the expected columns.

  5. test_strip_thinking_artifacts_think_tags
     <think>…</think> blocks are removed from the raw response.

  6. test_strip_thinking_artifacts_thinking_lines
     Lines beginning with "Thinking..." / "Thinking Process:" are removed.

  7. test_strip_thinking_artifacts_clean_passthrough
     Clean responses pass through _strip_thinking_artifacts unchanged.

  8. test_qwen_think_false_option_passed
     When _timed_ollama_call is invoked with a qwen3.5 model, it calls
     ollama.chat with options={"think": False} (not ollama.generate).
"""

import sys
import os
import sqlite3
import time
import threading
import unittest
from unittest.mock import patch, MagicMock

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_context(**overrides) -> dict:
    base = {
        "app_category":        "Software Development",
        "active_window":       "VS Code — main.py",
        "stress_index":        0.70,
        "valence":             -0.30,
        "arousal":              0.50,
        "session_minutes":     60,
        "minutes_since_break": 35,
        "blink_rate":          11.5,
        "trigger_reason":      "high_stress",
    }
    base.update(overrides)
    return base


def _in_memory_conn():
    """Open a fresh :memory: SQLite connection for test isolation."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMComparisonMode(unittest.TestCase):

    # ── Test 1 ────────────────────────────────────────────────────────────────

    def test_both_models_called_in_comparison_mode(self):
        """Both Llama and Qwen are invoked when LLM_COMPARISON_MODE is True."""
        from src.llm.recommender import WellbeingRecommender

        called_models: list[str] = []

        def fake_timed_call(model: str, prompt: str) -> tuple[str, float]:
            called_models.append(model)
            return f"Take a short break. ({model})", 120.0

        with patch("src.llm.recommender.LLM_COMPARISON_MODE", True), \
             patch("src.llm.recommender._save_comparison_row"):

            rec = WellbeingRecommender(timeout=5, db_path=":memory:")
            rec._ollama_ok = True  # skip import probe
            rec._timed_ollama_call = fake_timed_call  # type: ignore[method-assign]

            result = rec._generate_with_comparison(_make_context())

            # Give the background thread time to call Qwen
            time.sleep(0.2)

        # Primary response returned
        self.assertIsNotNone(result)
        self.assertIn("Take a short break", result)

        # Both models must have been called
        from src.utils.config import OLLAMA_MODEL_PRIMARY, OLLAMA_MODEL_SECONDARY
        self.assertIn(OLLAMA_MODEL_PRIMARY,   called_models,
                      "Primary model (Llama) was not called")
        self.assertIn(OLLAMA_MODEL_SECONDARY, called_models,
                      "Secondary model (Qwen) was not called")

    # ── Test 2 ────────────────────────────────────────────────────────────────

    def test_single_model_used_when_comparison_off(self):
        """Only the primary model is invoked when LLM_COMPARISON_MODE is False."""
        from src.llm.recommender import WellbeingRecommender
        from src.utils.config import OLLAMA_MODEL_PRIMARY, OLLAMA_MODEL_SECONDARY

        called_models: list[str] = []

        def fake_timed_call(model: str, prompt: str) -> tuple[str, float]:
            called_models.append(model)
            return "Stretch and breathe.", 90.0

        with patch("src.llm.recommender.LLM_COMPARISON_MODE", False):
            rec = WellbeingRecommender(timeout=5, db_path=":memory:")
            rec._ollama_ok = True
            rec._timed_ollama_call = fake_timed_call  # type: ignore[method-assign]

            msg = rec.generate(_make_context())

        self.assertTrue(len(msg) > 0)
        self.assertIn(OLLAMA_MODEL_PRIMARY, called_models,
                      "Primary model was not called")
        self.assertNotIn(OLLAMA_MODEL_SECONDARY, called_models,
                         "Secondary model was called in single-model mode")

    # ── Test 3 ────────────────────────────────────────────────────────────────

    def test_failed_secondary_model_does_not_block_primary(self):
        """A Qwen error does not suppress the primary recommendation."""
        from src.llm.recommender import WellbeingRecommender
        from src.utils.config import OLLAMA_MODEL_PRIMARY, OLLAMA_MODEL_SECONDARY

        def fake_timed_call(model: str, prompt: str) -> tuple[str, float]:
            if model == OLLAMA_MODEL_SECONDARY:
                raise RuntimeError("Qwen model not found")
            return "Step outside for a moment.", 115.0

        with patch("src.llm.recommender.LLM_COMPARISON_MODE", True), \
             patch("src.llm.recommender._save_comparison_row"):

            rec = WellbeingRecommender(timeout=5, db_path=":memory:")
            rec._ollama_ok = True
            rec._timed_ollama_call = fake_timed_call  # type: ignore[method-assign]

            result = rec._generate_with_comparison(_make_context())

            # Let the background thread attempt (and fail) its Qwen call
            time.sleep(0.2)

        # Primary response must still be returned despite secondary failure
        self.assertIsNotNone(result, "Primary response is None after secondary failure")
        self.assertIn("Step outside", result)

    # ── Test 4 ────────────────────────────────────────────────────────────────

    def test_comparison_saved_to_db(self):
        """After a dual-model call, one row appears in llm_comparisons."""
        from src.llm.recommender import WellbeingRecommender, _save_comparison_row
        from src.utils.db import ensure_llm_comparisons_table

        # Use an in-memory DB shared between the recommender and the assertion.
        mem_conn = _in_memory_conn()
        ensure_llm_comparisons_table(mem_conn)
        mem_conn.close()

        # Use a real on-disk file so the daemon thread can open it by path.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            # Initialise schema
            conn = sqlite3.connect(db_path)
            ensure_llm_comparisons_table(conn)
            conn.close()

            from src.utils.config import OLLAMA_MODEL_PRIMARY, OLLAMA_MODEL_SECONDARY

            def fake_timed_call(model: str, prompt: str) -> tuple[str, float]:
                if model == OLLAMA_MODEL_PRIMARY:
                    return "Take a walk.", 100.0
                return "Breathe deeply.", 200.0

            with patch("src.llm.recommender.LLM_COMPARISON_MODE", True):
                rec = WellbeingRecommender(timeout=5, db_path=db_path)
                rec._ollama_ok = True
                rec._timed_ollama_call = fake_timed_call  # type: ignore[method-assign]

                rec._generate_with_comparison(_make_context())

                # Wait for the background thread to finish saving
                time.sleep(0.5)

            # Verify the row was written
            conn2 = sqlite3.connect(db_path)
            rows = conn2.execute("SELECT * FROM llm_comparisons").fetchall()
            conn2.close()

            self.assertEqual(len(rows), 1, "Expected exactly one comparison row")

            row = rows[0]
            col_names = [d[0] for d in conn2.execute(
                "SELECT * FROM llm_comparisons LIMIT 0"
            ).description] if False else None  # description not available after close

            # Re-open to get column names
            conn3 = sqlite3.connect(db_path)
            conn3.row_factory = sqlite3.Row
            row_dict = dict(conn3.execute("SELECT * FROM llm_comparisons").fetchone())
            conn3.close()

            self.assertIsNotNone(row_dict["llama_response"],
                                 "llama_response should not be None")
            self.assertIsNotNone(row_dict["qwen_response"],
                                 "qwen_response should not be None")
            self.assertAlmostEqual(row_dict["llama_latency_ms"], 100.0, places=0)
            self.assertAlmostEqual(row_dict["qwen_latency_ms"],  200.0, places=0)
            self.assertEqual(row_dict["trigger_reason"], "high_stress")

        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass


class TestStripThinkingArtifacts(unittest.TestCase):
    """Unit tests for the _strip_thinking_artifacts safety-net function."""

    def setUp(self):
        from src.llm.recommender import _strip_thinking_artifacts
        self._strip = _strip_thinking_artifacts

    # ── Test 5 ────────────────────────────────────────────────────────────────

    def test_strip_thinking_artifacts_think_tags(self):
        """<think>…</think> blocks are stripped from the raw response."""
        raw = (
            "<think>The user appears stressed after a long session. "
            "I should suggest a break involving physical movement.</think>\n"
            "Take a short walk — even five minutes outside will reset your focus."
        )
        result = self._strip(raw)
        self.assertNotIn("<think>", result)
        self.assertNotIn("</think>", result)
        self.assertNotIn("I should suggest", result)
        self.assertIn("Take a short walk", result)

    def test_strip_thinking_artifacts_think_tags_multiline(self):
        """Multiline <think>…</think> blocks are fully removed."""
        raw = (
            "<think>\n"
            "Step 1: analyse the context.\n"
            "Step 2: determine appropriate intervention.\n"
            "</think>\n"
            "Try box breathing: inhale 4s, hold 4s, exhale 4s."
        )
        result = self._strip(raw)
        self.assertNotIn("<think>", result)
        self.assertNotIn("Step 1:", result)
        self.assertIn("Try box breathing", result)

    # ── Test 6 ────────────────────────────────────────────────────────────────

    def test_strip_thinking_artifacts_thinking_lines(self):
        """Lines beginning with 'Thinking...' or 'Thinking Process:' are removed."""
        raw = (
            "Thinking...\n"
            "Thinking Process: evaluate stress level and suggest action.\n"
            "Take a 5-minute break and stretch your shoulders."
        )
        result = self._strip(raw)
        self.assertNotIn("Thinking...", result)
        self.assertNotIn("Thinking Process:", result)
        self.assertIn("Take a 5-minute break", result)

    # ── Test 7 ────────────────────────────────────────────────────────────────

    def test_strip_thinking_artifacts_clean_passthrough(self):
        """A response with no artifacts passes through unchanged."""
        clean = "Stand up, roll your shoulders back, and take three slow breaths."
        result = self._strip(clean)
        self.assertEqual(result, clean)

    def test_strip_thinking_artifacts_empty_after_strip(self):
        """A response that is entirely a think block becomes an empty string."""
        raw = "<think>Just reasoning, no recommendation.</think>"
        result = self._strip(raw)
        self.assertEqual(result, "")


class TestQwenAPIOption(unittest.TestCase):
    """Verify that _timed_ollama_call routes Qwen through ollama.chat with think=False."""

    # ── Test 8 ────────────────────────────────────────────────────────────────

    def test_qwen_think_false_option_passed(self):
        """ollama.chat is called with options={'think': False} for qwen3.5 models."""
        from src.llm.recommender import WellbeingRecommender

        chat_calls: list[dict] = []

        def fake_chat(**kwargs):
            chat_calls.append(kwargs)
            # Return a minimal chat response object
            msg = MagicMock()
            msg.content = "Take a short break."
            resp = MagicMock()
            resp.message = msg
            return resp

        import ollama as _ollama_mod

        with patch.object(_ollama_mod, "chat", side_effect=fake_chat) as mock_chat, \
             patch.object(_ollama_mod, "generate") as mock_generate:

            rec = WellbeingRecommender(timeout=5)
            text, latency = rec._timed_ollama_call("qwen3.5:4b", "Some prompt")

            # ollama.chat must have been called, not generate
            mock_chat.assert_called_once()
            mock_generate.assert_not_called()

            # The think=False option must be present
            call_kwargs = chat_calls[0]
            self.assertIn("options", call_kwargs)
            self.assertFalse(
                call_kwargs["options"]["think"],
                "think option must be False for qwen3.5"
            )
            # Correct model forwarded
            self.assertEqual(call_kwargs["model"], "qwen3.5:4b")

        self.assertEqual(text, "Take a short break.")
        self.assertGreaterEqual(latency, 0.0)

    def test_llama_uses_generate_not_chat(self):
        """ollama.generate is called (not chat) for llama3.1 models."""
        from src.llm.recommender import WellbeingRecommender

        import ollama as _ollama_mod

        def fake_generate(**kwargs):
            resp = MagicMock()
            resp.response = "Step outside for a moment."
            return resp

        with patch.object(_ollama_mod, "generate", side_effect=fake_generate) as mock_gen, \
             patch.object(_ollama_mod, "chat") as mock_chat:

            rec = WellbeingRecommender(timeout=5)
            text, latency = rec._timed_ollama_call("llama3.1:8b", "Some prompt")

            mock_gen.assert_called_once()
            mock_chat.assert_not_called()

        self.assertEqual(text, "Step outside for a moment.")


if __name__ == "__main__":
    unittest.main()
