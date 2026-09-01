"""
test_evaluator.py
-----------------
Tests for src/evaluation/llm_evaluator.py.

Coverage
--------
  1. test_ollama_judge_generates_response
     Mock ollama.chat; verify OllamaJudge.generate() returns a string.

  2. test_evaluate_comparisons_empty_db
     Use tmp_db fixture; create the llm_comparisons table but leave it
     empty; verify evaluate_comparisons() returns an empty DataFrame
     without crashing or calling any LLM.

  3. test_winner_calculation
     Verify _determine_winner() for three score combinations:
       llama=0.85, qwen=0.72  → "llama"
       llama=0.70, qwen=0.85  → "qwen"
       llama=0.80, qwen=0.82  → "tie" (diff < 0.05)

  4. test_none_response_scores_zero
     If a model's response is None, _evaluate_one_response() must
     return (0.0, 0.0, 0.0) without calling any metric.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Test 1 ────────────────────────────────────────────────────────────────────

class TestOllamaJudge(unittest.TestCase):

    def test_ollama_judge_generates_response(self):
        """OllamaJudge.generate() returns the model's response as a string."""
        from src.evaluation.llm_evaluator import OllamaJudge
        import ollama as _ollama_mod

        expected_text = "This is a judge evaluation response."

        def fake_chat(**kwargs):
            msg      = MagicMock()
            msg.content = expected_text
            resp     = MagicMock()
            resp.message = msg
            return resp

        with patch.object(_ollama_mod, "chat", side_effect=fake_chat) as mock_chat:
            judge  = OllamaJudge(model_name="gemma4:12b")
            result = judge.generate("Evaluate this recommendation.")

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemma4:12b")
        # think=False must be present in options
        self.assertFalse(call_kwargs["options"]["think"])
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected_text)

    def test_ollama_judge_get_model_name(self):
        """OllamaJudge.get_model_name() returns the model identifier."""
        from src.evaluation.llm_evaluator import OllamaJudge
        judge = OllamaJudge(model_name="gemma4:12b")
        self.assertEqual(judge.get_model_name(), "gemma4:12b")


# ── Test 2 ────────────────────────────────────────────────────────────────────

class TestEvaluateComparisonsEmptyDB(unittest.TestCase):

    def test_evaluate_comparisons_empty_db(self, tmp_path=None):
        """evaluate_comparisons() returns empty DataFrame for an empty table."""
        import tempfile
        import sqlite3
        from src.evaluation.llm_evaluator import evaluate_comparisons
        from src.utils.db import ensure_llm_comparisons_table, open_db

        # Create a fresh DB with the llm_comparisons table but no rows.
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            conn = open_db(db_path)
            ensure_llm_comparisons_table(conn)
            conn.close()

            df = evaluate_comparisons(db_path=db_path)

            self.assertTrue(df.empty, "Expected empty DataFrame for an empty table")
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    def test_evaluate_comparisons_missing_table(self):
        """evaluate_comparisons() returns empty DataFrame when table is absent."""
        import tempfile
        from src.evaluation.llm_evaluator import evaluate_comparisons
        from src.utils.db import open_db

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            # Open (creates the file) but don't create llm_comparisons.
            conn = open_db(db_path)
            conn.close()

            df = evaluate_comparisons(db_path=db_path)
            self.assertTrue(df.empty)
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass


# ── Test 3 ────────────────────────────────────────────────────────────────────

class TestWinnerCalculation(unittest.TestCase):

    def setUp(self):
        from src.evaluation.llm_evaluator import _determine_winner
        self._winner = _determine_winner

    def test_llama_wins(self):
        self.assertEqual(self._winner(0.85, 0.72), "llama")

    def test_qwen_wins(self):
        self.assertEqual(self._winner(0.70, 0.85), "qwen")

    def test_tie_when_diff_below_threshold(self):
        """Difference of 0.02 is below the 0.05 threshold → tie."""
        self.assertEqual(self._winner(0.80, 0.82), "tie")

    def test_tie_exact_threshold(self):
        """Difference exactly equal to threshold → tie (< not <=)."""
        self.assertEqual(self._winner(0.80, 0.85), "tie")

    def test_tie_both_none(self):
        self.assertEqual(self._winner(None, None), "tie")

    def test_qwen_wins_when_llama_none(self):
        self.assertEqual(self._winner(None, 0.80), "qwen")

    def test_llama_wins_when_qwen_none(self):
        self.assertEqual(self._winner(0.80, None), "llama")


# ── Test 4 ────────────────────────────────────────────────────────────────────

class TestNoneResponseScoresZero(unittest.TestCase):

    def test_none_response_returns_triple_zero(self):
        """_evaluate_one_response(None) returns (0.0, 0.0, 0.0) without
        calling any metric."""
        from src.evaluation.llm_evaluator import _evaluate_one_response

        sentinel = MagicMock()  # will raise if called
        result = _evaluate_one_response(
            response   = None,
            input_text = "Trigger: high_stress. Context: {}",
            metrics    = [sentinel, sentinel, sentinel],
        )

        sentinel.measure.assert_not_called()
        self.assertEqual(result, (0.0, 0.0, 0.0))

    def test_non_none_response_calls_metrics(self):
        """A real response string triggers metric.measure() for each metric."""
        from src.evaluation.llm_evaluator import _evaluate_one_response

        mock_metric = MagicMock()
        mock_metric.score = 0.9

        result = _evaluate_one_response(
            response   = "Take a short walk to reset your focus.",
            input_text = "Trigger: high_stress. Context: {}",
            metrics    = [mock_metric, mock_metric, mock_metric],
        )

        self.assertEqual(mock_metric.measure.call_count, 3)
        self.assertEqual(result, (0.9, 0.9, 0.9))


if __name__ == "__main__":
    unittest.main()
