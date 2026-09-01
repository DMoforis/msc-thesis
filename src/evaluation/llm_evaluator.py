"""
llm_evaluator.py
----------------
Post-hoc evaluation of LLM recommendation quality using DeepEval metrics.
Compares Llama 3.1:8b vs Qwen 3.5:4b responses stored in llm_comparisons.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Judge
-----
  Gemma 4:12b via local Ollama — no external API calls, no OpenAI key needed.
  Override with --model or the OLLAMA_MODEL_JUDGE config constant.

Metrics (per response)
----------------------
  1. AnswerRelevancyMetric — is the recommendation relevant to the stress context?
  2. WellbeingAppropriateness (GEval) — appropriate tone, actionable, concise?
  3. ContextAwareness (GEval) — does it reference the trigger, activity, mood?

Output
------
  DataFrame with per-row scores for both models.
  Saved to data/exports/llm_evaluation_YYYY-MM-DD.xlsx.

Usage
-----
  python src/evaluation/llm_evaluator.py --limit 20 --verbose
  python src/evaluation/llm_evaluator.py --model gemma4:12b
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.config import DB_PATH, EXPORT_DIR, OLLAMA_MODEL_JUDGE

# ── Ollama judge backend ──────────────────────────────────────────────────────

from deepeval.models.base_model import DeepEvalBaseLLM
import ollama as _ollama_client


class OllamaJudge(DeepEvalBaseLLM):
    """Local Ollama model used as the independent DeepEval judge.

    Passes ``think=False`` so chain-of-thought is suppressed for models
    that support it (e.g. Gemma 4 with thinking mode).
    """

    def __init__(self, model_name: str = OLLAMA_MODEL_JUDGE):
        self.model_name = model_name

    def load_model(self) -> str:
        return self.model_name

    def generate(self, prompt: str, *args, **kwargs) -> str:
        response = _ollama_client.chat(
            model    = self.model_name,
            messages = [{"role": "user", "content": prompt}],
            options  = {"think": False},
        )
        return response.message.content

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


# ── Winner calculation ────────────────────────────────────────────────────────

def _determine_winner(
    llama_avg:     Optional[float],
    qwen_avg:      Optional[float],
    tie_threshold: float = 0.05,
) -> str:
    """Return 'llama', 'qwen', or 'tie' from two average scores.

    A tie is declared when the absolute difference is below tie_threshold.
    None scores (evaluation failure) lose against any valid score.
    """
    if llama_avg is None and qwen_avg is None:
        return "tie"
    if llama_avg is None:
        return "qwen"
    if qwen_avg is None:
        return "llama"
    if abs(llama_avg - qwen_avg) < tie_threshold:
        return "tie"
    return "llama" if llama_avg > qwen_avg else "qwen"


# ── Per-response evaluation ───────────────────────────────────────────────────

def _evaluate_one_response(
    response:   Optional[str],
    input_text: str,
    metrics:    list,
    verbose:    bool = False,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Run all three metrics against one model response.

    Parameters
    ----------
    response   : the model's recommendation text (None → skip, return 0.0s)
    input_text : the trigger + context string used as the LLMTestCase input
    metrics    : list of [AnswerRelevancyMetric, WellbeingGEval, ContextGEval]
    verbose    : pass _show_indicator=True to deepeval for progress output

    Returns
    -------
    (relevancy, wellbeing, context_awareness) — each is float or None on error
    Returns (0.0, 0.0, 0.0) for a None response without calling any metric.
    """
    from deepeval.test_case import LLMTestCase

    if response is None:
        return 0.0, 0.0, 0.0

    test_case = LLMTestCase(input=input_text, actual_output=response)
    scores: list[Optional[float]] = []

    for metric in metrics:
        try:
            metric.measure(test_case, _show_indicator=verbose)
            scores.append(metric.score)
        except Exception as exc:
            print(
                f"[Evaluator] Warning: {metric.__class__.__name__} failed "
                f"({type(exc).__name__}: {exc})"
            )
            scores.append(None)

    # Always return a 3-tuple even if fewer metrics ran
    while len(scores) < 3:
        scores.append(None)

    return scores[0], scores[1], scores[2]


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate_comparisons(
    db_path:     Optional[str] = None,
    judge_model: Optional[str] = None,
    limit:       Optional[int] = None,
    verbose:     bool = False,
) -> "pd.DataFrame":
    """
    Read llm_comparisons rows and evaluate both model responses with DeepEval.

    Parameters
    ----------
    db_path     : path to stress_monitor.db (default: config.DB_PATH)
    judge_model : Ollama model to use as judge (default: config.OLLAMA_MODEL_JUDGE)
    limit       : evaluate only the last N rows (default: all)
    verbose     : print per-row scores as they compute

    Returns
    -------
    DataFrame with columns: timestamp, trigger_reason,
      llama_relevancy, llama_wellbeing, llama_context_awareness,
      qwen_relevancy, qwen_wellbeing, qwen_context_awareness,
      llama_avg_score, qwen_avg_score, llama_latency_ms, qwen_latency_ms, winner.

    Saves results to data/exports/llm_evaluation_YYYY-MM-DD.xlsx.
    Returns an empty DataFrame if the table is empty or missing.
    """
    import sqlite3
    import pandas as pd
    from deepeval.metrics import AnswerRelevancyMetric, GEval
    from deepeval.test_case import SingleTurnParams

    db_path     = db_path     or DB_PATH
    judge_model = judge_model or OLLAMA_MODEL_JUDGE

    # ── Load comparison rows ──────────────────────────────────────────────────
    if not os.path.exists(db_path):
        print(f"[Evaluator] Database not found: {db_path}")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='llm_comparisons'"
        ).fetchone()
        if not exists:
            print("[Evaluator] Table 'llm_comparisons' does not exist — nothing to evaluate.")
            return pd.DataFrame()

        query = "SELECT * FROM llm_comparisons ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()

    if not rows:
        print("[Evaluator] llm_comparisons table is empty — nothing to evaluate.")
        return pd.DataFrame()

    print(f"[Evaluator] Evaluating {len(rows)} row(s) — judge: {judge_model}")

    # ── Build metrics (created once, shared across all rows) ──────────────────
    judge = OllamaJudge(model_name=judge_model)

    metrics = [
        AnswerRelevancyMetric(
            threshold    = 0.7,
            model        = judge,
            async_mode   = False,
            verbose_mode = verbose,
        ),
        GEval(
            name               = "WellbeingAppropriateness",
            evaluation_params  = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            criteria           = (
                "Evaluate whether the recommendation is:\n"
                "1. Appropriate for a well-being assistant (not medical advice)\n"
                "2. Actionable and specific to the context provided\n"
                "3. Friendly and non-intrusive in tone\n"
                "4. Concise (maximum 2-3 sentences)\n"
                "Score 0-1 where 1 is perfect."
            ),
            threshold    = 0.7,
            model        = judge,
            async_mode   = False,
            verbose_mode = verbose,
        ),
        GEval(
            name               = "ContextAwareness",
            evaluation_params  = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            criteria           = (
                "Does the recommendation demonstrate awareness of:\n"
                "1. The specific trigger reason (high_stress/disengagement/"
                "eye_strain/negative_affect/prolonged_idle/positive_flow)\n"
                "2. The current desktop activity (app category)\n"
                "3. The emotional state (valence/arousal emotion label)\n"
                "Score 0-1 where 1 = highly context-aware."
            ),
            threshold    = 0.7,
            model        = judge,
            async_mode   = False,
            verbose_mode = verbose,
        ),
    ]

    # ── Evaluate each row ─────────────────────────────────────────────────────
    results = []

    for i, row in enumerate(rows, 1):
        row_dict   = dict(row)
        timestamp  = row_dict.get("timestamp", "")
        trigger    = row_dict.get("trigger_reason") or "unknown"
        llama_resp = row_dict.get("llama_response")
        qwen_resp  = row_dict.get("qwen_response")
        llama_ms   = row_dict.get("llama_latency_ms")
        qwen_ms    = row_dict.get("qwen_latency_ms")
        ctx_raw    = row_dict.get("context_json") or ""

        try:
            ctx_dict = json.loads(ctx_raw) if ctx_raw else {}
        except (json.JSONDecodeError, TypeError):
            ctx_dict = {}

        input_text = (
            f"Trigger: {trigger}. "
            f"Context: {json.dumps(ctx_dict) if ctx_dict else ctx_raw}"
        )

        if verbose:
            print(f"\n[Evaluator] Row {i}/{len(rows)}: {timestamp} — {trigger}")

        l_rel, l_wb, l_ctx = _evaluate_one_response(llama_resp, input_text, metrics, verbose)
        q_rel, q_wb, q_ctx = _evaluate_one_response(qwen_resp,  input_text, metrics, verbose)

        def _safe_avg(*scores: Optional[float]) -> Optional[float]:
            valid = [s for s in scores if s is not None]
            return round(sum(valid) / len(valid), 4) if valid else None

        llama_avg = _safe_avg(l_rel, l_wb, l_ctx)
        qwen_avg  = _safe_avg(q_rel, q_wb, q_ctx)

        results.append({
            "timestamp":               timestamp,
            "trigger_reason":          trigger,
            "llama_relevancy":         l_rel,
            "llama_wellbeing":         l_wb,
            "llama_context_awareness": l_ctx,
            "qwen_relevancy":          q_rel,
            "qwen_wellbeing":          q_wb,
            "qwen_context_awareness":  q_ctx,
            "llama_avg_score":         llama_avg,
            "qwen_avg_score":          qwen_avg,
            "llama_latency_ms":        llama_ms,
            "qwen_latency_ms":         qwen_ms,
            "winner":                  _determine_winner(llama_avg, qwen_avg),
        })

    df = pd.DataFrame(results)

    # ── Save to Excel ─────────────────────────────────────────────────────────
    os.makedirs(EXPORT_DIR, exist_ok=True)
    date_tag = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(EXPORT_DIR, f"llm_evaluation_{date_tag}.xlsx")
    df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"[Evaluator] Results saved → {out_path}")

    wins = df["winner"].value_counts().to_dict()
    print(
        f"[Evaluator] Summary — "
        f"Llama wins: {wins.get('llama', 0)}, "
        f"Qwen wins: {wins.get('qwen', 0)}, "
        f"Ties: {wins.get('tie', 0)}"
    )

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LLM comparison rows using DeepEval + Ollama judge."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the last N comparison rows (default: all)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Judge model override (default: {OLLAMA_MODEL_JUDGE})",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-row scores as they compute",
    )
    args = parser.parse_args()

    df = evaluate_comparisons(
        judge_model = args.model,
        limit       = args.limit,
        verbose     = args.verbose,
    )
    if not df.empty:
        print("\n" + df.to_string(index=False))
