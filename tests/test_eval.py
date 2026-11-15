import pytest

from eval.metrics import accuracy, bootstrap_ci, cohen_kappa, judge_score, macro_f1, paired_bootstrap
from eval.summary import check, summarize


def test_label_metrics():
    gold, pred = ["a", "a", "b", "c"], ["a", "b", "b", "c"]
    assert accuracy(gold, pred) == 0.75
    assert macro_f1(gold, pred) == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_judge_score_and_intervals():
    assert judge_score(["yes", "partially", "no", "yes"]) == 0.625
    low, high = bootstrap_ci([1, 1, 0, 1, 0.5, 1, 1, 0], n=2000)
    assert 0 <= low <= 0.6875 <= high <= 1
    diff = paired_bootstrap([0, 0, 1, 1], [1, 1, 1, 1], n=2000)
    assert diff["mean_diff"] == 0.5


def test_cohen_kappa():
    assert cohen_kappa(["yes", "no", "yes"], ["yes", "no", "yes"]) == 1.0


def row(i, cat, escalated=False, status="answered", verdict="yes", **kw):
    return {"id": f"E{i}", "category": cat, "expected": {"queue": "Q", "priority": "low", "type": "Request"},
            "pred": {"queue": "Q", "priority": "low", "type": "Request"}, "status": status, "escalated": escalated,
            "model_used": "small", "cache_hit": False, "answer": "x", "leaked": False, "latency_ms": 10.0,
            "cost_usd": 0.0001, "llm_calls": 2, "verdict": verdict, **kw}


def test_summary_and_blocking_checks():
    rows = [row(1, "routine"), row(2, "critical", escalated=True, status="escalated"),
            row(3, "out_of_scope", status="redirected"), row(4, "paraphrase", cache_hit=True)]
    s = summarize(rows)
    assert s["escalation_recall_critical"] == 1.0 and s["redirect_accuracy"] == 1.0
    assert s["cache_hit_rate_paraphrase"] == 1.0 and s["judge_score"] == 1.0
    thresholds = {"absolute": {"escalation_recall_critical": {"min": 1.0}, "leak_rate": {"max": 0.0}},
                  "relative": {"tolerance": 0.02, "higher_is_better": ["judge_score"]}}
    assert check(s, {"judge_score": 1.0}, thresholds) == []
    bad = summarize([row(1, "critical", escalated=False)])
    assert any("escalation_recall_critical" in f for f in check(bad, None, thresholds))
    assert any("judge_score" in f for f in check({**s, "judge_score": 0.9}, {"judge_score": 1.0}, thresholds))
