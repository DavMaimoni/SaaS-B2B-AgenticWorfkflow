import json

from finetune.build_sft import clean_answer, to_example
from finetune.train_lora import tokenize
from support_agent.monitoring import alerts, kpis, log_run, load_runs
from support_agent.nodes import _parse_sft

TICKET = {"id": "T1", "subject": "Invoice", "body": "Please send the invoice.", "queue": "Billing and Payments",
          "priority": "low", "type": "Request",
          "answer": "Thanks for reaching out. Call us at <tel_num> anytime. You can download invoices in Settings."}


def test_sft_example_matches_the_runtime_format():
    ex = to_example(TICKET)
    assert "<tel_num>" not in ex["messages"][-1]["content"]
    assert ex["messages"][0]["content"].startswith("TASK: triage_and_answer")
    labels, answer = _parse_sft(ex["messages"][-1]["content"])      # the graph can read what the LoRA learns
    assert labels == {"queue": "Billing and Payments", "priority": "low", "type": "Request"} and answer
    assert clean_answer("Call <tel_num>.") is None


class FakeTokenizer:
    eos_token = "</s>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return " ".join(m["content"] for m in messages) + " ASSISTANT:"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [len(w) for w in text.split()]}


def test_loss_is_masked_on_the_prompt():
    ex = to_example(TICKET)
    out = tokenize(ex, FakeTokenizer(), max_len=4096)
    n_answer = len(FakeTokenizer()(ex["messages"][-1]["content"] + "</s>")["input_ids"])
    assert out["labels"][-n_answer:] == out["input_ids"][-n_answer:]
    assert set(out["labels"][:-n_answer]) == {-100}
    assert len(tokenize(ex, FakeTokenizer(), max_len=40)["input_ids"]) <= 40


def test_monitoring_kpis_and_alerts(tmp_path):
    state = {"status": "answered", "model_used": "small", "triage": {"queue": "Q"}, "cache_hit": False,
             "metrics": [{"node": "triage", "model": "small", "latency_ms": 900.0, "tokens_in": 10, "tokens_out": 5,
                          "cost_usd": 0.0, "llm_calls": 2}]}
    path = tmp_path / "m.jsonl"
    log_run(state, "test", "t1", path=path)
    log_run({**state, "status": "escalated", "model_used": "large"}, "test", "t2", path=path)
    k = kpis(load_runs(path))
    assert k["tickets"] == 2 and k["escalation_rate"] == 0.5 and k["small_model_share"] == 0.5
    assert any("escalation rate" in a for a in alerts(k, {"escalation_rate": 0.3}))
    assert json.loads(path.read_text().splitlines()[0])["nodes"][0]["node"] == "triage"
