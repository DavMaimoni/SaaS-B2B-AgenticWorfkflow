import json

import pandas as pd
import pytest

from support_agent.config import DATA_DIR, ModelSpec
from support_agent.data import load_raw, normalize, stratified_split
from support_agent.llm import LLM, ReplayMiss, parse_json


def test_normalize_drops_unknown_labels_duplicates_and_other_languages():
    df = pd.DataFrame([
        {"subject": "A", "body": "My invoice is wrong, please fix it.", "answer": "We will fix the invoice for you.",
         "queue": "Billing and Payments", "priority": "Critical", "type": "problem", "language": "en"},
        {"subject": "A", "body": "My invoice is wrong, please fix it.", "answer": "We will fix the invoice for you.",
         "queue": "Billing and Payments", "priority": "high", "type": "Problem", "language": "en"},
        {"subject": "B", "body": "Meine Rechnung ist falsch, bitte korrigieren.", "answer": "Wir korrigieren das.",
         "queue": "Billing and Payments", "priority": "low", "type": "Problem", "language": "de"},
        {"subject": "C", "body": "Some unrelated queue that we do not handle.", "answer": "Not our queue at all.",
         "queue": "Arts and Entertainment", "priority": "low", "type": "Request", "language": "en"},
    ])
    out, report = normalize(df)
    assert len(out) == 1 and out.iloc[0]["priority"] == "high" and out.iloc[0]["type"] == "Problem"
    assert report["dropped_duplicates"] == 1 and report["dropped_language"] == 1
    assert report["unknown_queues"] == {"Arts and Entertainment": 1}


def test_split_keeps_every_queue_in_test_and_has_no_overlap():
    df, _ = normalize(load_raw(DATA_DIR / "sample_tickets.jsonl"))
    parts = stratified_split(df, seed=1)
    ids = [set(p["id"]) for p in parts.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(p) for p in parts.values()) == len(df)


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


def test_record_then_replay(tmp_path, monkeypatch):
    spec = ModelSpec("large", "openrouter", "some/model", "http://x", None, 1.0, 2.0)
    calls = []

    def fake_call(self, messages, temperature, max_tokens, json_mode, sample):
        calls.append(messages)
        return json.dumps({"answer": "recorded"}), 100, 20
    monkeypatch.setattr(LLM, "_call_api", fake_call)
    msgs = [{"role": "system", "content": "TASK: answer"}, {"role": "user", "content": "hi"}]
    rec = LLM(spec, mode="record", recordings_dir=tmp_path)
    first = rec.chat(msgs)
    assert first.cost_usd == pytest.approx((100 * 1.0 + 20 * 2.0) / 1e6)
    replay = LLM(spec, mode="replay", recordings_dir=tmp_path)
    assert replay.chat(msgs).text == first.text and len(calls) == 1
    with pytest.raises(ReplayMiss):
        replay.chat([{"role": "user", "content": "never recorded"}])
