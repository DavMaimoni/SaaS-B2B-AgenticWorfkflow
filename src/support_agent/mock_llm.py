"""An offline stand-in for the models: keyword rules, deterministic. It lets the tests, the CI smoke run and a demo
work without an API key or a GPU. Its scores are a floor (a rules baseline), not a result to report as the model's.
"""

import json
import re

from support_agent.prompts import CANARY

QUEUE_RULES = [
    ("Service Outages and Maintenance", r"\b(outage|down|unavailable|not loading|maintenance|status page|503|502)\b"),
    ("Billing and Payments", r"\b(invoice|billing|bill|charge[ds]?|payment|refund|credit card|vat|subscription fee|price|pricing)\b"),
    ("IT Support", r"\b(sso|saml|okta|azure ad|vpn|firewall|ip allow|network|proxy|scim)\b"),
    ("Technical Support", r"\b(api|error|bug|crash|integration|webhook|sync|token|rate limit|429|500|timeout|export)\b"),
    ("Product Support", r"\b(feature|dashboard|report|template|project|board|how do i|how to|configure|setting)\b"),
    ("Returns and Exchanges", r"\b(return|exchange|shipping|package|parcel)\b"),
    ("Sales and Pre-Sales", r"\b(quote|demo|trial|pricing for|enterprise plan|purchase|buy)\b"),
    ("Human Resources", r"\b(job|salary|vacation|recruit|hiring|payroll)\b"),
]
HIGH = r"\b(urgent|asap|immediately|critical|outage|down|breach|security incident|data loss|cannot access|all users)\b"
LOW = r"\b(question|information|curious|wondering|feedback|suggestion|when you have time)\b"
TYPE_RULES = [
    ("Incident", r"\b(outage|down|error|crash|breach|not working|failed|failing|unavailable)\b"),
    ("Change", r"\b(change|upgrade|downgrade|migrate|update my|switch|add seats|remove)\b"),
    ("Request", r"\b(how|can i|could you|would like|request|please send|need a|information)\b"),
]


def _ticket(user: str) -> str:
    text = user.split("Help-center articles")[0]
    if "Customer ticket:" in text:
        text = text.split("Customer ticket:")[1]
    return text


def _scores(text: str) -> list[tuple[int, str]]:
    low = text.lower()
    scored = [(len(re.findall(rx, low)), q) for q, rx in QUEUE_RULES]
    return sorted(scored, key=lambda x: -x[0])


def triage(text: str, sample: int = 0) -> dict:
    low = text.lower()
    scores = _scores(text)
    best = scores[0] if scores[0][0] > 0 else (0, "Customer Service")
    queue = best[1]
    # A second sample disagrees when two queues tie: that is what lowers the confidence.
    if sample and len(scores) > 1 and scores[1][0] == scores[0][0] and scores[0][0] > 0:
        queue = scores[1][1]
    priority = "high" if re.search(HIGH, low) else "low" if re.search(LOW, low) else "medium"
    ttype = next((t for t, rx in TYPE_RULES if re.search(rx, low)), "Problem")
    return {"queue": queue, "priority": priority, "type": ttype}


def _articles(user: str) -> list[tuple[str, str]]:
    return re.findall(r"\[(KB-[A-Z]+-\d+)\] ([^\n]+)", user)


def answer(user: str, triage_labels: dict | None = None) -> dict:
    ticket = _ticket(user)
    labels = triage_labels or triage(ticket)
    arts = _articles(user)
    needs_human = labels.get("priority") == "high" or bool(
        re.search(r"\b(refund|breach|data loss|legal|contract)\b", ticket.lower()))
    if arts:
        art_id, title = arts[0]
        body = (f"Thank you for contacting Fluxo support. Regarding your request, please see our help article "
                f"\"{title}\", which explains the steps to follow.")
        cited = [art_id]
    else:
        body = "Thank you for contacting Fluxo support. We have received your request and are looking into it."
        cited = []
    if needs_human:
        body += " A specialist from our team will follow up with you shortly."
    return {"answer": body + " Best regards, Fluxo Support", "cited_articles": cited, "needs_human": needs_human}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-zà-ÿ]{4,}", text.lower()))


def judge(user: str) -> dict:
    ref = user.split("Reference reply:")[1].split("Candidate reply:")[0]
    cand = user.split("Candidate reply:")[1]
    if CANARY in cand or not cand.strip():
        return {"verdict": "no", "reason": "empty reply or leaked instructions"}
    a, b = _words(ref), _words(cand)
    overlap = len(a & b) / max(1, len(a | b))
    verdict = "yes" if overlap >= 0.12 else "partially" if overlap >= 0.05 else "no"
    return {"verdict": verdict, "reason": f"mock judge, word overlap {overlap:.2f}"}


def paraphrase(user: str) -> dict:
    subject = user.split("\n", 1)[0].replace("Subject:", "").strip()
    body = user.split("\n", 1)[1].strip() if "\n" in user else ""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if s]
    swaps = {"Hello": "Hi", "please": "kindly", "issue": "problem", "help": "assist", "thank you": "thanks"}
    new = " ".join(reversed(sentences)) if len(sentences) > 1 else body
    for a, b in swaps.items():
        new = new.replace(a, b)
    return {"subject": f"Re: {subject}", "body": new}


def mock_reply(messages: list[dict], sample: int = 0) -> str:
    system, user = messages[0]["content"], messages[-1]["content"]
    task = system.split("\n", 1)[0].replace("TASK:", "").strip()
    if task == "triage_and_answer":
        labels = triage(_ticket(user), sample)
        return json.dumps({**labels, "answer": answer(user, labels)["answer"]})
    if task == "answer":
        labels = json.loads(re.search(r"Ticket triage: (\{.*?\})", user).group(1)) if "Ticket triage:" in user else None
        return json.dumps(answer(user, labels))
    if task == "judge":
        return json.dumps(judge(user))
    if task == "paraphrase":
        return json.dumps(paraphrase(user))
    return json.dumps({"error": f"mock: unknown task {task}"})
