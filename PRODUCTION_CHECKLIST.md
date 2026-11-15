# Production readiness checklist

The agent goes to production only when every box is ticked with evidence (a link to a run, a test or a document).

## Quality

- [ ] 200-triple evaluation run on the release candidate; `eval/summary.py check` passes against `eval/baseline.json`
- [ ] 100 % of critical tickets escalated, 100 % of out-of-scope tickets redirected, 0 leaks on injections
- [ ] LLM judge calibrated on 40 human labels (Cohen's kappa reported, ≥ 0.6) — `eval/calibrate.py judge`
- [ ] LoRA vs base model compared with a paired bootstrap; model card filled in — `MODEL_CARD.md`
- [ ] Unnecessary escalations on routine tickets measured and accepted by the support lead

## Reliability

- [ ] CI green: unit tests + mock pipeline on every PR, replay smoke on every PR, nightly full run
- [ ] Fallback tested: small model unreachable → every ticket goes to the large model (`test_nodes.py`)
- [ ] Timeouts and retries on every model call (`llm.py`: 120 s, 3 retries) and every tool call (`tools.py`)
- [ ] Rate limits of the API providers known; alerting if the large model starts returning 429
- [ ] Writes idempotent (`idempotency_key` from thread + tool + arguments), tested (`test_mcp_server.py`)
- [ ] Human approval on escalations enabled (`HUMAN_IN_THE_LOOP=1`) with a persistent checkpointer

## Safety and data

- [ ] Refunds, data deletion, security incidents never answered by the small model alone (`routing.py`)
- [ ] No personal or account data in the semantic cache (`nodes.is_cacheable`) or in the logs (`runs/metrics.jsonl`
      stores labels and metrics, not ticket text)
- [ ] Cache threshold calibrated with 0 false hits on the trap pairs — `eval/calibrate.py cache`
- [ ] Cache invalidated when a help-center article changes; TTL set
- [ ] Prompt-injection triples pass; canary never appears in a reply
- [ ] Dataset licence (CC BY-NC 4.0) respected; data card up to date — `DATA_CARD.md`

## Operations

- [ ] Monitoring dashboard and `monitoring/report.py` alerts wired to a channel; an on-call owner named
- [ ] Alert thresholds set from the reference run (latency p95, cost per ticket, small-model share, escalation rate,
      sampled judge score)
- [ ] Judge sampled on 5 % of real traffic, reviewed weekly
- [ ] Rollback in one command: previous LoRA tag in Ollama, previous graph version in git
- [ ] Cost budget per day agreed, alert above it
- [ ] Runbook: what to do when the small model is down, when the large provider is down, when the CRM is down
