# Data card

## Tickets: Tobi-Bueck/customer-support-tickets

| | |
| --- | --- |
| Source | [huggingface.co/datasets/Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) |
| Licence | CC BY-NC 4.0 (attribution, non-commercial): fine for this course, not for a commercial product |
| Size | 61,765 rows, English and German (dataset card) |
| Fields used | `subject`, `body`, `answer`, `queue`, `priority`, `type`, `language`, `tag_*` |
| Origin | presented with a synthetic ticket generator and "no PII": treat as synthetic |

### Cleaning (`scripts/prepare_data.py`, report in `data/processed/stats.json`)

1. English only (`--language en`).
2. Queues: keep the 10 documented in the card (7 handled by the SaaS desk, 3 out of scope: Returns and Exchanges,
   Sales and Pre-Sales, Human Resources). The viewer shows 52 queue values: the others are dropped and listed in
   `unknown_queues`.
3. Priorities: rows use `low` / `medium` / `high`; `critical` and `urgent` are mapped to `high`.
4. Types: Incident, Request, Problem, Change.
5. Bodies or answers under 20 characters dropped.
6. Duplicates dropped: same normalised subject + body, or same first 200 normalised body characters.
7. Stratified split by (queue, priority): 80 % train, 10 % validation, 10 % test, seed 42.

To fill after running step 1: rows kept, distribution by queue and priority, rows dropped at each step.

| Split | Rows | Used for |
| --- | ---: | --- |
| train | | LoRA training (`finetune/build_sft.py`) |
| val | | LoRA early stopping, router threshold |
| test | | the 200 evaluation triples only |

### Known limitations

- Synthetic and multi-domain (healthcare, finance, consumer hardware...), not specific to a SaaS product.
- Reference answers are generic and not grounded in any help center, so they are a weak gold standard: the judge
  grades usefulness and absence of invention rather than closeness to the reference.
- Answers contain placeholders such as `<tel_num>`; the sentences that hold them are removed from training targets.
- Label noise: 52 queue values in the viewer against 10 documented.

## Fluxo help center (`data/kb/`)

17 short articles written for this project about a fictional product (plans, billing, refunds, API, integrations,
SSO, security, outages). They define the facts the agent may state. Any change to an article must invalidate the
cache entries that cite it (`SemanticCache.invalidate_article`).

## Simulated CRM (`src/support_agent/mcp_server/crm.py`)

20 fictional companies with `example.com` emails, plans, seats, billing cycle and three invoices each.
One account is read-only after a failed payment, to test billing scenarios.

## Demo tickets (`data/sample_tickets.jsonl`)

48 tickets written by hand in the dataset's schema, for offline tests and demos only. Never report results on them.
