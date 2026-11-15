.PHONY: test demo data triples eval ci-baseline report dashboard mcp studio

test:            ## unit tests (offline)
	uv run pytest -q

demo:            ## the whole pipeline on mock models and the demo tickets
	uv run python scripts/prepare_data.py --sample
	uv run python eval/build_triples.py --sample
	uv run python eval/run_eval.py --out eval/runs/demo

data:            ## download, clean and split the Tobi-Bueck dataset
	uv run python scripts/prepare_data.py

triples:         ## 200 evaluation triples + 40 smoke triples
	uv run python eval/build_triples.py

eval:            ## full evaluation with the models configured in .env
	uv run python eval/run_eval.py --out eval/runs/latest

ci-baseline:     ## refresh the mock baseline after an intended behaviour change
	SMALL_PROVIDER=mock LARGE_PROVIDER=mock JUDGE_PROVIDER=mock uv run python eval/run_eval.py --triples eval/triples.sample.jsonl --out eval/runs/ci-mock
	uv run python -c "import json; s=json.load(open('eval/runs/ci-mock/summary.json')); b=json.load(open('eval/baseline_ci.json')); json.dump({k: s.get(k, v) for k, v in b.items()}, open('eval/baseline_ci.json','w'), indent=2)"

report:          ## monitoring KPIs and alerts
	uv run python monitoring/report.py

dashboard:       ## monitoring dashboard
	uv run streamlit run monitoring/dashboard.py

mcp:             ## run the MCP server on stdio (for MCP Inspector, Claude Desktop...)
	uv run python -m support_agent.mcp_server.server

studio:          ## LangGraph Studio
	uv run langgraph dev
