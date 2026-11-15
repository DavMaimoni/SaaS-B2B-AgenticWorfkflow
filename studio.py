"""LangGraph Studio entry point (`uv run langgraph dev`). Studio provides the checkpointer.

Try: {"subject": "Charged twice", "body": "Our card was charged twice this month, please refund.",
      "customer_email": "admin@quanta-labs.example.com"}
With HUMAN_IN_THE_LOOP=1 in .env, escalated tickets pause on `escalate`: resume with {"approve": true}.
"""

from support_agent.graph import build_graph

graph = build_graph()
