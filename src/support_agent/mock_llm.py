import json
from typing import Any
from langchain_core.messages import AIMessage

class MockResponse(AIMessage):
    def __init__(self, content: str):
        super().__init__(content=content)
        self.text = content
        self.latency_ms = 15.0
        self.cost_usd = 0.0
        self.tokens_in = 50
        self.tokens_out = 20

class MockLLM:
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get("name", "mock")

    def with_structured_output(self, schema: Any, *args, **kwargs) -> Any:
        """Intercepts LangChain's structured output calls used by the evaluation judge."""
        class StructuredMock:
            def invoke(self, messages, *a, **kw):
                data = {
                    "score": 1.0, 
                    "reasoning": "Perfect alignment with reference.", 
                    "holding_status": "skipped"
                }
                
                # Pydantic v2 support
                if hasattr(schema, "model_construct"):
                    valid_keys = schema.model_fields.keys()
                    filtered_data = {k: v for k, v in data.items() if k in valid_keys}
                    return schema.model_construct(**filtered_data)
                
                # Pydantic v1 fallback
                elif hasattr(schema, "__fields__"):
                    valid_keys = schema.__fields__.keys()
                    filtered_data = {k: v for k, v in data.items() if k in valid_keys}
                    return schema.construct(**filtered_data)
                
                return data

            def chat(self, messages, *a, **kw):
                return self.invoke(messages, *a, **kw)

        return StructuredMock()

    def invoke(self, messages, *args, **kwargs) -> MockResponse:
        return self._generate(messages)

    def chat(self, messages, *args, **kwargs) -> MockResponse:
        return self._generate(messages)

    def _generate(self, messages) -> MockResponse:
        msg_str = str(messages).lower()
        if "score" in msg_str or "judge" in msg_str:
            return MockResponse(json.dumps({
                "score": 1.0, 
                "reasoning": "Perfect.", 
                "holding_status": "skipped"
            }))
        return MockResponse("Support response.")
