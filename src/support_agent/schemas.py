"""The structured outputs the models must return. Pydantic validates them; a validation error is a failed call."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from support_agent.config import OUT_OF_SCOPE_QUEUES, PRIORITIES, QUEUES, TYPES

Queue = Literal[tuple(QUEUES + OUT_OF_SCOPE_QUEUES)]  # type: ignore[valid-type]
Priority = Literal[tuple(PRIORITIES)]  # type: ignore[valid-type]
TicketType = Literal[tuple(TYPES)]  # type: ignore[valid-type]


class Ticket(BaseModel):
    subject: str = ""
    body: str
    customer_email: str | None = None

    def text(self) -> str:
        return f"{self.subject.strip()}\n\n{self.body.strip()}".strip()


class Triage(BaseModel):
    """Output of the triage step (the LoRA's first job)."""
    queue: Queue
    priority: Priority
    type: TicketType
    in_scope: bool = Field(True, description="False for requests a SaaS support desk does not handle")

    @field_validator("queue", "priority", "type", mode="before")
    @classmethod
    def _canonical(cls, v, info):
        """Accept any casing and stray spaces from the model ("incident", "billing and payments ")."""
        if not isinstance(v, str):
            return v
        labels = {"queue": QUEUES + OUT_OF_SCOPE_QUEUES, "priority": PRIORITIES, "type": TYPES}[info.field_name]
        return {label.lower(): label for label in labels}.get(" ".join(v.split()).lower(), v)


class Draft(BaseModel):
    """Output of the answer step."""
    answer: str = Field(min_length=1)
    cited_articles: list[str] = Field(default_factory=list, description="KB article ids used, e.g. ['KB-BILL-01']")
    needs_human: bool = Field(False, description="True when the model cannot resolve the ticket safely")


class SftTarget(BaseModel):
    """What the LoRA learns to output in one go: the triage labels and the reply."""
    queue: str
    priority: str
    type: str
    answer: str
