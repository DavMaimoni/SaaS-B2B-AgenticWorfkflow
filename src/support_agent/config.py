"""Every setting in one place. Values come from the environment (.env), with defaults that run offline."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "kb"
CRM_PATH = Path(os.getenv("CRM_PATH", DATA_DIR / "crm.sqlite"))
RUNS_DIR = Path(os.getenv("RUNS_DIR", ROOT / "runs"))
RECORDINGS_DIR = ROOT / "eval" / "recordings"

# ---- labels (from the Tobi-Bueck dataset card, restricted to a SaaS support desk) --------------------------------
QUEUES = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "Billing and Payments",
    "IT Support",
    "Service Outages and Maintenance",
    "General Inquiry",
]
OUT_OF_SCOPE_QUEUES = ["Returns and Exchanges", "Sales and Pre-Sales", "Human Resources"]
PRIORITIES = ["low", "medium", "high"]
TYPES = ["Incident", "Request", "Problem", "Change"]

# Generic questions whose answer does not depend on the customer's account: safe to cache.
CACHEABLE_QUEUES = {"Customer Service", "Product Support", "General Inquiry", "Technical Support"}
CACHEABLE_TYPES = {"Request", "Problem"}


@dataclass(frozen=True)
class ModelSpec:
    name: str          # logical name used in metrics: "small" | "large" | "judge"
    provider: str      # "openrouter" | "ollama" | "openai_compatible" | "mock"
    model: str
    base_url: str | None
    api_key_env: str | None
    price_in: float    # USD per million input tokens
    price_out: float   # USD per million output tokens


def _spec(name: str, prefix: str, provider: str, model: str, price_in: float, price_out: float) -> ModelSpec:
    provider = os.getenv(f"{prefix}_PROVIDER") or provider
    defaults = {
        "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        "ollama": ("http://localhost:11434/v1", None),
        "openai_compatible": (None, "OPENAI_API_KEY"),
        "mock": (None, None),
    }
    base_url, key_env = defaults.get(provider, (None, None))
    return ModelSpec(
        name=name,
        provider=provider,
        model=os.getenv(f"{prefix}_MODEL") or model,
        base_url=os.getenv(f"{prefix}_BASE_URL") or base_url,
        api_key_env=os.getenv(f"{prefix}_API_KEY_ENV") or key_env,
        price_in=float(os.getenv(f"{prefix}_PRICE_IN", price_in)),
        price_out=float(os.getenv(f"{prefix}_PRICE_OUT", price_out)),
    )


@dataclass(frozen=True)
class Settings:
    # Small model = the LoRA served by Ollama. Local inference is priced at 0 by default; set SMALL_PRICE_* to
    # charge it at a hosted rate if you want a fairer comparison.
    small: ModelSpec = field(default_factory=lambda: _spec(
        "small", "SMALL", "mock", "fluxo-support-lora", 0.0, 0.0))
    # Large model = the fallback for risky tickets. DeepSeek through OpenRouter, as in the course labs.
    large: ModelSpec = field(default_factory=lambda: _spec(
        "large", "LARGE", "mock", "deepseek/deepseek-v4.1-flash", 0.033, 0.080))
    # Judge = a different model family than the generators.
    judge: ModelSpec = field(default_factory=lambda: _spec(
        "judge", "JUDGE", "mock", "google/gemini-2.5-flash", 0.30, 2.50))

    # live | record | replay  (record = call the API and save; replay = answer from eval/recordings only)
    llm_mode: str = field(default_factory=lambda: os.getenv("LLM_MODE", "live"))

    # Routing
    triage_samples: int = field(default_factory=lambda: int(os.getenv("TRIAGE_SAMPLES", 2)))           # self-consistency samples for confidence
    route_confidence_threshold: float = field(default_factory=lambda: float(os.getenv("ROUTE_CONFIDENCE", 0.75)))
    max_retries_on_large: int = 1

    # Semantic cache
    cache_enabled: bool = field(default_factory=lambda: os.getenv("CACHE_ENABLED", "1") == "1")
    cache_threshold: float = field(default_factory=lambda: float(os.getenv("CACHE_THRESHOLD", 0.92)))
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", 7 * 24 * 3600)))
    embedder: str = field(default_factory=lambda: os.getenv("EMBEDDER", "hashing"))                     # hashing | e5
    e5_model: str = field(default_factory=lambda: os.getenv("E5_MODEL", "intfloat/multilingual-e5-small"))

    # Retrieval
    kb_top_k: int = field(default_factory=lambda: int(os.getenv("KB_TOP_K", 3)))

    # Human in the loop: pause on escalations (interrupt) instead of only creating the ticket.
    human_in_the_loop: bool = field(default_factory=lambda: os.getenv("HUMAN_IN_THE_LOOP", "0") == "1")


def settings() -> Settings:
    return Settings()
