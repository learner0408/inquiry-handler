"""Single settings source of truth. No os.getenv elsewhere."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "local"
    DATABASE_URL: str

    WHATSAPP_VERIFY_TOKEN: str
    WHATSAPP_ACCESS_TOKEN: str
    WHATSAPP_PHONE_ID: str
    # HMAC key for X-Hub-Signature-256 (inbound Meta trust). Enforced lazily in
    # the webhook path so non-webhook paths never need it.
    WHATSAPP_APP_SECRET: str = ""

    GOOGLE_SERVICE_ACCOUNT_JSON: str
    GOOGLE_DOC_IDS: str

    EMBEDDING_MODEL: str = "nvidia/nemotron-3-embed-1b"
    NVIDIA_API_KEY: str
    NEMOTRON_EMBED_URL: str = "https://integrate.api.nvidia.com/v1"
    VECTOR_DIMS: int = 2048
    TOP_K: int = 8
    RERANK_TOP_N: int = 4
    RERANK_MODEL: str = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    RERANK_BASE_URL: str = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2"

    CHUNK_STRATEGY: str = "recursive"
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100

    LLM_PROVIDER: str = "anthropic"  # anthropic | openai; enforced in answer.py (lazy)
    LLM_MODEL: str = "claude-sonnet-4-6"
    LLM_TEMPERATURE: float = 0.2
    # LLM secrets are validated lazily in the LLM caller so retrieval/ingest
    # paths run key-free; no os.getenv elsewhere.
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    CACHE_SIM_THRESHOLD: float = 0.8
    HISTORY_MAX_MESSAGES: int = 20

    @property
    def google_doc_id_list(self) -> list[str]:
        return [d.strip() for d in self.GOOGLE_DOC_IDS.split(",") if d.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
