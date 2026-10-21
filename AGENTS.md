# AGENTS.md — inquiry-handler

Inquiry Bot: channel adapters → FastAPI core → RAG over Google Docs → LLM answer. WhatsApp is the first adapter; Slack/Telegram plug in later via same interface.

See `README.md` for quickstart, `ARCHITECTURE.md` for decisions + user flow. Do not duplicate them here.

## Stack
Python 3.12 + FastAPI + Uvicorn. `uv`-managed env only (`uv sync --frozen`). Deps canonical in `pyproject.toml + uv.lock` — never add `requirements.txt`. Postgres + pgvector (vectors + chat history). Evals via DeepEval.

## Framework
Default to LangChain for RAG/agent code (`langchain`, `langchain-openai`, `langchain-postgres`) and LangGraph when the workflow needs branching/loops/multi-step. Use raw SDK/SQL only when LangChain can't express it — note why in the change.

## Layout
- RAG stays under `src/rag/` — stages as children, never new top-level `src/*` for RAG code.
- `src/main.py`, `src/config.py` — bootstrap + settings only
- `src/channels/{base,whatsapp}.py` — adapters only, no LLM. New channel = new file impl `BaseChannel`.
- `src/webhooks/` — thin handlers → `channels` → `rag.generation.pipeline.handle_inquiry()`
- `src/rag/ingestion/` — `google_docs.py` (service-account, allowlisted `GOOGLE_DOC_IDS`, store `doc_id+revision+chunk_hash`)
- `src/rag/chunking/`, `src/rag/embeddings/` — dims must match pgvector index
- `src/rag/vectordb/chunks.py` — chunk embeddings behind `VectorDBInterface` (atomic per-doc replace)
- `src/rag/cache/semantic_cache.py` — checked BEFORE vectordb; store `{query_emb, answer, citations, revision}`; never cache fallbacks; invalidate on revision bump
- `src/rag/retrieval/` — `hybrid.py` (dense + BM25 RRF)
- `src/rag/reranking/reranker.py`
- `src/rag/generation/` — `pipeline.py` (sole orchestrator), `answer.py` (sole LLM caller)
- `src/conversation/` — `threads.py/history.py` (`conversation_threads` + `thread_messages`, `sender_id→thread_id`, load last N + summary, cap ~20 then summarize)
- `prompts/rag_prompts.yaml` — versioned only, never hardcoded
- `evals/` — DeepEval `test_*.py` (`Golden`/`ConversationalGolden` + `assert_test`), goldens in `evals/goldens/`
- Imports: all at file top (after docstring), stdlib → third-party → first-party, `ruff check` clean. No function-level imports except circular-guard/lazy-heavy-dep with an inline `# why:` note.

## Flow — order matters
`POST /webhooks/whatsapp` → verify `X-Hub-Signature-256` (HMAC-SHA256 raw body, key = `WHATSAPP_APP_SECRET`) → 200 fast (statuses ACK'd + ignored) → check-first dedupe on `external_id` (unique backstop in `history.save`) → background task → `history.load` → `cache.get` → [HIT return] / [MISS → retrieval → rerank → `answer.py` → `cache.set` + `history.save`] → send API.

## Commands
- `uv sync --frozen`, `cp .env.example .env`, `docker compose up -d db`, `uv run python scripts/init_db.py && uv run alembic stamp head` (fresh DB; later schema changes via real Alembic revisions in `migrations/versions/`)
- `uv run uvicorn src.main:app --reload` (expose via `ngrok http 8000` for Meta verify)
- `uv run pytest -q` / `uv run pytest tests/test_x.py -q` (unit only)
- `uv run ruff check .`, `uv run mypy src`
- `uv run alembic upgrade head` (apply pending migrations; also after fresh `init_db` + stamp)
- `uv run python -m src.rag.ingestion.ingest --doc-id <id> --dry-run`
- Evals only via `deepeval test run evals/test_rag.py` (never plain `pytest`)

## `src/config.py`
Single `pydantic-settings BaseSettings` + `lru_cache get_settings()`. Holds `DATABASE_URL, WHATSAPP_VERIFY_TOKEN/ACCESS_TOKEN/PHONE_ID/APP_SECRET, GOOGLE_SERVICE_ACCOUNT_JSON (path), GOOGLE_DOC_IDS, EMBEDDING_MODEL (nvidia/nemotron-3-embed-1b)/NEMOTRON_EMBED_URL/NVIDIA_API_KEY/VECTOR_DIMS (2048), CHUNK_STRATEGY/SIZE/OVERLAP, TOP_K, RERANK_MODEL/BASE_URL/TOP_N, LLM_PROVIDER (anthropic)/LLM_MODEL (claude-sonnet-4-6)/LLM_TEMPERATURE/ANTHROPIC_API_KEY, CACHE_SIM_THRESHOLD (0.8), HISTORY_MAX_TURNS, OPENAI_API_KEY (only when provider=openai)`. LLM secret enforced lazily in `answer.py`; `WHATSAPP_APP_SECRET` lazily in the webhook path (retrieval/ingest paths run key-free). No defaults for secrets; no `os.getenv` elsewhere.

## Gotchas
- Never read `.env` or `./secrets/*` contents (keys, tokens, JSON) into context — reference variable/file names only. Verify presence/shape via key names (e.g. `cut -d= -f1 .env`), never values.
- pydantic-settings precedence: `.env` overrides `config.py` defaults **field-by-field**. Adding a key later changes only that field — keep `LLM_PROVIDER` and `LLM_MODEL` consistent (provider default is `anthropic`; a stale openai-model `LLM_MODEL` in `.env` with no `LLM_PROVIDER` line collides).
- Native arm64 venv required: the Anaconda `uv` on this machine is x86_64 and installs broken native wheels. Use native arm64 `uv` (`/opt/homebrew/bin/uv`); sanity check `uv run python -c "import platform; print(platform.machine())"` prints `arm64`.
- Meta: GET verify returns `hub.challenge` as text/plain with **dot-keyed** query params (`hub.mode` etc.); POST ack <20s then async; `statuses` payloads ACK'd + ignored; fresh portfolios may lock test WABAs (error 131031) pending business verification — Meta-side gate, test locally via signed curl; 24h window; media = `media_id` (5-min URL).
- RAG-only, low temp, explicit “not in docs” fallback with citations.
- Public HTTPS required; subscribe to `messages` field in dashboard.

## Docs
Keep `README.md` (quickstart) and `ARCHITECTURE.md` (decisions + flow) up to date with every behavior/structure/env change — stale docs are a bug.
