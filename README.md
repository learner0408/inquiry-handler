# inquiry-handler

Inquiry Bot — answers user questions over messaging channels using RAG over Google Docs. WhatsApp is the first channel; Slack/Telegram reuse the same core.

## Quickstart

```bash
uv sync --frozen
cp .env.example .env   # fill secrets, allowlisted GOOGLE_DOC_IDS
docker compose up -d db
uv run python scripts/init_db.py && uv run alembic stamp head
uv run alembic upgrade head
uv run uvicorn src.main:app --reload
```

Expose for Meta verification:

```bash
ngrok http 8000
```

Then in Meta Dashboard → WhatsApp → Configuration: set webhook URL to `https://<ngrok>/webhooks/whatsapp`, verify token = `WHATSAPP_VERIFY_TOKEN`, subscribe to `messages`. `WHATSAPP_APP_SECRET` is the App Secret from Meta App Dashboard → Settings → Basic (used to verify `X-Hub-Signature-256`).

Test locally without Meta (compute the signature yourself):

```bash
python3 -c "
import json, hmac, hashlib
p = {'entry': [{'changes': [{'value': {'messages': [{'id': 'wamid.test1', 'from': '919999977777', 'type': 'text', 'text': {'body': 'What are the delivery charges?'}}]}}, {'field': 'messages'}]}]}
b = json.dumps(p, separators=(',', ':')).encode()
print('sha256=' + hmac.new(b'<your-app-secret>', b, hashlib.sha256).hexdigest())"
```

## Telegram (optional second channel)

1. Create a bot with @BotFather → copy the token.
2. Fill `TELEGRAM_BOT_TOKEN` and choose any random `TELEGRAM_WEBHOOK_SECRET` in `.env`.
3. Register the webhook once:

```bash
curl "https://api.telegram.org/bot<token>/setWebhook" \
  -d "url=https://<ngrok>/webhooks/telegram" \
  -d "secret_token=<TELEGRAM_WEBHOOK_SECRET>"
```

4. Message the bot — replies come from RAG; edits/callbacks/photo messages are ignored by design (`parse_telegram_payload` keeps only text-bearing `update.message`).

Ingest a doc (dry-run first):

```bash
uv run python -m src.rag.ingestion.ingest --doc-id <id> --dry-run
```

## Verify

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
```

## Evals

Prerequisites: Postgres running + docs ingested, `NVIDIA_API_KEY` and
`ANTHROPIC_API_KEY` in `.env` (retrieval, generation, and the Anthropic judge
are live — expect API costs; Meta creds are not needed).

```bash
uv run deepeval test run evals/test_rag.py
```

Golden inputs live in `evals/goldens/single_turn.json`. See `ARCHITECTURE.md`
for the metric suite.

## Docs

See `ARCHITECTURE.md` for decisions + user flow, `AGENTS.md` for agent guardrails.
