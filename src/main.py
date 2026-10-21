"""FastAPI bootstrap only. No business logic here."""

from fastapi import FastAPI

from src.webhooks.whatsapp import router as whatsapp_router

app = FastAPI(title="inquiry-handler")
app.include_router(whatsapp_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
