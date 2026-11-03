"""Telegram webhook tests: secret-token gate, ack-fast, non-text ignored, dedupe.

Real secret-token headers (test secret); pipeline/send/maintain/DB stubbed or
patched — no Telegram network. Payload shapes mirror real Bot API updates.
"""

import pytest
from fastapi.testclient import TestClient

from src.channels.telegram import parse_telegram_payload
from src.config import get_settings

TEST_SECRET = "test-webhook-secret"

MESSAGE_UPDATE = {
    "update_id": 9007199254740001,
    "message": {
        "message_id": 42,
        "from": {"id": 42, "is_bot": False,
                 "first_name": "Tester", "username": "tester",
                 "language_code": "en"},
        "chat": {"id": 42, "first_name": "Tester", "username": "tester",
                 "type": "private"},
        "date": 1768000000,
        "text": "What are the delivery charges?",
    },
}

NON_TEXT_UPDATE = {
    "update_id": 9007199254740002,
    "message": {
        "message_id": 43,
        "from": {"id": 43, "is_bot": False, "first_name": "Tester"},
        "chat": {"id": 43, "first_name": "Tester", "type": "private"},
        "date": 1768000001,
        "photo": [{"file_id": "f" * 30, "file_unique_id": "u" * 10,
                   "width": 100, "height": 100}],
    },
}

EDITED_UPDATE = {
    "update_id": 9007199254740003,
    "edited_message": {
        "message_id": 44,
        "from": {"id": 44, "is_bot": False, "first_name": "Tester"},
        "chat": {"id": 44, "first_name": "Tester", "type": "private"},
        "date": 1768000002,
        "edit_date": 1768000005,
        "text": "edited text",
    },
}


@pytest.fixture
def client(monkeypatch) -> TestClient:  # type: ignore[no-untyped-def]
    get_settings.cache_clear()
    settings = get_settings()
    settings.TELEGRAM_WEBHOOK_SECRET = TEST_SECRET  # type: ignore[misc]
    settings.TELEGRAM_BOT_TOKEN = "test-bot-token"  # type: ignore[misc]
    monkeypatch.setattr("src.webhooks.telegram.maintain", lambda channel, sender_id: None)

    from src.main import app

    with TestClient(app) as tc:
        yield tc


def _headers() -> dict:
    return {"Content-Type": "application/json",
            "X-Telegram-Bot-Api-Secret-Token": TEST_SECRET}


def test_parse_extracts_message_and_ignores_non_text() -> None:
    msg = parse_telegram_payload(MESSAGE_UPDATE)
    assert msg is not None and msg.sender_id == "42"
    assert msg.text == "What are the delivery charges?"
    assert msg.external_id == "9007199254740001"
    assert parse_telegram_payload(NON_TEXT_UPDATE) is None  # photo -> None
    assert parse_telegram_payload(EDITED_UPDATE) is None  # edits -> None


def test_post_rejects_bad_secret_before_parsing(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fired = []
    monkeypatch.setattr("src.webhooks.telegram.parse_telegram_payload",
                        lambda p: fired.append(p) or None)
    r = client.post("/webhooks/telegram", content=b'{"tampered": true}',
                    headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"})
    assert r.status_code == 403 and fired == []  # unverified input never parsed


def test_post_non_message_update_200_no_processing(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen = {}
    monkeypatch.setattr("src.webhooks.telegram._external_id_seen",
                        lambda mid: seen.setdefault("called", True))
    monkeypatch.setattr("src.webhooks.telegram.handle_inquiry",
                        lambda *a, **k: seen.setdefault("pipeline", True))
    r = client.post("/webhooks/telegram", json=EDITED_UPDATE, headers=_headers())
    assert r.status_code == 200 and seen == {}  # acknowledged, nothing dispatched


def test_post_message_dedupe_skip(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("src.webhooks.telegram._external_id_seen", lambda mid: True)
    monkeypatch.setattr("src.webhooks.telegram.handle_inquiry",
                        lambda *a, **k: pytest.fail("dupes must not re-process"))
    r = client.post("/webhooks/telegram", json=MESSAGE_UPDATE, headers=_headers())
    assert r.status_code == 200


def test_post_message_fresh_routes_to_pipeline_and_sends(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = {}

    class FakeResult:
        text = "Shipping varies by weight [docA1234]."

    monkeypatch.setattr("src.webhooks.telegram._external_id_seen", lambda mid: False)
    monkeypatch.setattr("src.webhooks.telegram.handle_inquiry",
                        lambda ch, sid, text, external_id=None: calls.setdefault(
                            "pipeline",
                            {"ch": ch, "sid": sid, "text": text,
                             "eid": external_id}) and FakeResult())

    async def fake_send(to: str, text: str) -> dict:
        calls.setdefault("send", {"to": to, "text": text})
        return {"sent": True}

    monkeypatch.setattr("src.webhooks.telegram.send_message", fake_send)
    r = client.post("/webhooks/telegram", json=MESSAGE_UPDATE, headers=_headers())
    assert r.status_code == 200
    import time

    for _ in range(50):
        if "send" in calls:
            break
        time.sleep(0.1)
    assert calls.get("pipeline", {}).get("eid") == str(MESSAGE_UPDATE["update_id"])
    assert calls["pipeline"]["ch"] == "telegram"
    assert calls["pipeline"]["sid"] == "42"
    assert calls["send"]["to"] == "42"
    assert calls["send"]["text"].startswith("Shipping")
