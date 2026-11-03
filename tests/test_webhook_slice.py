"""WhatsApp webhook tests: signature gate, ack-fast, statuses ignored, dedupe.

Real HMAC signatures (test secret); pipeline/send/history/DB stubbed or
patched — no Meta network. Fixture payloads mirror real Meta shapes
(131031-locked test WABA status included, from live Caller observation).
"""

import hashlib
import hmac
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from src.channels.whatsapp import parse_meta_payload
from src.config import get_settings

TEST_SECRET = "test-app-secret"

STATUS_131031 = {
    "object": "whatsapp_business_account",
    "entry": [{
        "id": "1432771782145367",
        "changes": [{
            "value": {
                "messaging_product": "whatsapp",
                "metadata": {"display_phone_number": "15551953655",
                             "phone_number_id": "1317632148110121"},
                "contacts": [{"wa_id": "817083001442", "user_id": "JP.1442400277828653"}],
                "statuses": [{
                    "id": "wamid." + uuid.uuid4().hex,
                    "status": "failed",
                    "timestamp": "1792412829",
                    "recipient_id": "817083001442",
                    "errors": [{"code": 131031,
                                "title": "Business Account locked",
                                "message": "Business Account locked",
                                "error_data": {"details": "Business account has been locked."},
                                "href": "https://developers.facebook.com/docs/whatsapp/cloud-api/support/error-codes/"}],
                }],
            },
            "field": "messages",
        }],
    }],
}

MESSAGE_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [{
        "id": "1432771782145367",
        "changes": [{
            "value": {
                "messaging_product": "whatsapp",
                "metadata": {"display_phone_number": "15551953655",
                             "phone_number_id": "1317632148110121"},
                "contacts": [{"profile": {"name": "Tester"}, "wa_id": "919999977777"}],
                "messages": [{
                    "from": "919999977777",
                    "id": "wamid." + uuid.uuid4().hex,
                    "timestamp": "1792412830",
                    "text": {"body": "What are the delivery charges?"},
                    "type": "text",
                }],
            },
            "field": "messages",
        }],
    }],
}


@pytest.fixture
def client(monkeypatch) -> TestClient:  # type: ignore[no-untyped-def]
    get_settings.cache_clear()
    settings = get_settings()
    settings.WHATSAPP_APP_SECRET = TEST_SECRET  # type: ignore[misc]
    settings.WHATSAPP_VERIFY_TOKEN = "vt-" + uuid.uuid4().hex[:6]  # type: ignore[misc]
    monkeypatch.setattr("src.webhooks.whatsapp.maintain", lambda channel, sender_id: None)

    from src.main import app

    with TestClient(app) as tc:
        yield tc


def _signed(payload: dict) -> dict:
    body = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(TEST_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return {"Content-Type": "application/json",
            "X-Hub-Signature-256": f"sha256={sig}"}


def _raw_signed(raw: bytes) -> dict:
    sig = hmac.new(TEST_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    return {"Content-Type": "application/json",
            "X-Hub-Signature-256": f"sha256={sig}"}


def test_get_verify_returns_challenge_plain(client) -> None:  # type: ignore[no-untyped-def]
    settings = get_settings()
    # Meta sends dot-keyed params — shapes must match real Meta exactly
    r = client.get(
        "/webhooks/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": settings.WHATSAPP_VERIFY_TOKEN,
                "hub.challenge": "CHALLENGE_123"},
    )
    assert r.status_code == 200 and r.text == "CHALLENGE_123"
    assert "text/plain" in r.headers["content-type"]


def test_get_verify_rejects_bad_token(client) -> None:  # type: ignore[no-untyped-def]
    r = client.get("/webhooks/whatsapp",
                   params={"hub.mode": "subscribe", "hub.verify_token": "wrong",
                           "hub.challenge": "x"})
    assert r.status_code == 403


def test_parse_meta_payload_extracts_message_and_ignores_statuses() -> None:
    msg = parse_meta_payload(MESSAGE_PAYLOAD)
    assert msg is not None and msg.sender_id == "919999977777"
    assert msg.text == "What are the delivery charges?"
    assert msg.external_id.startswith("wamid.")
    assert parse_meta_payload(STATUS_131031) is None  # statuses -> None


def test_post_rejects_bad_signature_before_parsing(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fired = []
    monkeypatch.setattr("src.webhooks.whatsapp.parse_meta_payload",
                        lambda p: fired.append(p) or None)
    r = client.post("/webhooks/whatsapp", content=b'{"tampered": true}',
                    headers={"X-Hub-Signature-256": "sha256=deadbeef"})
    assert r.status_code == 403 and fired == []  # unverified input never parsed


def test_post_status_payload_200_no_processing(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen = {}
    monkeypatch.setattr("src.webhooks.whatsapp._external_id_seen",
                        lambda mid: seen.setdefault("called", True))
    monkeypatch.setattr("src.webhooks.whatsapp.handle_inquiry",
                        lambda *a, **k: seen.setdefault("pipeline", True))
    sig_headers = _signed(STATUS_131031)
    r = client.post("/webhooks/whatsapp", json=STATUS_131031, headers=sig_headers)
    assert r.status_code == 200 and seen == {}  # acknowledged, nothing dispatched


def test_post_message_dedupe_skip(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("src.webhooks.whatsapp._external_id_seen", lambda mid: True)
    monkeypatch.setattr("src.webhooks.whatsapp.handle_inquiry",
                        lambda *a, **k: pytest.fail("dupes must not re-process"))
    r = client.post("/webhooks/whatsapp", json=MESSAGE_PAYLOAD, headers=_signed(MESSAGE_PAYLOAD))
    assert r.status_code == 200


def test_post_message_fresh_routes_to_pipeline_and_sends(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = {}

    class FakeResult:
        text = "Shipping varies by weight [docA1234]."

    monkeypatch.setattr("src.webhooks.whatsapp._external_id_seen", lambda mid: False)
    monkeypatch.setattr("src.webhooks.whatsapp.handle_inquiry",
                        lambda ch, sid, text, external_id=None: calls.setdefault(
                            "pipeline",
                            {"ch": ch, "sid": sid, "text": text,
                             "eid": external_id}) and FakeResult())
    async def fake_send(to: str, text: str) -> dict:
        calls.setdefault("send", {"to": to, "text": text})
        return {"sent": True}

    monkeypatch.setattr("src.webhooks.whatsapp.send_message", fake_send)
    r = client.post("/webhooks/whatsapp", json=MESSAGE_PAYLOAD, headers=_signed(MESSAGE_PAYLOAD))
    assert r.status_code == 200
    import time

    for _ in range(50):
        if "send" in calls:
            break
        time.sleep(0.1)
    assert calls.get("pipeline", {}).get("eid", "").startswith("wamid.")
    assert calls["pipeline"]["sid"] == "919999977777"
    assert calls["send"]["to"] == "919999977777"
    assert calls["send"]["text"].startswith("Shipping")


def test_message_body_verified_against_real_signature_order() -> None:
    # json.dumps key order must match what Meta signed: our _signed() builds the
    # same separators as the webhook reads raw bytes -> this round-trips.
    body = json.dumps(MESSAGE_PAYLOAD, separators=(",", ":")).encode()
    assert hmac.compare_digest(
        _raw_signed(body)["X-Hub-Signature-256"],
        _signed(MESSAGE_PAYLOAD)["X-Hub-Signature-256"],
    )
