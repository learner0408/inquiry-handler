"""Channel adapter contract. New channel = implement BaseChannel."""

from typing import Protocol


class InboundMessage(Protocol):
    channel: str
    sender_id: str
    text: str
    external_id: str  # e.g. Meta message.id for dedupe


class BaseChannel(Protocol):
    async def parse(self, payload: dict) -> InboundMessage | None: ...
    async def send(self, to: str, text: str) -> None: ...
