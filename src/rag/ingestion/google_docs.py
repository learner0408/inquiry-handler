"""Google Docs fetch: service-account read, allowlisted, text flattened.

Revision marker: the Docs API no longer returns revisionId, so revision_id is
the sha256 of the normalized full text (content-addressed). Identical text ->
identical revision -> re-ingest is a no-op; any edit changes the marker and
triggers replace + cache invalidation. Live fetch requires the service-account
email added as Viewer on each doc.
"""

import hashlib
from dataclasses import dataclass

from google.oauth2 import service_account
from googleapiclient.discovery import build

from src.config import get_settings


@dataclass
class FetchedDoc:
    doc_id: str
    revision_id: str
    title: str
    text: str


def _docs_service():  # type: ignore[no-untyped-def]
    settings = get_settings()
    creds = service_account.Credentials.from_service_account_file(
        settings.GOOGLE_SERVICE_ACCOUNT_JSON,
        scopes=["https://www.googleapis.com/auth/documents.readonly"],
    )
    return build("docs", "v1", credentials=creds)


def _flatten(structural_elements: list[dict]) -> str:
    """Extract plain text from Docs structural elements (paragraphs + tables)."""
    parts: list[str] = []
    for el in structural_elements:
        if "paragraph" in el:
            for pe in el["paragraph"].get("elements", []):
                tr = pe.get("textRun")
                if tr:
                    parts.append(tr.get("content", ""))
        elif "table" in el:
            for row in el["table"].get("tableRows", []):
                for cell in row.get("tableCells", []):
                    parts.append(_flatten(cell.get("content", [])))
    return "".join(parts)


def fetch_doc(doc_id: str) -> FetchedDoc:
    settings = get_settings()
    if doc_id not in settings.google_doc_id_list:
        raise ValueError(f"doc_id not in GOOGLE_DOC_IDS allowlist: {doc_id}")
    doc = _docs_service().documents().get(documentId=doc_id).execute()
    text = _flatten(doc.get("body", {}).get("content", [])).strip()
    revision_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return FetchedDoc(
        doc_id=doc_id,
        revision_id=revision_id,
        title=doc.get("title", ""),
        text=text,
    )
