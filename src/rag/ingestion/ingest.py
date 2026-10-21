"""Ingest CLI: --doc-id (repeatable, allowlist-enforced) + --dry-run/--commit.

Order: fetch -> chunk -> embed (passage) -> atomic_replace. Dry-run prints
counts and writes nothing.
"""

import argparse
import sys

from src.rag.chunking.chunker import get_chunker
from src.rag.embeddings.embedder import embed_passages
from src.rag.ingestion.google_docs import fetch_doc
from src.rag.vectordb.chunks import ChunkRow, PgVectorChunks


def ingest_doc(doc_id: str, dry_run: bool) -> dict:
    fetched = fetch_doc(doc_id)
    chunks = get_chunker().split(fetched.text)
    embeddings = embed_passages([c.text for c in chunks]) if chunks else []
    rows = [
        ChunkRow(chunk_index=c.chunk_index, text=c.text, chunk_hash=c.chunk_hash, embedding=e)
        for c, e in zip(chunks, embeddings, strict=True)
    ]
    if dry_run:
        return {"doc_id": doc_id, "revision_id": fetched.revision_id,
                "chunks": len(rows), "written": False}
    stats = PgVectorChunks().atomic_replace(fetched.doc_id, fetched.revision_id, rows)
    return {**stats, "written": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest Google Docs into pgvector.")
    parser.add_argument("--doc-id", action="append", default=[], help="Doc ID (repeatable).")
    parser.add_argument("--dry-run", action="store_true", help="Fetch/chunk/embed, write nothing.")
    parser.add_argument("--commit", action="store_true", help="Write to the DB.")
    args = parser.parse_args(argv)
    if not args.doc_id:
        parser.error("at least one --doc-id is required")
    if not args.dry_run and not args.commit:
        parser.error("pass --dry-run or --commit")
    for doc_id in args.doc_id:
        print(ingest_doc(doc_id, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
