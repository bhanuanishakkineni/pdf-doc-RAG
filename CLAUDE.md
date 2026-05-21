# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Package management uses `uv` (Python >= 3.13).

```bash
# Install / sync deps
uv sync

# Run the API server (auto-reload on edits)
uvicorn main:app --reload

# Run the Inngest-flavored app (separate FastAPI instance in inngest_app.py)
uvicorn inngest_app:app_inngest --reload --port 8001

# Inngest dev UI, pointed at the running server's /api/inngest route
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery

# Qdrant (required dependency)
docker run -d --name qdrant-vectordb -p 6333:6333 -v ./qdrant:/qdrant/storage qdrant/qdrant
```

No test suite or linter is configured.

## Required environment

`.env` must define `OPENAI_API_KEY`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `S3_BUCKET_NAME`. Loaded via `python-dotenv` at module import time in both `main.py` and `data_loader.py`. Qdrant is hardcoded to `http://localhost:6333` in `vector_db.py`.

## Architecture

This is a RAG pipeline with **two parallel implementations** of the same ingest/query flow:

1. **`main.py`** — synchronous REST endpoints (`/ingest`, `/query`, `/docs/presign`, `/docs/upload`, `/health`). This is the actively wired app (`uvicorn main:app`). Helper functions `_load`, `_upsert`, `_search` call the same shared modules described below.
2. **`inngest_app.py`** — event-driven version of the same logic exposed as Inngest functions (`qa-rag-project/ingest_pdf`, `qa-rag-project/query_pdf`) on a separate FastAPI instance `app_inngest`. Inngest serializes step results via `PydanticSerializer`, so step return types must be pydantic models defined in `custom_types.py`. **The two apps are not mounted together** — `main.py` does not import or include the Inngest routes.

Shared core (used by both):

- **`data_loader.py`** — `load_and_chunk_pdf(path)` reads with `PDFReader` and splits with `SentenceSplitter(chunk_size=1000, chunk_overlap=200)`. `embed_texts(list[str])` calls OpenAI `text-embedding-3-large` (3072-dim). The OpenAI `client` is exported from this module and reused by `main.py` for chat completions.
- **`vector_db.py`** — `QdrantVectorDB` wraps the Qdrant client. Collection name `docs`, vector dim **3072** (must match `EMBED_DIM` in `data_loader.py`), distance cosine. Auto-creates the collection on first instantiation. `search()` returns `{"contexts": [...], "sources": [...]}` (note: `sources` is a list deduped from a set, so order is not stable).
- **`custom_types.py`** — pydantic models for request/response and Inngest step payloads. The Inngest serializer requires step inputs/outputs to be these types.

PDF upload flow is **S3-mediated**: client calls `/docs/presign` to get an upload URL + key, uploads the PDF directly to S3, then calls `/ingest` with that key. The `/docs/upload` endpoint exists for local testing only — production callers should PUT to the presigned URL themselves. Note that `_load()` in `main.py` passes the S3 `pdf_key` straight to `load_and_chunk_pdf`, which expects a local file path — bridging S3 → local path is not yet implemented.

## Things to know when editing

- **Embedding dim is duplicated** in `data_loader.py` (`EMBED_DIM = 3072`) and `vector_db.py` (`dim=3072` default). Changing the embedding model means updating both *and* recreating the Qdrant collection (the existing collection won't auto-migrate).
- **Two code paths to keep in sync**: the same load/embed/upsert logic exists in `main.py._load`/`_upsert`/`_search` *and* inside the Inngest functions in `inngest_app.py`. Bug fixes to one likely need mirroring in the other.
- The OpenAI chat model is referenced as `"gpt-5.4-mini"` in both `main.py` and `inngest_app.py` — if that's wrong upstream, both call sites need updating.
- CORS in `main.py` is configured for `http://localhost:5173` (Vite default), suggesting an external frontend.
