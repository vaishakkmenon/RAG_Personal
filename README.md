# Personal-Docs RAG Bot

## What this is
Private RAG assistant over my resume, transcripts, cert PDFs, and write-ups. Evidence-first; cites sources; abstains with "I don't know." when evidence is missing.

## Run mode (Phase 0)
- Mode: Local (Python + local model backend)
- Requirements: Python 3.11+, Ollama (or your chosen local model host)

## Quick start
1) Ensure directories exist: `data/docs/personal/pdfs`, `data/docs/personal/mds/`, `data/vectordb`.
2) (No ingestion yet — that starts in Phase 2 after Phase 1 data prep.)

## Repository layout
- `app/` — API/server and RAG logic
- `scripts/` — utilities (ingestion, eval, maintenance)
- `data/docs/personal/pdfs` — original PDFs (untracked)
- `data/docs/personal/mds/` — cleaned Markdown (tracked)
- `data/vectordb/` — local vector database

## Next steps
- Phase 1: Convert/redact PDFs → clean Markdown (in `data/md/`) with consistent headings.