# Phase 11 — Streamlit Web Interface

## Start

```powershell
docker compose up -d --build
docker compose ps
```

Open:

- Web UI: <http://localhost:8501>
- FastAPI documentation: <http://localhost:8500/docs>

## User workflow

1. Select an existing candidate profile or create one.
2. Upload a PDF or DOCX master resume.
3. Click **Upload and prepare profile**.
4. Open **New application**.
5. Paste the URL, job metadata, and complete description.
6. Click **Generate resume and cover letter**.
7. Review the match, documents, and provenance.
8. Download Markdown or PDF.

LinkedIn is not scraped. The complete description must be pasted into the form.

## Verification

```powershell
docker compose exec jobpilot-api uv run pytest tests/test_phase11_web_contract.py -q
docker compose exec jobpilot-api uv run pytest -q
```

Phase 12 will add persisted application tracking and status dashboards.
