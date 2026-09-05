# JobPilot

JobPilot is a human-in-the-loop, Agentic RAG-based job application assistant.

This repository currently contains the Phase 1 development foundation only. It includes a clean FastAPI application structure, SQLAlchemy 2 async PostgreSQL integration, Alembic migration scaffolding, Docker Compose orchestration, and the base test/linting configuration.

## Prerequisites

The host system should only require:

- Docker
- Git

The project runs entirely inside Docker for local development. No Python, PostgreSQL, Redis, Node.js, or other runtime dependencies should be installed on the host for this repository.

## Environment

Configuration is supplied via a `.env` file based on `.env.example`.

Copy the example file:

```bash
cp .env.example .env
```

The environment variables include:

- `DATABASE_URL`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `APP_ENV`

## Build

Build the API container:

```bash
docker compose build
```

## Start

Start the full stack:

```bash
docker compose up --build
```

The API is exposed on port `8000` and PostgreSQL is exposed on port `5432` from the host.

## Stop

Stop and remove containers:

```bash
docker compose down
```

If you also want to remove persistent PostgreSQL volume data:

```bash
docker compose down -v
```

## Enter the Container

To enter the API container:

```bash
docker compose exec jobpilot-api sh
```

## Running Tests

Run the test suite inside the API container:

```bash
docker compose exec jobpilot-api pytest
```

Or run a specific test:

```bash
docker compose exec jobpilot-api pytest tests/test_api.py
```

## Running Migrations

Alembic is configured for migrations. Use:

```bash
docker compose exec jobpilot-api alembic upgrade head
```

Create a new revision:

```bash
docker compose exec jobpilot-api alembic revision --autogenerate -m "message"
```

## Swagger Documentation

After the API is running, open:

http://localhost:8000/docs

The OpenAPI specification is also available at:

http://localhost:8000/openapi.json

## Project Layout

```text
jobpilot/
├── app/
│   ├── api/
│   ├── core/
│   ├── db/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   ├── agents/
│   ├── rag/
│   └── main.py
├── data/
│   ├── candidate/
│   └── jobs/
├── tests/
├── alembic/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

## Future Phase 2 Roadmap

Phase 2 should implement:

- Candidate profile ingestion and schema design
- Job ingestion from manually provided URLs or pasted job descriptions
- Vector/indexing and retrieval scaffolding for RAG
- Agent orchestration layer for analysis and decision support
- CV and cover letter generation models and templates
- Human-in-the-loop review workflow and application decision services
## Personal application package

After the candidate profile and evidence are populated, JobPilot can generate a
grounded resume and cover letter from an existing job:

```bash
uv run python scripts/generate_application_package.py \
  --profile-id PROFILE_UUID \
  --job-id JOB_UUID
```

Or ingest a manually copied job description and generate both documents:

```bash
uv run python scripts/generate_application_package.py \
  --profile-id PROFILE_UUID \
  --job-file data/jobs/robotics-engineer.txt \
  --title "Senior Robotics Engineer" \
  --company "Example Company" \
  --location "Abu Dhabi" \
  --job-url "https://example.com/jobs/robotics-engineer"
```

The command writes structured JSON with provenance, editable Markdown, and PDF
files beneath `outputs/<company>-<role>/`. Review every generated document
before applying. LinkedIn pages are not scraped; paste the complete description
into the job file.
## Web interface

Start the API and Streamlit interface:

```bash
docker compose up -d --build
```

Open `http://localhost:8501`. The guided workflow lets a user select or create a
profile, upload and prepare a PDF/DOCX master resume, paste a complete job
description, review the match, and download grounded resume and cover-letter
Markdown/PDF files. Job URLs are stored for reference; LinkedIn is not scraped.
