# Phase 10 — Personal Application Package

Phase 10 adds local generation of a grounded tailored resume and cover letter.
It intentionally does not add cover-letter persistence, migrations, or API
history endpoints.

## Environment

Add these values to `.env`:

```env
COVER_LETTER_LLM_PROVIDER=ollama
OLLAMA_COVER_LETTER_MODEL=qwen2.5:7b
OLLAMA_COVER_LETTER_TIMEOUT_SECONDS=300
```

The existing resume, job-requirement, embedding, database, and Ollama settings
must also remain configured.

## Generate from an existing job

Run inside `jobpilot-api`:

```bash
uv run python scripts/generate_application_package.py \
  --profile-id PROFILE_UUID \
  --job-id JOB_UUID
```

## Generate from a copied job description

Save the complete description as a UTF-8 text file, then run:

```bash
uv run python scripts/generate_application_package.py \
  --profile-id PROFILE_UUID \
  --job-file data/jobs/robotics-engineer.txt \
  --title "Senior Robotics Engineer" \
  --company "Example Company" \
  --location "Abu Dhabi" \
  --job-url "https://www.linkedin.com/jobs/view/JOB_ID"
```

LinkedIn is not scraped. The command uses the text stored in `--job-file`.

## Output

```text
outputs/company-role/
├── manifest.json
├── resume.json
├── resume.md
├── resume.pdf
├── cover-letter.json
├── cover-letter.md
└── cover-letter.pdf
```

The JSON files retain evidence provenance. Markdown and PDF contain only the
candidate-facing documents.

## Verification

```bash
uv run ruff check \
  app/schemas/cover_letter.py \
  app/llm/cover_letter_provider.py \
  app/llm/fake_cover_letter_provider.py \
  app/llm/ollama_cover_letter_provider.py \
  app/llm/cover_letter_provider_factory.py \
  app/services/cover_letter_generation_service.py \
  app/services/cover_letter_grounding_validator.py \
  app/services/cover_letter_markdown_renderer.py \
  app/services/cover_letter_pdf_renderer.py \
  scripts/generate_application_package.py \
  tests/test_cover_letter_phase10.py

uv run pytest tests/test_cover_letter_phase10.py -q
uv run pytest -q
```

Always review the generated Markdown or PDF before submitting an application.
