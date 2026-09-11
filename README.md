# JobPilot

JobPilot is a human-in-the-loop, agentic RAG-based job application assistant. The application is designed to run locally through Docker, with a FastAPI backend, PostgreSQL database, Ollama model service, and a Streamlit front end.

This project is meant to be easy to clone and run on a fresh machine as long as Docker is installed and running.

## What this project includes

- FastAPI backend for job, profile, evidence, matching, and resume-generation workflows
- PostgreSQL database with pgvector support
- Ollama local model runtime for LLM-based features
- Streamlit UI for candidate profile review and application generation
- Alembic migration setup
- Test suite for backend behavior

---

## Requirements

Before you begin, install:

- Git
- Docker Desktop (Windows/macOS) or Docker Engine + Docker Compose (Linux)

Important:

- You do not need to install Python, PostgreSQL, or Node.js on your host machine for normal local use.
- Docker must be running before starting the app.

---

## 1) Clone the repository

```bash
git clone https://github.com/<your-user>/jobpilot.git
cd jobpilot
```

If you already have the repo locally, make sure you are in the project root.

---

## 2) Create your environment file

Copy the example environment file:

On Linux/macOS:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

This project uses the values in `.env` for the database and app settings. The included example is already set up for local Docker usage.

---

## 3) Build and start the stack

From the project root, run:

```bash
docker compose up --build
```

This starts:

- PostgreSQL on port `55432` on the host
- Ollama on the Docker network
- The JobPilot API on port `8500` on the host
- The Streamlit UI on port `8502` on the host

If you want to start in the background instead:

```bash
docker compose up --build -d
```

---

## 4) Check that the app is running

Once the containers are up, verify the API health endpoint:

```bash
curl http://localhost:8500/health
```

Expected response:

```json
{"status":"ok"}
```

You can also check the logs if startup seems slow:

```bash
docker compose logs -f jobpilot-api
```

If the LLM container is still pulling models, this may take a few minutes on first run.

---

## 5) Open the application

After startup completes, open these URLs in your browser:

- API docs: http://localhost:8500/docs
- OpenAPI schema: http://localhost:8500/openapi.json
- UI: http://localhost:8502

The web interface lets you create or select a profile, upload a resume, review extracted data, and generate resume/cover-letter outputs.

---

## 6) Stop the application

To stop the services:

```bash
docker compose down
```

To also remove the persistent database volume:

```bash
docker compose down -v
```

Use `-v` only if you want to fully wipe the local PostgreSQL data.

---

## 7) Useful commands

Rebuild after changes:

```bash
docker compose build
```

Restart only the API:

```bash
docker compose restart jobpilot-api
```

View running containers:

```bash
docker compose ps
```

Open a shell inside the API container:

```bash
docker compose exec jobpilot-api sh
```

Run tests inside the API container:

```bash
docker compose exec jobpilot-api pytest
```

Run a specific test:

```bash
docker compose exec jobpilot-api pytest tests/test_api.py
```

Apply migrations:

```bash
docker compose exec jobpilot-api alembic upgrade head
```

Create a new migration:

```bash
docker compose exec jobpilot-api alembic revision --autogenerate -m "message"
```

---

## 8) Project layout

```text
jobpilot/
├── app/
│   ├── agents/
│   ├── api/
│   ├── core/
│   ├── cv/
│   ├── db/
│   ├── embeddings/
│   ├── llm/
│   ├── matching/
│   ├── models/
│   ├── rag/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── data/
│   ├── candidate/
│   └── jobs/
├── scripts/
├── storage/
├── tests/
├── ui/
├── alembic/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 9) Common troubleshooting

### Docker is not running

Start Docker Desktop or your Docker service, then retry:

```bash
docker compose up --build
```

### Port already in use

Check whether another process is using `8500`, `8502`, or `55432`, then stop it or adjust the port mapping in `docker-compose.yml`.

### Ollama model download is slow

The first startup may take several minutes because the project pulls a model into the Ollama container. Wait for the container to finish initialization before using the LLM features.

### API not responding

Run:

```bash
docker compose logs --tail=100 jobpilot-api
```

Then check whether PostgreSQL and Ollama came up successfully.

---

## 10) Recommended first run

For a fresh clone, this is the simplest path:

```bash
git clone https://github.com/<your-user>/jobpilot.git
cd jobpilot
Copy-Item .env.example .env   # Windows PowerShell
# or: cp .env.example .env   # Linux/macOS

docker compose up --build -d
```

Then open:

- http://localhost:8502
- http://localhost:8500/docs

---

## 11) Notes for local development

- The app is configured to run primarily inside Docker.
- The backend is exposed on port `8500` and the UI on `8502` to avoid clashes with other local services.
- The database is persisted in Docker volumes, so your data remains available between restarts unless you run `docker compose down -v`.
- Some advanced features depend on Ollama and may require more time to initialize the first time.

This setup is intended to make a fresh clone usable with minimal setup and no manual installation of backend dependencies on the host machine.
