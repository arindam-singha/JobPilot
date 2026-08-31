FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/home/jobpilot/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

RUN useradd --create-home --home-dir /home/jobpilot --uid 10001 --user-group jobpilot

WORKDIR /app

COPY pyproject.toml README.md ./
COPY uv.lock ./

RUN mkdir -p /app/app /app/tests /app/alembic

COPY . .

RUN uv sync --frozen --no-install-project --dev

USER jobpilot

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
