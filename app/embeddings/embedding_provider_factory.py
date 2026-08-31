from __future__ import annotations

import os

from app.embeddings.embedding_provider import (
    EmbeddingProvider,
)
from app.embeddings.fake_embedding_provider import (
    FakeEmbeddingProvider,
)
from app.embeddings.ollama_embedding_provider import (
    OllamaEmbeddingProvider,
)


class EmbeddingProviderConfigurationError(Exception):
    """Raised when embedding-provider configuration is invalid."""


def create_embedding_provider() -> EmbeddingProvider:
    provider_name = os.getenv(
        "EMBEDDING_PROVIDER",
        "fake",
    ).strip().casefold()

    dimensions_raw = os.getenv(
        "OLLAMA_EMBEDDING_DIMENSIONS",
        "768",
    ).strip()

    try:
        dimensions = int(dimensions_raw)
    except ValueError as exc:
        raise EmbeddingProviderConfigurationError(
            "OLLAMA_EMBEDDING_DIMENSIONS must be an integer"
        ) from exc

    if dimensions <= 0:
        raise EmbeddingProviderConfigurationError(
            "OLLAMA_EMBEDDING_DIMENSIONS must be greater than zero"
        )

    if provider_name == "fake":
        return FakeEmbeddingProvider(
            dimensions=dimensions,
        )

    if provider_name == "ollama":
        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://ollama:11434",
        ).strip()

        model = os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            "",
        ).strip()

        timeout_raw = os.getenv(
            "OLLAMA_EMBEDDING_TIMEOUT_SECONDS",
            "300",
        ).strip()

        api_key = os.getenv(
            "OLLAMA_API_KEY",
            "",
        ).strip() or None

        if not base_url:
            raise EmbeddingProviderConfigurationError(
                "OLLAMA_BASE_URL is required"
            )

        if not model:
            raise EmbeddingProviderConfigurationError(
                "OLLAMA_EMBEDDING_MODEL is required"
            )

        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise EmbeddingProviderConfigurationError(
                "OLLAMA_EMBEDDING_TIMEOUT_SECONDS must be numeric"
            ) from exc

        if timeout_seconds <= 0:
            raise EmbeddingProviderConfigurationError(
                "OLLAMA_EMBEDDING_TIMEOUT_SECONDS must be greater than zero"
            )

        return OllamaEmbeddingProvider(
            base_url=base_url,
            model=model,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )

    raise EmbeddingProviderConfigurationError(
        f"Unsupported embedding provider: {provider_name!r}"
    )