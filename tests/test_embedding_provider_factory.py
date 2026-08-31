from __future__ import annotations

import pytest

from app.embeddings.embedding_provider_factory import (
    EmbeddingProviderConfigurationError,
    create_embedding_provider,
)
from app.embeddings.fake_embedding_provider import FakeEmbeddingProvider
from app.embeddings.ollama_embedding_provider import (
    OllamaEmbeddingProvider,
)


def test_factory_defaults_to_fake_provider(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "EMBEDDING_PROVIDER",
        raising=False,
    )
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_DIMENSIONS",
        "8",
    )

    provider = create_embedding_provider()

    assert isinstance(provider, FakeEmbeddingProvider)
    assert provider.dimensions == 8


def test_factory_creates_ollama_provider(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "EMBEDDING_PROVIDER",
        "ollama",
    )
    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://ollama:11434",
    )
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_MODEL",
        "embeddinggemma",
    )
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_DIMENSIONS",
        "768",
    )
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_TIMEOUT_SECONDS",
        "300",
    )

    provider = create_embedding_provider()

    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider.model_name == "embeddinggemma"
    assert provider.dimensions == 768


def test_factory_rejects_invalid_dimensions(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_DIMENSIONS",
        "invalid",
    )

    with pytest.raises(
        EmbeddingProviderConfigurationError,
        match="must be an integer",
    ):
        create_embedding_provider()


def test_factory_rejects_non_positive_dimensions(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_DIMENSIONS",
        "0",
    )

    with pytest.raises(
        EmbeddingProviderConfigurationError,
        match="greater than zero",
    ):
        create_embedding_provider()


def test_factory_requires_ollama_model(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "EMBEDDING_PROVIDER",
        "ollama",
    )
    monkeypatch.delenv(
        "OLLAMA_EMBEDDING_MODEL",
        raising=False,
    )
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_DIMENSIONS",
        "768",
    )

    with pytest.raises(
        EmbeddingProviderConfigurationError,
        match="OLLAMA_EMBEDDING_MODEL",
    ):
        create_embedding_provider()


def test_factory_rejects_invalid_timeout(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "EMBEDDING_PROVIDER",
        "ollama",
    )
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_MODEL",
        "embeddinggemma",
    )
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_DIMENSIONS",
        "768",
    )
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_TIMEOUT_SECONDS",
        "invalid",
    )

    with pytest.raises(
        EmbeddingProviderConfigurationError,
        match="must be numeric",
    ):
        create_embedding_provider()


def test_factory_rejects_unknown_provider(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "EMBEDDING_PROVIDER",
        "unknown",
    )
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_DIMENSIONS",
        "768",
    )

    with pytest.raises(
        EmbeddingProviderConfigurationError,
        match="Unsupported embedding provider",
    ):
        create_embedding_provider()