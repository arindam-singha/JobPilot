from __future__ import annotations

import json

import httpx
import pytest
from app.embeddings.embedding_provider import EmbeddingProviderError
from app.embeddings.ollama_embedding_provider import (
    OllamaEmbeddingProvider,
)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )


@pytest.mark.asyncio
async def test_ollama_provider_returns_single_embedding() -> None:
    vector = [0.1] * 768

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"

        return httpx.Response(
            200,
            json={"embeddings": [vector]},
        )

    async with _client(handler) as client:
        provider = OllamaEmbeddingProvider(
            base_url="http://ollama:11434",
            model="embeddinggemma",
            dimensions=768,
            client=client,
        )

        result = await provider.embed_text("Python")

    assert result == vector
    assert provider.provider_name == "ollama"
    assert provider.model_name == "embeddinggemma"
    assert provider.dimensions == 768


@pytest.mark.asyncio
async def test_ollama_provider_embeds_batch() -> None:
    first = [0.1] * 768
    second = [0.2] * 768

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())

        assert payload["model"] == "embeddinggemma"
        assert payload["input"] == ["Python", "ROS2"]
        assert payload["truncate"] is True

        return httpx.Response(
            200,
            json={"embeddings": [first, second]},
        )

    async with _client(handler) as client:
        provider = OllamaEmbeddingProvider(
            base_url="http://ollama:11434",
            model="embeddinggemma",
            dimensions=768,
            client=client,
        )

        result = await provider.embed_batch(
            ["Python", "ROS2"]
        )

    assert result == [first, second]


@pytest.mark.asyncio
async def test_ollama_provider_returns_empty_batch_without_request() -> None:
    provider = OllamaEmbeddingProvider(
        base_url="http://ollama:11434",
        model="embeddinggemma",
        dimensions=768,
    )

    assert await provider.embed_batch([]) == []


@pytest.mark.asyncio
async def test_ollama_provider_rejects_empty_input() -> None:
    provider = OllamaEmbeddingProvider(
        base_url="http://ollama:11434",
        model="embeddinggemma",
        dimensions=768,
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        await provider.embed_text("   ")


@pytest.mark.asyncio
async def test_ollama_provider_rejects_wrong_dimensions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[0.1] * 10]},
        )

    async with _client(handler) as client:
        provider = OllamaEmbeddingProvider(
            base_url="http://ollama:11434",
            model="embeddinggemma",
            dimensions=768,
            client=client,
        )

        with pytest.raises(
            EmbeddingProviderError,
            match="dimensions do not match",
        ):
            await provider.embed_text("Python")


@pytest.mark.asyncio
async def test_ollama_provider_rejects_wrong_embedding_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[0.1] * 768]},
        )

    async with _client(handler) as client:
        provider = OllamaEmbeddingProvider(
            base_url="http://ollama:11434",
            model="embeddinggemma",
            dimensions=768,
            client=client,
        )

        with pytest.raises(
            EmbeddingProviderError,
            match="unexpected number",
        ):
            await provider.embed_batch(
                ["Python", "ROS2"]
            )


@pytest.mark.asyncio
async def test_ollama_provider_rejects_missing_embeddings() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with _client(handler) as client:
        provider = OllamaEmbeddingProvider(
            base_url="http://ollama:11434",
            model="embeddinggemma",
            dimensions=768,
            client=client,
        )

        with pytest.raises(
            EmbeddingProviderError,
            match="does not contain embeddings",
        ):
            await provider.embed_text("Python")


@pytest.mark.asyncio
async def test_ollama_provider_rejects_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": "model failure"},
        )

    async with _client(handler) as client:
        provider = OllamaEmbeddingProvider(
            base_url="http://ollama:11434",
            model="embeddinggemma",
            dimensions=768,
            client=client,
        )

        with pytest.raises(
            EmbeddingProviderError,
            match="HTTP 500",
        ):
            await provider.embed_text("Python")


@pytest.mark.asyncio
async def test_ollama_provider_rejects_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": "model not found"},
        )

    async with _client(handler) as client:
        provider = OllamaEmbeddingProvider(
            base_url="http://ollama:11434",
            model="embeddinggemma",
            dimensions=768,
            client=client,
        )

        with pytest.raises(
            EmbeddingProviderError,
            match="model not found",
        ):
            await provider.embed_text("Python")


def test_ollama_provider_rejects_empty_base_url() -> None:
    with pytest.raises(
        ValueError,
        match="base URL must not be empty",
    ):
        OllamaEmbeddingProvider(
            base_url=" ",
            model="embeddinggemma",
            dimensions=768,
        )


def test_ollama_provider_rejects_empty_model() -> None:
    with pytest.raises(
        ValueError,
        match="model must not be empty",
    ):
        OllamaEmbeddingProvider(
            base_url="http://ollama:11434",
            model=" ",
            dimensions=768,
        )


def test_ollama_provider_rejects_invalid_dimensions() -> None:
    with pytest.raises(
        ValueError,
        match="dimensions must be greater than zero",
    ):
        OllamaEmbeddingProvider(
            base_url="http://ollama:11434",
            model="embeddinggemma",
            dimensions=0,
        )