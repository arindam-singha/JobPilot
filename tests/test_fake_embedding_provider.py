from __future__ import annotations

import math

import pytest
from app.embeddings.fake_embedding_provider import FakeEmbeddingProvider


def test_fake_provider_has_stable_identity() -> None:
    provider = FakeEmbeddingProvider(dimensions=8)

    assert provider.provider_name == "fake"
    assert provider.model_name == "fake-embedding-model"
    assert provider.dimensions == 8


@pytest.mark.asyncio
async def test_fake_provider_returns_expected_dimensions() -> None:
    provider = FakeEmbeddingProvider(dimensions=8)

    vector = await provider.embed_text("Python and PyTorch")

    assert len(vector) == 8
    assert all(isinstance(value, float) for value in vector)


@pytest.mark.asyncio
async def test_fake_provider_returns_normalized_vector() -> None:
    provider = FakeEmbeddingProvider(dimensions=8)

    vector = await provider.embed_text("Computer vision")

    magnitude = math.sqrt(sum(value * value for value in vector))

    assert magnitude == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_fake_provider_is_deterministic() -> None:
    provider = FakeEmbeddingProvider(dimensions=8)

    first = await provider.embed_text("Python")
    second = await provider.embed_text("Python")

    assert first == second


@pytest.mark.asyncio
async def test_fake_provider_returns_different_vectors_for_different_text() -> None:
    provider = FakeEmbeddingProvider(dimensions=8)

    first = await provider.embed_text("Python")
    second = await provider.embed_text("ROS2")

    assert first != second


@pytest.mark.asyncio
async def test_fake_provider_embeds_batch() -> None:
    provider = FakeEmbeddingProvider(dimensions=8)

    vectors = await provider.embed_batch(
        [
            "Python",
            "ROS2",
            "Computer Vision",
        ]
    )

    assert len(vectors) == 3
    assert all(len(vector) == 8 for vector in vectors)


@pytest.mark.asyncio
async def test_fake_provider_returns_empty_batch() -> None:
    provider = FakeEmbeddingProvider(dimensions=8)

    assert await provider.embed_batch([]) == []


@pytest.mark.asyncio
async def test_fake_provider_rejects_empty_text() -> None:
    provider = FakeEmbeddingProvider(dimensions=8)

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        await provider.embed_text("   ")


def test_fake_provider_rejects_invalid_dimensions() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        FakeEmbeddingProvider(dimensions=0)