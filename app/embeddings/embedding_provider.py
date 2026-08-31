from __future__ import annotations

from typing import Protocol


class EmbeddingProviderError(Exception):
    """Base exception for embedding-provider failures."""


class EmbeddingProvider(Protocol):
    """Interface implemented by text-embedding providers."""

    @property
    def provider_name(self) -> str:
        """Stable provider identifier."""

    @property
    def model_name(self) -> str:
        """Embedding model identifier."""

    @property
    def dimensions(self) -> int:
        """Expected vector dimensionality."""

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """Generate one embedding vector."""

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""