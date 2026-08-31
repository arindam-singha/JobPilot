from __future__ import annotations

import hashlib
import math


class FakeEmbeddingProvider:
    """Deterministic embedding provider for tests."""

    def __init__(
        self,
        *,
        dimensions: int = 8,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        self._dimensions = dimensions

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-embedding-model"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        cleaned = text.strip()

        if not cleaned:
            raise ValueError("text must not be empty or whitespace")

        digest = hashlib.sha256(
            cleaned.encode("utf-8")
        ).digest()

        values = [
            float(digest[index % len(digest)]) - 127.5
            for index in range(self._dimensions)
        ]

        magnitude = math.sqrt(
            sum(value * value for value in values)
        )

        if magnitude == 0:
            return [0.0] * self._dimensions

        return [
            value / magnitude
            for value in values
        ]

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        return [
            await self.embed_text(text)
            for text in texts
        ]