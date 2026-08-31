from __future__ import annotations

import json
import math

import httpx

from app.embeddings.embedding_provider import (
    EmbeddingProviderError,
)


class OllamaEmbeddingProvider:
    """Ollama-backed text embedding provider."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimensions: int,
        timeout_seconds: float = 300.0,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        cleaned_base_url = base_url.strip().rstrip("/")
        cleaned_model = model.strip()

        if not cleaned_base_url:
            raise ValueError("Ollama base URL must not be empty")

        if not cleaned_model:
            raise ValueError("Ollama embedding model must not be empty")

        if dimensions <= 0:
            raise ValueError(
                "Embedding dimensions must be greater than zero"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "Ollama embedding timeout must be greater than zero"
            )

        self._base_url = cleaned_base_url
        self._model = cleaned_model
        self._dimensions = dimensions
        self._timeout_seconds = timeout_seconds
        self._api_key = api_key.strip() if api_key else None
        self._client = client

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:
        vectors = await self.embed_batch([text])

        if len(vectors) != 1:
            raise EmbeddingProviderError(
                "Ollama returned an unexpected number of embeddings"
            )

        return vectors[0]

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")

        if not texts:
            return []

        cleaned_texts: list[str] = []

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("each embedding input must be a string")

            cleaned = text.strip()

            if not cleaned:
                raise ValueError(
                    "embedding input must not be empty or whitespace"
                )

            cleaned_texts.append(cleaned)

        headers = {
            "Content-Type": "application/json",
        }

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "input": cleaned_texts,
            "truncate": True,
        }

        try:
            response = await self._post(
                payload=payload,
                headers=headers,
            )
            response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise EmbeddingProviderError(
                "Ollama timed out while generating embeddings"
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise EmbeddingProviderError(
                f"Ollama returned HTTP "
                f"{exc.response.status_code} while generating embeddings"
            ) from exc

        except httpx.RequestError as exc:
            raise EmbeddingProviderError(
                "Unable to connect to Ollama embedding service"
            ) from exc

        try:
            response_data = response.json()
        except json.JSONDecodeError as exc:
            raise EmbeddingProviderError(
                "Ollama returned an invalid JSON embedding response"
            ) from exc

        error_message = response_data.get("error")

        if error_message:
            raise EmbeddingProviderError(
                f"Ollama embedding error: {error_message}"
            )

        embeddings = response_data.get("embeddings")

        if not isinstance(embeddings, list):
            raise EmbeddingProviderError(
                "Ollama response does not contain embeddings"
            )

        if len(embeddings) != len(cleaned_texts):
            raise EmbeddingProviderError(
                "Ollama returned an unexpected number of embeddings"
            )

        validated: list[list[float]] = []

        for vector in embeddings:
            validated.append(
                self._validate_vector(vector)
            )

        return validated

    async def _post(
        self,
        *,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> httpx.Response:
        url = f"{self._base_url}/api/embed"

        if self._client is not None:
            return await self._client.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._timeout_seconds,
            )

        async with httpx.AsyncClient() as client:
            return await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._timeout_seconds,
            )

    def _validate_vector(
        self,
        vector: object,
    ) -> list[float]:
        if not isinstance(vector, list):
            raise EmbeddingProviderError(
                "Ollama returned a non-list embedding"
            )

        if len(vector) != self._dimensions:
            raise EmbeddingProviderError(
                "Ollama embedding dimensions do not match configuration: "
                f"expected {self._dimensions}, got {len(vector)}"
            )

        converted: list[float] = []

        for value in vector:
            if not isinstance(value, int | float):
                raise EmbeddingProviderError(
                    "Ollama embedding contains a non-numeric value"
                )

            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                raise EmbeddingProviderError(
                    "Ollama embedding contains a non-finite value"
                )

            converted.append(numeric_value)

        return converted