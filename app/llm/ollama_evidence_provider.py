from __future__ import annotations

import json

import httpx
from pydantic import ValidationError

from app.cv.models import TextChunk
from app.llm.evidence_provider import EvidenceExtractionProviderError
from app.schemas.llm_evidence import LlmChunkExtractionResult


SYSTEM_PROMPT = """
You extract factual candidate evidence from CV or resume text.

Return only evidence explicitly supported by the supplied text.

Rules:
- Do not invent employers, dates, skills, metrics, qualifications, or results.
- Create separate evidence items when the text contains distinct facts.
- Keep each item concise but sufficiently descriptive.
- Use only these evidence_type values:
  summary, skill, experience, project, education, publication,
  certification, achievement, general.
- Confidence must be between 0.0 and 1.0.
- Return an empty evidence list when there is no useful candidate evidence.
""".strip()


class OllamaEvidenceExtractionProvider:
    """Ollama-backed structured candidate-evidence extraction provider."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        cleaned_base_url = base_url.strip().rstrip("/")
        cleaned_model = model.strip()

        if not cleaned_base_url:
            raise ValueError("Ollama base URL must not be empty")

        if not cleaned_model:
            raise ValueError("Ollama model must not be empty")

        if timeout_seconds <= 0:
            raise ValueError("Ollama timeout must be greater than zero")

        self._base_url = cleaned_base_url
        self._model = cleaned_model
        self._timeout_seconds = timeout_seconds
        self._api_key = api_key.strip() if api_key else None
        self._client = client

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def extract_evidence(
        self,
        chunk: TextChunk,
    ) -> LlmChunkExtractionResult:
        section = chunk.section or "unclassified"

        user_prompt = (
            f"CV section: {section}\n"
            f"Chunk index: {chunk.index}\n\n"
            "Extract candidate evidence from this text:\n\n"
            f"{chunk.text}"
        )

        headers = {
            "Content-Type": "application/json",
        }

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "format": LlmChunkExtractionResult.model_json_schema(),
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }

        try:
            if self._client is not None:
                response = await self._client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self._base_url}/api/chat",
                        json=payload,
                        headers=headers,
                        timeout=self._timeout_seconds,
                    )

            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise EvidenceExtractionProviderError(
                f"Ollama timed out while processing chunk {chunk.index}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EvidenceExtractionProviderError(
                f"Ollama returned HTTP {exc.response.status_code} "
                f"for chunk {chunk.index}"
            ) from exc
        except httpx.RequestError as exc:
            raise EvidenceExtractionProviderError(
                f"Unable to connect to Ollama for chunk {chunk.index}"
            ) from exc

        try:
            response_data = response.json()
        except json.JSONDecodeError as exc:
            raise EvidenceExtractionProviderError(
                "Ollama returned an invalid JSON response"
            ) from exc

        error_message = response_data.get("error")

        if error_message:
            raise EvidenceExtractionProviderError(
                f"Ollama error: {error_message}"
            )

        message = response_data.get("message")

        if not isinstance(message, dict):
            raise EvidenceExtractionProviderError(
                "Ollama response does not contain a message"
            )

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise EvidenceExtractionProviderError(
                "Ollama response does not contain structured content"
            )

        try:
            return LlmChunkExtractionResult.model_validate_json(content)
        except ValidationError as exc:
            raise EvidenceExtractionProviderError(
                "Ollama returned content that does not match "
                "the evidence schema"
            ) from exc