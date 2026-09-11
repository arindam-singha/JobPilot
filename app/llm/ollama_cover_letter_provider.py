from __future__ import annotations

import json
from uuid import UUID

import httpx
from pydantic import ValidationError

from app.llm.cover_letter_provider import CoverLetterGenerationProviderError
from app.schemas.cover_letter import (
    CoverLetterGenerationContext,
    GroundedCoverLetterParagraph,
    TailoredCoverLetterContent,
)

SYSTEM_PROMPT = """
You write concise, factual, tailored cover letters.

Rules:
- Use only facts explicitly present in the supplied candidate evidence.
- Never invent employers, roles, dates, durations, skills, metrics, education,
  certifications, achievements, responsibilities, motivations, or outcomes.
- Every opening, body, and closing paragraph must cite one or more evidence_ids.
- Cite only evidence that supports the complete paragraph.
- Do not claim missing job requirements.
- Preserve all numeric values exactly as written in evidence.
- Use first person, a professional natural tone, and plain text.
- Keep the letter concise: one opening, one to three body paragraphs, and one closing.
- Do not use markdown, tables, lists, placeholders, or bracketed text.
- Return JSON conforming exactly to the supplied schema.
- The candidate's name is supplied as candidate_name.
- Never invent or substitute a person's name.
- Do not include a person's name inside opening, body, or closing paragraphs.
- The generated sign_off is ignored by the application.
""".strip()


class OllamaCoverLetterGenerationProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str = "qwen2.5:7b",
        timeout_seconds: float = 300.0,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._model = model.strip()
        self._timeout_seconds = timeout_seconds
        self._api_key = api_key.strip() if api_key else None
        self._client = client
        if not self._base_url or not self._model:
            raise ValueError("Ollama base URL and model are required")
        if timeout_seconds <= 0:
            raise ValueError("Ollama timeout must be greater than zero")

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate_cover_letter(
        self,
        context: CoverLetterGenerationContext,
    ) -> TailoredCoverLetterContent:
        schema = TailoredCoverLetterContent.model_json_schema()
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_prompt(context, schema)},
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0,
                        },
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = await self._post(payload, headers)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise CoverLetterGenerationProviderError("Ollama timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise CoverLetterGenerationProviderError(
                f"Ollama returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise CoverLetterGenerationProviderError("Unable to connect to Ollama") from exc

        try:
            response_data = response.json()
            content = response_data["message"]["content"]
            result = TailoredCoverLetterContent.model_validate_json(content)
        except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise CoverLetterGenerationProviderError(
                "Ollama returned invalid structured cover-letter content"
            ) from exc

        self._validate_references(result, context)
        return result

    async def _post(
        self,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> httpx.Response:
        url = f"{self._base_url}/api/chat"
        if self._client is not None:
            return await self._client.post(
                url, json=payload, headers=headers, timeout=self._timeout_seconds
            )
        async with httpx.AsyncClient() as client:
            return await client.post(
                url, json=payload, headers=headers, timeout=self._timeout_seconds
            )

    @staticmethod
    def _build_prompt(
        context: CoverLetterGenerationContext,
        schema: dict[str, object],
    ) -> str:
        evidence = [
            {
                "evidence_id": str(item.evidence_id),
                "title": item.title,
                "content": item.content,
                "relevant_requirements": [trace.requirement for trace in item.requirement_traces],
            }
            for item in context.grounding.selected_evidence
        ]
        trusted_input = {
            "job": {
                "title": context.job_title,
                "company": context.company,
                "location": context.location,
                "description": context.job_description,
                "matched_requirements": context.grounding.matched_requirements,
            },
            "candidate_name": context.header.full_name,
            "evidence_catalog": evidence,
        }
        return (
            "Generate a tailored cover letter from this trusted input.\n\n"
            f"Input JSON:\n{json.dumps(trusted_input, sort_keys=True)}\n\n"
            "Output schema:\n"
            f"{json.dumps(schema, sort_keys=True, separators=(',', ':'))}"
        )

    @staticmethod
    def _validate_references(
        result: TailoredCoverLetterContent,
        context: CoverLetterGenerationContext,
    ) -> None:
        allowed = {item.evidence_id for item in context.grounding.selected_evidence}
        paragraphs: list[GroundedCoverLetterParagraph] = [
            result.opening,
            *result.body_paragraphs,
            result.closing,
        ]
        unknown: set[UUID] = {
            evidence_id
            for paragraph in paragraphs
            for evidence_id in paragraph.evidence_ids
            if evidence_id not in allowed
        }
        if unknown:
            raise CoverLetterGenerationProviderError(
                "Ollama cited evidence outside the trusted grounding bundle"
            )
