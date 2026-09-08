from __future__ import annotations

import json
from uuid import UUID

import httpx
from pydantic import ValidationError

from app.llm.resume_provider import ResumeGenerationProviderError
from app.schemas.tailored_resume import (
    GroundedResumeStatement,
    ResumeGenerationContext,
    TailoredResumeContent,
)

SYSTEM_PROMPT = """
You tailor resume wording using verified candidate evidence.

Rules:
- Use only facts explicitly present in the supplied evidence_catalog.
- The job description describes the employer's requirements, not the
  candidate's capabilities.
- Never copy a technology, tool, qualification, employer, institution,
  responsibility or achievement from the job description unless it also
  appears explicitly in candidate evidence.
- Never invent or infer employers, roles, dates, durations, skills, metrics,
  education, certifications, achievements, responsibilities or outcomes.
- Never combine separate evidence records in a way that creates a new claim.
- Preserve employer names, institution names, dates and numeric values exactly.
- Do not omit supplied professional experience or education evidence.
- Use these conventional section headings only:
  Professional Experience
  Selected Projects
  Education
  Selected Publications
  Certifications
  Awards and Recognition
  Additional Information
- Do not create headings such as Technical Expertise or Core Technical Skills.
- Every statement must cite evidence_ids that directly support the complete
  statement.
- A valid evidence ID is not sufficient when its content does not support the
  statement.
- Do not mention missing_requirements as candidate capabilities.
- Use concise, single-column, ATS-compatible plain text.
- Return JSON conforming exactly to the supplied schema.
""".strip()


class OllamaResumeGenerationProvider:
    """Ollama-backed provider for grounded structured resume content."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "qwen2.5:7b",
        timeout_seconds: float = 300.0,
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

    async def generate_resume(
        self,
        context: ResumeGenerationContext,
    ) -> TailoredResumeContent:
        schema = TailoredResumeContent.model_json_schema()
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._build_prompt(context=context, schema=schema),
                },
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0,
                        "num_ctx": 8192,},
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = await self._post(payload=payload, headers=headers)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ResumeGenerationProviderError(
                "Ollama timed out while generating the tailored resume"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ResumeGenerationProviderError(
                f"Ollama returned HTTP {exc.response.status_code} "
                "while generating the tailored resume"
            ) from exc
        except httpx.RequestError as exc:
            raise ResumeGenerationProviderError(
                "Unable to connect to Ollama while generating the tailored resume"
            ) from exc

        try:
            response_data = response.json()
        except json.JSONDecodeError as exc:
            raise ResumeGenerationProviderError("Ollama returned an invalid JSON response") from exc

        error_message = response_data.get("error")
        if error_message:
            raise ResumeGenerationProviderError(f"Ollama error: {error_message}")

        message = response_data.get("message")
        if not isinstance(message, dict):
            raise ResumeGenerationProviderError("Ollama response does not contain a message")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ResumeGenerationProviderError(
                "Ollama response does not contain structured resume content"
            )

        try:
            resume = TailoredResumeContent.model_validate_json(content)
        except ValidationError as exc:
            raise ResumeGenerationProviderError(
                "Ollama returned content that does not match the resume schema"
            ) from exc

        self._validate_evidence_references(resume=resume, context=context)
        return resume

    async def _post(
        self,
        *,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> httpx.Response:
        url = f"{self._base_url}/api/chat"
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

    @staticmethod
    def _build_prompt(
        *,
        context: ResumeGenerationContext,
        schema: dict[str, object],
    ) -> str:
        evidence_catalog = [
            {
                "evidence_id": str(evidence.evidence_id),
                "evidence_type": evidence.evidence_type,
                "title": evidence.title,
                "content": evidence.content,
                "relevant_requirements": [
                    trace.requirement for trace in evidence.requirement_traces
                ],
            }
            for evidence in context.grounding.selected_evidence
        ]
        input_data = {
            "job": {
                "title": context.job_title,
                "company": context.company,
                "location": context.location,
                "description": context.job_description,
                "matched_requirements": context.grounding.matched_requirements,
                "missing_requirements": context.grounding.missing_requirements,
            },
            "candidate_name": context.header.full_name,
            "evidence_catalog": evidence_catalog,
        }
        return (
            "Generate structured tailored resume content from this trusted input.\n\n"
            f"Input JSON:\n{json.dumps(input_data, sort_keys=True)}\n\n"
            "Your response must conform to this JSON schema:\n"
            f"{json.dumps(schema, sort_keys=True, separators=(',', ':'))}"
        )

    @staticmethod
    def _validate_evidence_references(
        *,
        resume: TailoredResumeContent,
        context: ResumeGenerationContext,
    ) -> None:
        allowed_ids = {evidence.evidence_id for evidence in context.grounding.selected_evidence}
        statements: list[GroundedResumeStatement] = [
            *resume.professional_summary,
            *resume.skills,
            *(statement for section in resume.sections for statement in section.statements),
        ]
        unknown_ids: set[UUID] = {
            evidence_id
            for statement in statements
            for evidence_id in statement.evidence_ids
            if evidence_id not in allowed_ids
        }
        if unknown_ids:
            raise ResumeGenerationProviderError(
                "Ollama cited evidence outside the trusted grounding bundle"
            )
