from __future__ import annotations

import json
import math
import re

import httpx
from pydantic import ValidationError

from app.llm.job_requirements_provider import (
    JobRequirementsProviderError,
)
from app.schemas.job_requirements import JobRequirementsData

SYSTEM_PROMPT = """
You extract structured hiring requirements from job descriptions.

Return only information explicitly supported by the supplied job title,
company, location, and description.

Rules:
- Do not invent skills, responsibilities, education, certifications,
  experience, or years of experience.
- Put mandatory skills in required_skills.
- Put optional, preferred, desirable, or nice-to-have skills in
  preferred_skills.
- required_experience should describe experience areas, not individual tools.
- responsibilities should contain the role's expected duties.
- education_requirements should contain degrees or academic requirements.
- certifications should contain explicitly requested certifications.
- domain_keywords should contain concise industry or technical domains.
- minimum_experience_years must be null unless a numeric minimum is explicit.
- Do not duplicate entries.
- Preserve important technology names such as Python, ROS2, PyTorch,
  FastAPI, Docker, Kubernetes, PostgreSQL, Isaac Sim, and RAG.
- Return empty lists when a category is not present.
- Extract every explicit item under headings such as Required Skills,
  Qualifications, Requirements, Responsibilities, Education, and Experience.
- Do not return all matchable categories empty when the description contains
  explicit qualifications or requirements.
- Separate degrees, experience areas, tools, and responsibilities into their
  corresponding schema fields.
- minimum_experience_years must be a finite JSON number such as 5, never
  Infinity, NaN, an inequality, or a string.
""".strip()


class OllamaJobRequirementsProvider:
    """Ollama-backed structured job-requirement extraction provider."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
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

    async def extract_requirements(
        self,
        *,
        title: str,
        company: str,
        location: str | None,
        description: str,
    ) -> JobRequirementsData:
        schema = JobRequirementsData.model_json_schema()
        normalized_description = self._normalize_description(description)

        prompt = self._build_prompt(
            title=title,
            company=company,
            location=location,
            description=normalized_description,
            schema=schema,
        )

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "format": schema,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }

        headers = {
            "Content-Type": "application/json",
        }

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = await self._post(
                payload=payload,
                headers=headers,
            )
            response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise JobRequirementsProviderError(
                "Ollama timed out while extracting job requirements"
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise JobRequirementsProviderError(
                f"Ollama returned HTTP {exc.response.status_code} "
                "while extracting job requirements"
            ) from exc

        except httpx.RequestError as exc:
            raise JobRequirementsProviderError(
                "Unable to connect to Ollama while extracting " "job requirements"
            ) from exc

        try:
            response_data = response.json()
        except json.JSONDecodeError as exc:
            return self._fallback_or_raise(
                description=normalized_description,
                message="Ollama returned an invalid JSON response",
                cause=exc,
            )

        if not isinstance(response_data, dict):
            return self._fallback_or_raise(
                description=normalized_description,
                message="Ollama returned an invalid response object",
            )

        error_message = response_data.get("error")

        if error_message:
            raise JobRequirementsProviderError(f"Ollama error: {error_message}")

        message = response_data.get("message")

        if not isinstance(message, dict):
            return self._fallback_or_raise(
                description=normalized_description,
                message="Ollama response does not contain a message",
            )

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            return self._fallback_or_raise(
                description=normalized_description,
                message="Ollama response does not contain structured content",
            )

        try:
            result = self._parse_content(
                content=content,
                description=normalized_description,
            )
        except (ValueError, ValidationError) as exc:
            return self._fallback_or_raise(
                description=normalized_description,
                message=(
                    "Ollama returned content that does not match "
                    "the job-requirements schema"
                ),
                cause=exc,
            )

        if (
            self._description_has_explicit_requirements(normalized_description)
            and self._matchable_requirement_count(result) == 0
        ):
            result = self._apply_deterministic_fallback(
                result=result,
                description=normalized_description,
            )

        if (
            self._description_has_explicit_requirements(normalized_description)
            and self._matchable_requirement_count(result) == 0
        ):
            raise JobRequirementsProviderError(
                "Neither Ollama nor the deterministic parser found "
                "matchable requirements"
            )

        return result

    @classmethod
    def _parse_content(
        cls,
        *,
        content: str,
        description: str,
    ) -> JobRequirementsData:
        payload = json.loads(content)

        if not isinstance(payload, dict):
            raise ValueError("Structured content must be a JSON object")

        raw_years = payload.get("minimum_experience_years")
        explicit_years = cls._extract_minimum_experience_years(description)

        if raw_years is None:
            payload["minimum_experience_years"] = explicit_years
        else:
            try:
                numeric_years = float(raw_years)
            except (TypeError, ValueError):
                numeric_years = math.nan

            if not math.isfinite(numeric_years):
                payload["minimum_experience_years"] = explicit_years

        return JobRequirementsData.model_validate(payload)

    @staticmethod
    def _normalize_description(description: str) -> str:
        normalized = description.replace("•", "\n- ")
        normalized = normalized.replace("\u00a0", " ")
        return normalized.strip()

    @staticmethod
    def _extract_minimum_experience_years(
        description: str,
    ) -> float | None:
        matches = re.findall(
            r"(?<!\d)(\d+(?:\.\d+)?)\s*\+?\s*years?\b",
            description,
            flags=re.IGNORECASE,
        )

        if not matches:
            return None

        return min(float(value) for value in matches)

    @staticmethod
    def _description_has_explicit_requirements(
        description: str,
    ) -> bool:
        return bool(
            re.search(
                r"\b(required|requirements?|qualifications?|"
                r"must have|experience|skills?)\b",
                description,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _matchable_requirement_count(
        result: JobRequirementsData,
    ) -> int:
        return sum(
            len(items)
            for items in (
                result.required_skills,
                result.preferred_skills,
                result.required_experience,
                result.education_requirements,
                result.certifications,
            )
        )

    @classmethod
    def _apply_deterministic_fallback(
        cls,
        *,
        result: JobRequirementsData,
        description: str,
    ) -> JobRequirementsData:
        """Recover explicit requirements without inventing text."""
        extracted = cls._extract_section_bullets(description)
        payload = result.model_dump()

        for field_name, items in extracted.items():
            if not payload.get(field_name):
                payload[field_name] = items

        if payload.get("minimum_experience_years") is None:
            payload["minimum_experience_years"] = cls._extract_minimum_experience_years(
                description
            )

        return JobRequirementsData.model_validate(payload)

    @classmethod
    def _extract_section_bullets(
        cls,
        description: str,
    ) -> dict[str, list[str]]:
        extracted: dict[str, list[str]] = {
            "required_skills": [],
            "preferred_skills": [],
            "required_experience": [],
            "responsibilities": [],
            "education_requirements": [],
            "certifications": [],
        }
        current_section: str | None = None

        for raw_line in description.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            is_bullet = line.startswith(("-", "*"))
            heading = line.rstrip(":").strip()
            if not is_bullet and cls._looks_like_heading(line):
                classified_heading = cls._classify_heading(heading)
                if classified_heading is not None:
                    current_section = classified_heading
                    continue

            item = line.lstrip("-* ").strip() if is_bullet else line
            if not item:
                continue

            if current_section is None:
                if not cls._looks_like_requirement_line(item):
                    continue
                target = cls._classify_unheaded_item(item)
            elif current_section == "required_qualifications":
                target = cls._classify_required_qualification(item)
            else:
                target = current_section

            cls._append_unique(extracted[target], item)

        return extracted

    @staticmethod
    def _classify_heading(heading: str) -> str | None:
        lowered = heading.casefold()

        if any(
            marker in lowered
            for marker in ("responsibilit", "duties", "what you will do")
        ):
            return "responsibilities"
        if any(
            marker in lowered for marker in ("preferred", "nice to have", "desirable")
        ):
            return "preferred_skills"
        if "certification" in lowered:
            return "certifications"
        if "education" in lowered:
            return "education_requirements"
        if any(
            marker in lowered
            for marker in (
                "required",
                "qualification",
                "requirements",
                "must have",
                "skills",
            )
        ):
            return "required_qualifications"
        return None

    @staticmethod
    def _classify_required_qualification(item: str) -> str:
        lowered = item.casefold()

        if re.search(
            r"\b(ph\.?d\.?|doctorate|master(?:'s|s)?|bachelor(?:'s|s)?|"
            r"degree|diploma)\b",
            lowered,
        ):
            return "education_requirements"
        if "certif" in lowered or "licen" in lowered:
            return "certifications"
        if re.search(
            r"\b(experience|years?|background|track record)\b",
            lowered,
        ):
            return "required_experience"
        return "required_skills"

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        stripped = line.strip()
        if stripped.endswith(":"):
            return True

        return len(stripped.split()) <= 8 and not stripped.endswith((".", "!", "?"))

    @staticmethod
    def _looks_like_requirement_line(item: str) -> bool:
        return bool(
            re.match(
                r"^(?:have|has|are|be|can|must|should|need|needs|"
                r"prefer|love|know|understand|possess|demonstrate|"
                r"experience|proficiency|knowledge|ability|degree|"
                r"familiarity|comfortable|fluen(?:t|cy)|if applying|"
                r"you will|you'll|responsible for|design|develop|build|"
                r"lead|manage|collaborate)\b",
                item,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _classify_unheaded_item(cls, item: str) -> str:
        lowered = item.casefold()

        if re.match(
            r"^(?:you will|you'll|responsible for|design|develop|build|"
            r"lead|manage|collaborate)\b",
            lowered,
        ):
            return "responsibilities"
        if any(
            marker in lowered
            for marker in ("preferred qualification", "nice to have", "ideally")
        ):
            return "preferred_skills"
        return cls._classify_required_qualification(item)

    @classmethod
    def _fallback_or_raise(
        cls,
        *,
        description: str,
        message: str,
        cause: Exception | None = None,
    ) -> JobRequirementsData:
        fallback = cls._apply_deterministic_fallback(
            result=JobRequirementsData(),
            description=description,
        )

        if cls._matchable_requirement_count(fallback) > 0:
            return fallback

        error = JobRequirementsProviderError(message)
        if cause is not None:
            raise error from cause
        raise error

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        normalized = value.casefold()
        if all(existing.casefold() != normalized for existing in items):
            items.append(value)

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
        title: str,
        company: str,
        location: str | None,
        description: str,
        schema: dict[str, object],
    ) -> str:
        location_text = location or "Not specified"
        schema_text = json.dumps(
            schema,
            sort_keys=True,
            separators=(",", ":"),
        )

        return (
            "Extract structured job requirements from the following job.\n\n"
            f"Title: {title}\n"
            f"Company: {company}\n"
            f"Location: {location_text}\n\n"
            "Job description:\n"
            f"{description}\n\n"
            "Your response must conform to this JSON schema:\n"
            f"{schema_text}"
        )
