from __future__ import annotations

import json

import httpx
from pydantic import ValidationError

from app.llm.skill_gap_provider import SkillGapGenerationProviderError
from app.schemas.skill_gap_report import SkillGapGenerationContext, SkillGapReportContent

SYSTEM_PROMPT = """
You are an evidence-based technical career preparation adviser.

Reassess preliminary missing job skills against the supplied candidate evidence,
then create a focused preparation report.

Rules:
- Evaluate only preliminary_missing_skills. Do not invent additional gaps.
- Recognize common abbreviations, spelling variants, product aliases, and clear
  semantic equivalents when candidate evidence supports them.
- If a requirement is supported under another name, put it in
  resolved_equivalences and do not list it as missing.
- Do not treat adjacent or related technology as hands-on evidence. For example,
  PyTorch does not prove TensorRT and Gazebo does not prove MuJoCo.
- Copy every job_requirement and missing skill exactly from the preliminary list.
- For every genuine gap, provide concise, practical preparation guidance.
- Prioritize required or repeatedly emphasized skills over optional skills.
- Never state that the candidate possesses a missing skill.
- Return JSON conforming exactly to the supplied schema.
""".strip()


class OllamaSkillGapGenerationProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 600.0,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._model = model.strip()
        self._timeout_seconds = timeout_seconds
        self._api_key = api_key.strip() if api_key else None
        self._client = client
        if not self._base_url or not self._model or timeout_seconds <= 0:
            raise ValueError("Valid Ollama skill-gap configuration is required")

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate_skill_gap_report(
        self, context: SkillGapGenerationContext
    ) -> SkillGapReportContent:
        schema = SkillGapReportContent.model_json_schema()
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_prompt(context, schema)},
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192},
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = await self._post(payload, headers)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            return SkillGapReportContent.model_validate_json(content)
        except httpx.TimeoutException as exc:
            raise SkillGapGenerationProviderError("Ollama skill-gap generation timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise SkillGapGenerationProviderError(
                f"Ollama returned HTTP {exc.response.status_code} for skill-gap generation"
            ) from exc
        except httpx.RequestError as exc:
            raise SkillGapGenerationProviderError("Unable to connect to Ollama") from exc
        except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise SkillGapGenerationProviderError(
                "Ollama returned invalid structured skill-gap content"
            ) from exc

    async def _post(self, payload: dict[str, object], headers: dict[str, str]) -> httpx.Response:
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
    def _build_prompt(context: SkillGapGenerationContext, schema: dict[str, object]) -> str:
        trusted_input = context.model_dump(mode="json")
        return (
            "Generate the skill-gap and preparation report from this trusted input.\n\n"
            f"Input JSON:\n{json.dumps(trusted_input, sort_keys=True)}\n\n"
            "Output schema:\n"
            f"{json.dumps(schema, sort_keys=True, separators=(',', ':'))}"
        )
