from __future__ import annotations

import os

from app.llm.fake_skill_gap_provider import FakeSkillGapGenerationProvider
from app.llm.ollama_skill_gap_provider import OllamaSkillGapGenerationProvider
from app.llm.skill_gap_provider import SkillGapGenerationProvider


class SkillGapProviderConfigurationError(Exception):
    pass


def create_skill_gap_generation_provider() -> SkillGapGenerationProvider:
    provider_name = os.getenv("SKILL_GAP_LLM_PROVIDER", "fake").strip().casefold()
    if provider_name == "fake":
        return FakeSkillGapGenerationProvider()
    if provider_name != "ollama":
        raise SkillGapProviderConfigurationError(
            f"Unsupported skill-gap provider: {provider_name!r}"
        )
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").strip()
    model = os.getenv("OLLAMA_SKILL_GAP_MODEL", "qwen2.5:7b").strip()
    timeout_raw = os.getenv("OLLAMA_SKILL_GAP_TIMEOUT_SECONDS", "600").strip()
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError as exc:
        raise SkillGapProviderConfigurationError(
            "OLLAMA_SKILL_GAP_TIMEOUT_SECONDS must be numeric"
        ) from exc
    return OllamaSkillGapGenerationProvider(
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        api_key=os.getenv("OLLAMA_API_KEY", "").strip() or None,
    )
