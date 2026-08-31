from __future__ import annotations

import os

from app.llm.cover_letter_provider import CoverLetterGenerationProvider
from app.llm.fake_cover_letter_provider import FakeCoverLetterGenerationProvider
from app.llm.ollama_cover_letter_provider import OllamaCoverLetterGenerationProvider


class CoverLetterProviderConfigurationError(Exception):
    """Raised when cover-letter provider configuration is invalid."""


def create_cover_letter_generation_provider() -> CoverLetterGenerationProvider:
    provider_name = os.getenv("COVER_LETTER_LLM_PROVIDER", "fake").strip().casefold()
    if provider_name == "fake":
        return FakeCoverLetterGenerationProvider()
    if provider_name != "ollama":
        raise CoverLetterProviderConfigurationError(
            f"Unsupported cover-letter provider: {provider_name!r}"
        )

    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").strip()
    model = os.getenv("OLLAMA_COVER_LETTER_MODEL", "qwen2.5:7b").strip()
    timeout_raw = os.getenv("OLLAMA_COVER_LETTER_TIMEOUT_SECONDS", "300").strip()
    api_key = os.getenv("OLLAMA_API_KEY", "").strip() or None
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError as exc:
        raise CoverLetterProviderConfigurationError(
            "OLLAMA_COVER_LETTER_TIMEOUT_SECONDS must be numeric"
        ) from exc
    if not base_url or not model or timeout_seconds <= 0:
        raise CoverLetterProviderConfigurationError(
            "Valid Ollama cover-letter configuration is required"
        )
    return OllamaCoverLetterGenerationProvider(
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        api_key=api_key,
    )
