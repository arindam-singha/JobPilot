from __future__ import annotations

import os

from app.llm.fake_resume_provider import FakeResumeGenerationProvider
from app.llm.ollama_resume_provider import OllamaResumeGenerationProvider
from app.llm.resume_provider import ResumeGenerationProvider


class ResumeProviderConfigurationError(Exception):
    """Raised when resume-generation provider configuration is invalid."""


def create_resume_generation_provider() -> ResumeGenerationProvider:
    """Create the configured structured resume-generation provider."""

    provider_name = os.getenv("RESUME_LLM_PROVIDER", "fake").strip().casefold()
    if provider_name == "fake":
        return FakeResumeGenerationProvider()

    if provider_name == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").strip()
        model = os.getenv("OLLAMA_RESUME_MODEL", "qwen2.5:7b").strip()
        timeout_raw = os.getenv("OLLAMA_RESUME_TIMEOUT_SECONDS", "300").strip()
        api_key = os.getenv("OLLAMA_API_KEY", "").strip() or None

        if not base_url:
            raise ResumeProviderConfigurationError(
                "OLLAMA_BASE_URL is required when RESUME_LLM_PROVIDER=ollama"
            )
        if not model:
            raise ResumeProviderConfigurationError(
                "OLLAMA_RESUME_MODEL is required when RESUME_LLM_PROVIDER=ollama"
            )
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise ResumeProviderConfigurationError(
                "OLLAMA_RESUME_TIMEOUT_SECONDS must be numeric"
            ) from exc
        if timeout_seconds <= 0:
            raise ResumeProviderConfigurationError(
                "OLLAMA_RESUME_TIMEOUT_SECONDS must be greater than zero"
            )

        return OllamaResumeGenerationProvider(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )

    raise ResumeProviderConfigurationError(
        f"Unsupported resume generation provider: {provider_name!r}"
    )
