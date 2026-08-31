from __future__ import annotations

import os

from app.llm.fake_job_requirements_provider import (
    FakeJobRequirementsProvider,
)
from app.llm.job_requirements_provider import (
    JobRequirementsProvider,
)
from app.llm.ollama_job_requirements_provider import (
    OllamaJobRequirementsProvider,
)


class JobRequirementsProviderConfigurationError(Exception):
    """Raised when job-requirement provider configuration is invalid."""


def create_job_requirements_provider() -> JobRequirementsProvider:
    """Create the configured job-requirement extraction provider."""

    provider_name = os.getenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "fake",
    ).strip().casefold()

    if provider_name == "fake":
        return FakeJobRequirementsProvider()

    if provider_name == "ollama":
        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://ollama:11434",
        ).strip()

        model = os.getenv(
            "OLLAMA_JOB_REQUIREMENTS_MODEL",
            "",
        ).strip()

        api_key = os.getenv(
            "OLLAMA_API_KEY",
            "",
        ).strip() or None

        timeout_raw = os.getenv(
            "OLLAMA_TIMEOUT_SECONDS",
            "300",
        ).strip()

        if not base_url:
            raise JobRequirementsProviderConfigurationError(
                "OLLAMA_BASE_URL is required when "
                "JOB_REQUIREMENTS_LLM_PROVIDER=ollama"
            )

        if not model:
            raise JobRequirementsProviderConfigurationError(
                "OLLAMA_JOB_REQUIREMENTS_MODEL is required when "
                "JOB_REQUIREMENTS_LLM_PROVIDER=ollama"
            )

        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise JobRequirementsProviderConfigurationError(
                "OLLAMA_TIMEOUT_SECONDS must be numeric"
            ) from exc

        if timeout_seconds <= 0:
            raise JobRequirementsProviderConfigurationError(
                "OLLAMA_TIMEOUT_SECONDS must be greater than zero"
            )

        return OllamaJobRequirementsProvider(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )

    raise JobRequirementsProviderConfigurationError(
        f"Unsupported job requirements provider: {provider_name!r}"
    )