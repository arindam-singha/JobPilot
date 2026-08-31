from __future__ import annotations

import os

from app.llm.evidence_provider import EvidenceExtractionProvider
from app.llm.fake_evidence_provider import FakeEvidenceExtractionProvider
from app.llm.ollama_evidence_provider import (
    OllamaEvidenceExtractionProvider,
)


class EvidenceProviderConfigurationError(Exception):
    """Raised when the configured evidence provider is invalid."""


def create_evidence_provider() -> EvidenceExtractionProvider:
    """Create the evidence provider configured through environment variables."""

    provider_name = os.getenv(
        "EVIDENCE_LLM_PROVIDER",
        "fake",
    ).strip().casefold()

    if provider_name == "fake":
        return FakeEvidenceExtractionProvider()

    if provider_name == "ollama":
        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://host.docker.internal:11434",
        ).strip()

        model = os.getenv(
            "OLLAMA_EVIDENCE_MODEL",
            "",
        ).strip()

        api_key = os.getenv(
            "OLLAMA_API_KEY",
            "",
        ).strip() or None

        timeout_raw = os.getenv(
            "OLLAMA_TIMEOUT_SECONDS",
            "120",
        ).strip()

        if not base_url:
            raise EvidenceProviderConfigurationError(
                "OLLAMA_BASE_URL is required when "
                "EVIDENCE_LLM_PROVIDER=ollama"
            )

        if not model:
            raise EvidenceProviderConfigurationError(
                "OLLAMA_EVIDENCE_MODEL is required when "
                "EVIDENCE_LLM_PROVIDER=ollama"
            )

        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise EvidenceProviderConfigurationError(
                "OLLAMA_TIMEOUT_SECONDS must be numeric"
            ) from exc

        if timeout_seconds <= 0:
            raise EvidenceProviderConfigurationError(
                "OLLAMA_TIMEOUT_SECONDS must be greater than zero"
            )

        return OllamaEvidenceExtractionProvider(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )

    raise EvidenceProviderConfigurationError(
        f"Unsupported evidence provider: {provider_name!r}"
    )