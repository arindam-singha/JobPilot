from __future__ import annotations

import pytest

from app.llm.fake_job_requirements_provider import (
    FakeJobRequirementsProvider,
)
from app.llm.job_requirements_provider_factory import (
    JobRequirementsProviderConfigurationError,
    create_job_requirements_provider,
)
from app.llm.ollama_job_requirements_provider import (
    OllamaJobRequirementsProvider,
)


def test_factory_defaults_to_fake_provider(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        raising=False,
    )

    provider = create_job_requirements_provider()

    assert isinstance(
        provider,
        FakeJobRequirementsProvider,
    )


def test_factory_creates_fake_provider(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "fake",
    )

    provider = create_job_requirements_provider()

    assert isinstance(
        provider,
        FakeJobRequirementsProvider,
    )


def test_factory_creates_ollama_provider(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "ollama",
    )
    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://ollama:11434",
    )
    monkeypatch.setenv(
        "OLLAMA_JOB_REQUIREMENTS_MODEL",
        "qwen2.5:7b",
    )
    monkeypatch.setenv(
        "OLLAMA_TIMEOUT_SECONDS",
        "300",
    )

    provider = create_job_requirements_provider()

    assert isinstance(
        provider,
        OllamaJobRequirementsProvider,
    )
    assert provider.provider_name == "ollama"
    assert provider.model_name == "qwen2.5:7b"


def test_factory_requires_ollama_model(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "ollama",
    )
    monkeypatch.delenv(
        "OLLAMA_JOB_REQUIREMENTS_MODEL",
        raising=False,
    )

    with pytest.raises(
        JobRequirementsProviderConfigurationError,
        match="OLLAMA_JOB_REQUIREMENTS_MODEL",
    ):
        create_job_requirements_provider()


def test_factory_rejects_non_numeric_timeout(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "ollama",
    )
    monkeypatch.setenv(
        "OLLAMA_JOB_REQUIREMENTS_MODEL",
        "qwen2.5:7b",
    )
    monkeypatch.setenv(
        "OLLAMA_TIMEOUT_SECONDS",
        "invalid",
    )

    with pytest.raises(
        JobRequirementsProviderConfigurationError,
        match="must be numeric",
    ):
        create_job_requirements_provider()


def test_factory_rejects_non_positive_timeout(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "ollama",
    )
    monkeypatch.setenv(
        "OLLAMA_JOB_REQUIREMENTS_MODEL",
        "qwen2.5:7b",
    )
    monkeypatch.setenv(
        "OLLAMA_TIMEOUT_SECONDS",
        "0",
    )

    with pytest.raises(
        JobRequirementsProviderConfigurationError,
        match="greater than zero",
    ):
        create_job_requirements_provider()


def test_factory_rejects_unknown_provider(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "unknown",
    )

    with pytest.raises(
        JobRequirementsProviderConfigurationError,
        match="Unsupported job requirements provider",
    ):
        create_job_requirements_provider()