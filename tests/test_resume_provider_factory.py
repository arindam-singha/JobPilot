from __future__ import annotations

import pytest
from app.llm.fake_resume_provider import FakeResumeGenerationProvider
from app.llm.ollama_resume_provider import OllamaResumeGenerationProvider
from app.llm.resume_provider_factory import (
    ResumeProviderConfigurationError,
    create_resume_generation_provider,
)


@pytest.fixture(autouse=True)
def clean_database() -> None:
    """Override the suite-wide database fixture for pure factory tests."""


def test_factory_defaults_to_fake(monkeypatch) -> None:
    monkeypatch.delenv("RESUME_LLM_PROVIDER", raising=False)
    assert isinstance(create_resume_generation_provider(), FakeResumeGenerationProvider)


def test_factory_creates_ollama_with_qwen_default(monkeypatch) -> None:
    monkeypatch.setenv("RESUME_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_RESUME_MODEL", raising=False)

    provider = create_resume_generation_provider()

    assert isinstance(provider, OllamaResumeGenerationProvider)
    assert provider.model_name == "qwen2.5:7b"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            {
                "RESUME_LLM_PROVIDER": "ollama",
                "OLLAMA_RESUME_MODEL": " ",
            },
            "OLLAMA_RESUME_MODEL",
        ),
        (
            {
                "RESUME_LLM_PROVIDER": "ollama",
                "OLLAMA_RESUME_TIMEOUT_SECONDS": "invalid",
            },
            "must be numeric",
        ),
        (
            {
                "RESUME_LLM_PROVIDER": "ollama",
                "OLLAMA_RESUME_TIMEOUT_SECONDS": "0",
            },
            "greater than zero",
        ),
        ({"RESUME_LLM_PROVIDER": "unknown"}, "Unsupported"),
    ],
)
def test_factory_rejects_invalid_configuration(monkeypatch, environment, message) -> None:
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ResumeProviderConfigurationError, match=message):
        create_resume_generation_provider()
