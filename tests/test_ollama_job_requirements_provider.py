from __future__ import annotations

import json

import httpx
import pytest

from app.llm.job_requirements_provider import (
    JobRequirementsProviderError,
)
from app.llm.ollama_job_requirements_provider import (
    OllamaJobRequirementsProvider,
)


def _mock_transport(
    handler,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )


@pytest.mark.asyncio
async def test_ollama_provider_returns_structured_requirements() -> None:
    structured_result = {
        "required_skills": [
            "Python",
            "ROS2",
        ],
        "preferred_skills": [
            "Isaac Sim",
        ],
        "required_experience": [
            "Robotics engineering",
        ],
        "responsibilities": [
            "Develop autonomous robotic systems",
        ],
        "education_requirements": [
            "Master's degree in Robotics",
        ],
        "certifications": [],
        "domain_keywords": [
            "robotics",
            "autonomous systems",
        ],
        "minimum_experience_years": 5,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(structured_result),
                }
            },
        )

    async with _mock_transport(handler) as client:
        provider = OllamaJobRequirementsProvider(
            base_url="http://ollama:11434",
            model="qwen2.5:7b",
            client=client,
        )

        result = await provider.extract_requirements(
            title="Senior Robotics Engineer",
            company="Example Robotics",
            location="Abu Dhabi",
            description=(
                "Requires 5 years of robotics experience, "
                "Python and ROS2. Isaac Sim is preferred."
            ),
        )

    assert result.required_skills == [
        "Python",
        "ROS2",
    ]
    assert result.preferred_skills == [
        "Isaac Sim",
    ]
    assert result.minimum_experience_years == 5
    assert provider.provider_name == "ollama"
    assert provider.model_name == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_ollama_provider_sends_json_schema() -> None:
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(
            json.loads(request.content.decode())
        )

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "required_skills": [],
                            "preferred_skills": [],
                            "required_experience": [],
                            "responsibilities": [],
                            "education_requirements": [],
                            "certifications": [],
                            "domain_keywords": [],
                            "minimum_experience_years": None,
                        }
                    ),
                }
            },
        )

    async with _mock_transport(handler) as client:
        provider = OllamaJobRequirementsProvider(
            base_url="http://ollama:11434",
            model="qwen2.5:7b",
            client=client,
        )

        await provider.extract_requirements(
            title="Engineer",
            company="Example",
            location=None,
            description="Perform engineering work.",
        )

    assert captured_payload["model"] == "qwen2.5:7b"
    assert captured_payload["stream"] is False
    assert isinstance(captured_payload["format"], dict)

    options = captured_payload["options"]
    assert isinstance(options, dict)
    assert options["temperature"] == 0


@pytest.mark.asyncio
async def test_ollama_provider_prompt_contains_job_data() -> None:
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(
            json.loads(request.content.decode())
        )

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "required_skills": [],
                            "preferred_skills": [],
                            "required_experience": [],
                            "responsibilities": [],
                            "education_requirements": [],
                            "certifications": [],
                            "domain_keywords": [],
                            "minimum_experience_years": None,
                        }
                    ),
                }
            },
        )

    async with _mock_transport(handler) as client:
        provider = OllamaJobRequirementsProvider(
            base_url="http://ollama:11434",
            model="qwen2.5:7b",
            client=client,
        )

        await provider.extract_requirements(
            title="Senior AI Engineer",
            company="Example AI",
            location="Remote",
            description="Build production AI systems with Python.",
        )

    messages = captured_payload["messages"]

    assert isinstance(messages, list)
    assert len(messages) == 2

    user_content = messages[1]["content"]

    assert "Senior AI Engineer" in user_content
    assert "Example AI" in user_content
    assert "Remote" in user_content
    assert "Build production AI systems" in user_content


@pytest.mark.asyncio
async def test_ollama_provider_rejects_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "error": "model failure",
            },
        )

    async with _mock_transport(handler) as client:
        provider = OllamaJobRequirementsProvider(
            base_url="http://ollama:11434",
            model="qwen2.5:7b",
            client=client,
        )

        with pytest.raises(
            JobRequirementsProviderError,
            match="HTTP 500",
        ):
            await provider.extract_requirements(
                title="Engineer",
                company="Example",
                location=None,
                description="Example description",
            )


@pytest.mark.asyncio
async def test_ollama_provider_rejects_api_error_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "error": "model not found",
            },
        )

    async with _mock_transport(handler) as client:
        provider = OllamaJobRequirementsProvider(
            base_url="http://ollama:11434",
            model="missing-model",
            client=client,
        )

        with pytest.raises(
            JobRequirementsProviderError,
            match="model not found",
        ):
            await provider.extract_requirements(
                title="Engineer",
                company="Example",
                location=None,
                description="Example description",
            )


@pytest.mark.asyncio
async def test_ollama_provider_rejects_missing_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={},
        )

    async with _mock_transport(handler) as client:
        provider = OllamaJobRequirementsProvider(
            base_url="http://ollama:11434",
            model="qwen2.5:7b",
            client=client,
        )

        with pytest.raises(
            JobRequirementsProviderError,
            match="does not contain a message",
        ):
            await provider.extract_requirements(
                title="Engineer",
                company="Example",
                location=None,
                description="Example description",
            )


@pytest.mark.asyncio
async def test_ollama_provider_rejects_empty_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "   ",
                }
            },
        )

    async with _mock_transport(handler) as client:
        provider = OllamaJobRequirementsProvider(
            base_url="http://ollama:11434",
            model="qwen2.5:7b",
            client=client,
        )

        with pytest.raises(
            JobRequirementsProviderError,
            match="structured content",
        ):
            await provider.extract_requirements(
                title="Engineer",
                company="Example",
                location=None,
                description="Example description",
            )


@pytest.mark.asyncio
async def test_ollama_provider_rejects_invalid_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "required_skills": "Python",
                        }
                    ),
                }
            },
        )

    async with _mock_transport(handler) as client:
        provider = OllamaJobRequirementsProvider(
            base_url="http://ollama:11434",
            model="qwen2.5:7b",
            client=client,
        )

        with pytest.raises(
            JobRequirementsProviderError,
            match="does not match",
        ):
            await provider.extract_requirements(
                title="Engineer",
                company="Example",
                location=None,
                description="Example description",
            )


def test_ollama_provider_rejects_empty_base_url() -> None:
    with pytest.raises(
        ValueError,
        match="base URL must not be empty",
    ):
        OllamaJobRequirementsProvider(
            base_url=" ",
            model="qwen2.5:7b",
        )


def test_ollama_provider_rejects_empty_model() -> None:
    with pytest.raises(
        ValueError,
        match="model must not be empty",
    ):
        OllamaJobRequirementsProvider(
            base_url="http://ollama:11434",
            model=" ",
        )


def test_ollama_provider_rejects_invalid_timeout() -> None:
    with pytest.raises(
        ValueError,
        match="timeout must be greater than zero",
    ):
        OllamaJobRequirementsProvider(
            base_url="http://ollama:11434",
            model="qwen2.5:7b",
            timeout_seconds=0,
        )