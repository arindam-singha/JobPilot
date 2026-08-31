from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest
from app.llm.ollama_resume_provider import OllamaResumeGenerationProvider
from app.llm.resume_provider import ResumeGenerationProviderError
from app.schemas.tailored_resume import (
    ResumeGenerationContext,
    ResumeGroundingBundle,
    ResumeHeader,
    ResumeRequirementTrace,
    SelectedResumeEvidence,
)


@pytest.fixture(autouse=True)
def clean_database() -> None:
    """Override the suite-wide database fixture for pure provider tests."""


def _context() -> ResumeGenerationContext:
    profile_id = uuid4()
    evidence = SelectedResumeEvidence(
        evidence_id=uuid4(),
        profile_id=profile_id,
        evidence_type="project",
        title="Defect detection",
        content="Reduced visual inspection time from 10 minutes to 5 seconds.",
        source_type="candidate_project",
        source_id=uuid4(),
        selection_score=0.94,
        requirement_traces=[
            ResumeRequirementTrace(
                category="required_experience",
                requirement="Visual inspection automation",
                deterministic_score=1.0,
                semantic_score=0.85,
                hybrid_score=0.94,
                rank=1,
            )
        ],
    )
    return ResumeGenerationContext(
        job_title="Senior Computer Vision Engineer",
        company="Example Manufacturing",
        location="Abu Dhabi",
        job_description="Automate manufacturing visual inspection.",
        header=ResumeHeader(full_name="Candidate"),
        grounding=ResumeGroundingBundle(
            job_id=uuid4(),
            profile_id=profile_id,
            hybrid_overall_score=92,
            matched_requirements=["Visual inspection automation"],
            missing_requirements=["Kubernetes"],
            selected_evidence=[evidence],
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            minimum_evidence_score=0.6,
            max_evidence_per_requirement=3,
            max_total_evidence=12,
        ),
    )


def _content(context: ResumeGenerationContext, *, evidence_id=None) -> dict:
    return {
        "target_title": context.job_title,
        "professional_summary": [
            {
                "text": "Computer vision engineer for manufacturing inspection.",
                "evidence_ids": [
                    str(evidence_id or context.grounding.selected_evidence[0].evidence_id)
                ],
            }
        ],
        "sections": [],
        "skills": [],
    }


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_ollama_provider_returns_grounded_structured_content() -> None:
    context = _context()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps(_content(context))}},
        )

    async with _client(handler) as client:
        provider = OllamaResumeGenerationProvider(
            base_url="http://ollama:11434",
            model="qwen2.5:7b",
            client=client,
        )
        result = await provider.generate_resume(context)

    assert result.target_title == context.job_title
    assert result.professional_summary[0].evidence_ids == [
        context.grounding.selected_evidence[0].evidence_id
    ]
    assert provider.provider_name == "ollama"
    assert provider.model_name == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_ollama_request_contains_schema_job_and_evidence() -> None:
    context = _context()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps(_content(context))}},
        )

    async with _client(handler) as client:
        await OllamaResumeGenerationProvider(
            base_url="http://ollama:11434",
            client=client,
        ).generate_resume(context)

    assert captured["model"] == "qwen2.5:7b"
    assert captured["stream"] is False
    assert isinstance(captured["format"], dict)
    assert captured["options"] == {"temperature": 0}
    messages = captured["messages"]
    assert isinstance(messages, list)
    prompt = messages[1]["content"]
    assert context.job_title in prompt
    assert context.job_description in prompt
    assert str(context.grounding.selected_evidence[0].evidence_id) in prompt
    assert context.grounding.selected_evidence[0].content in prompt
    assert "Kubernetes" in prompt


@pytest.mark.asyncio
async def test_ollama_provider_rejects_unknown_evidence_reference() -> None:
    context = _context()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps(_content(context, evidence_id=uuid4()))}},
        )

    async with _client(handler) as client:
        provider = OllamaResumeGenerationProvider(
            base_url="http://ollama:11434",
            client=client,
        )
        with pytest.raises(ResumeGenerationProviderError, match="outside"):
            await provider.generate_resume(context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500, json={"error": "failure"}), "HTTP 500"),
        (httpx.Response(200, json={"error": "model missing"}), "model missing"),
        (httpx.Response(200, json={}), "does not contain a message"),
        (
            httpx.Response(200, json={"message": {"content": " "}}),
            "structured resume content",
        ),
        (
            httpx.Response(200, json={"message": {"content": "not-json"}}),
            "does not match the resume schema",
        ),
    ],
)
async def test_ollama_provider_rejects_invalid_responses(response, message) -> None:
    async with _client(lambda request: response) as client:
        provider = OllamaResumeGenerationProvider(
            base_url="http://ollama:11434",
            client=client,
        )
        with pytest.raises(ResumeGenerationProviderError, match=message):
            await provider.generate_resume(_context())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_url": " ", "model": "qwen2.5:7b"}, "base URL"),
        ({"base_url": "http://ollama", "model": " "}, "model"),
        (
            {"base_url": "http://ollama", "model": "qwen2.5:7b", "timeout_seconds": 0},
            "timeout",
        ),
    ],
)
def test_ollama_provider_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        OllamaResumeGenerationProvider(**kwargs)


@pytest.mark.asyncio
async def test_ollama_provider_sends_api_key() -> None:
    context = _context()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps(_content(context))}},
        )

    async with _client(handler) as client:
        await OllamaResumeGenerationProvider(
            base_url="http://ollama",
            api_key="secret",
            client=client,
        ).generate_resume(context)
