from __future__ import annotations

import json
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.cv.models import TextChunk
from app.llm.evidence_provider import EvidenceExtractionProviderError
from app.models.candidate_evidence import CandidateEvidence
from app.schemas.candidate_document import CandidateDocumentCreate
from app.schemas.candidate_profile import CandidateProfileCreate
from app.services.candidate_document_service import CandidateDocumentService
from app.services.candidate_profile_service import CandidateProfileService
from app.services.llm_evidence_extraction_service import LLM_SOURCE_TYPE


async def _create_profile(
    database_session,
    *,
    full_name: str = "LLM API Test Candidate",
):
    service = CandidateProfileService(database_session)

    return await service.create_profile(
        CandidateProfileCreate(full_name=full_name)
    )


async def _create_document(
    database_session,
    *,
    profile_id,
    extracted_text: str | None,
    extraction_status: str = "extracted",
):
    service = CandidateDocumentService(database_session)

    return await service.create_document(
        profile_id,
        CandidateDocumentCreate(
            filename="resume.pdf",
            content_type="application/pdf",
            storage_path="/tmp/resume.pdf",
            file_size=100,
            extracted_text=extracted_text,
            extraction_status=extraction_status,
            extraction_error=None,
        ),
    )


def _generation_url(profile_id, document_id) -> str:
    return (
        f"/api/v1/candidate-profile/{profile_id}"
        f"/documents/{document_id}/evidence/generate-llm"
    )


@pytest.mark.asyncio
async def test_generate_llm_evidence_returns_201(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_LLM_PROVIDER", "fake")

    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        PROFESSIONAL SUMMARY:
        Robotics and artificial intelligence engineer.

        TECHNICAL SKILLS:
        Python, PyTorch, ROS2 and Docker.

        EXPERIENCE:
        Developed computer vision systems for manufacturing.
        """,
    )

    response = await async_client.post(
        _generation_url(profile.id, document.id)
    )

    assert response.status_code == 201

    body = response.json()

    assert len(body) == 3
    assert [item["evidence_type"] for item in body] == [
        "summary",
        "skill",
        "experience",
    ]

    assert all(
        item["source_type"] == LLM_SOURCE_TYPE
        for item in body
    )

    assert all(
        item["source_id"] == str(document.id)
        for item in body
    )


@pytest.mark.asyncio
async def test_generate_llm_evidence_persists_records(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_LLM_PROVIDER", "fake")

    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        SKILLS:
        Python and PyTorch.

        PROJECTS:
        Built an automotive defect-detection platform.
        """,
    )

    response = await async_client.post(
        _generation_url(profile.id, document.id)
    )

    assert response.status_code == 201

    result = await database_session.execute(
        select(CandidateEvidence).where(
            CandidateEvidence.profile_id == profile.id,
            CandidateEvidence.source_type == LLM_SOURCE_TYPE,
            CandidateEvidence.source_id == document.id,
        )
    )

    persisted = list(result.scalars().all())

    assert len(persisted) == 2
    assert [item.evidence_type for item in persisted] == [
        "skill",
        "project",
    ]


@pytest.mark.asyncio
async def test_generate_llm_evidence_contains_provenance(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_LLM_PROVIDER", "fake")

    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        EXPERIENCE:
        Reduced inspection time from 15 minutes to 5 seconds.
        """,
    )

    response = await async_client.post(
        _generation_url(profile.id, document.id)
    )

    assert response.status_code == 201

    body = response.json()

    assert len(body) == 1

    metadata = json.loads(body[0]["metadata_json"])

    assert metadata["provider"] == "fake"
    assert metadata["model"] == "fake-evidence-model"
    assert metadata["confidence"] == 1.0
    assert metadata["chunk_index"] == 0
    assert metadata["item_index"] == 0
    assert metadata["section"] == "experience"
    assert metadata["start_offset"] >= 0
    assert metadata["end_offset"] > metadata["start_offset"]
    assert metadata["generation_method"] == "llm_structured_extraction"


@pytest.mark.asyncio
async def test_generate_llm_evidence_is_idempotent(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_LLM_PROVIDER", "fake")

    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        SKILLS:
        Python and PyTorch.

        EXPERIENCE:
        Built machine-learning systems.
        """,
    )

    url = _generation_url(profile.id, document.id)

    first_response = await async_client.post(url)
    second_response = await async_client.post(url)

    assert first_response.status_code == 201
    assert second_response.status_code == 201

    assert len(first_response.json()) == 2
    assert len(second_response.json()) == 2

    result = await database_session.execute(
        select(CandidateEvidence).where(
            CandidateEvidence.profile_id == profile.id,
            CandidateEvidence.source_type == LLM_SOURCE_TYPE,
            CandidateEvidence.source_id == document.id,
        )
    )

    persisted = list(result.scalars().all())

    assert len(persisted) == 2


@pytest.mark.asyncio
async def test_generate_llm_evidence_preserves_manual_evidence(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_LLM_PROVIDER", "fake")

    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SKILLS:\nPython.",
    )

    manual_evidence = CandidateEvidence(
        profile_id=profile.id,
        evidence_type="achievement",
        title="Manual achievement",
        content="Received an engineering award.",
        source_type="manual",
        source_id=None,
        metadata_json=None,
    )

    database_session.add(manual_evidence)
    await database_session.commit()
    await database_session.refresh(manual_evidence)

    response = await async_client.post(
        _generation_url(profile.id, document.id)
    )

    assert response.status_code == 201

    retained = await database_session.get(
        CandidateEvidence,
        manual_evidence.id,
    )

    assert retained is not None
    assert retained.source_type == "manual"


@pytest.mark.asyncio
async def test_unknown_document_returns_404(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_LLM_PROVIDER", "fake")

    profile = await _create_profile(database_session)

    response = await async_client.post(
        _generation_url(profile.id, uuid4())
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Candidate document not found"


@pytest.mark.asyncio
async def test_cross_profile_document_access_returns_404(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_LLM_PROVIDER", "fake")

    owner = await _create_profile(
        database_session,
        full_name="Document Owner",
    )

    other = await _create_profile(
        database_session,
        full_name="Other Candidate",
    )

    document = await _create_document(
        database_session,
        profile_id=owner.id,
        extracted_text="SKILLS:\nPython.",
    )

    response = await async_client.post(
        _generation_url(other.id, document.id)
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Candidate document not found"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extraction_status",
    [
        "uploaded",
        "extracting",
        "failed",
    ],
)
async def test_document_not_ready_returns_409(
    async_client,
    database_session,
    monkeypatch,
    extraction_status: str,
) -> None:
    monkeypatch.setenv("EVIDENCE_LLM_PROVIDER", "fake")

    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SKILLS:\nPython.",
        extraction_status=extraction_status,
    )

    response = await async_client.post(
        _generation_url(profile.id, document.id)
    )

    assert response.status_code == 409
    assert "not ready" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_missing_extracted_text_returns_422(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_LLM_PROVIDER", "fake")

    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text=None,
        extraction_status="extracted",
    )

    response = await async_client.post(
        _generation_url(profile.id, document.id)
    )

    assert response.status_code == 422
    assert (
        "no usable extracted text"
        in response.json()["detail"].lower()
    )


@pytest.mark.asyncio
async def test_invalid_uuid_returns_422(
    async_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_LLM_PROVIDER", "fake")

    response = await async_client.post(
        "/api/v1/candidate-profile/not-a-uuid"
        "/documents/not-a-uuid/evidence/generate-llm"
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_provider_failure_returns_502(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SKILLS:\nPython.",
    )

    class FailingProvider:
        @property
        def provider_name(self) -> str:
            return "failing-provider"

        @property
        def model_name(self) -> str:
            return "failing-model"

        async def extract_evidence(self, chunk: TextChunk):
            raise EvidenceExtractionProviderError(
                "Provider unavailable"
            )

    monkeypatch.setattr(
        "app.api.routes.llm_evidence_generation."
        "create_evidence_provider",
        lambda: FailingProvider(),
    )

    response = await async_client.post(
        _generation_url(profile.id, document.id)
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Evidence extraction provider failed"
    )


@pytest.mark.asyncio
async def test_invalid_provider_configuration_returns_503(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    from app.llm.evidence_provider_factory import (
        EvidenceProviderConfigurationError,
    )

    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SKILLS:\nPython.",
    )

    def raise_configuration_error():
        raise EvidenceProviderConfigurationError(
            "Unsupported evidence provider"
        )

    monkeypatch.setattr(
        "app.api.routes.llm_evidence_generation."
        "create_evidence_provider",
        raise_configuration_error,
    )

    response = await async_client.post(
        _generation_url(profile.id, document.id)
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Unsupported evidence provider"
    )