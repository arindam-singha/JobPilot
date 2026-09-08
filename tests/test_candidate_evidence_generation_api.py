from __future__ import annotations

from uuid import uuid4

import pytest
from app.models.candidate_evidence import CandidateEvidence
from app.schemas.candidate_document import CandidateDocumentCreate
from app.schemas.candidate_profile import CandidateProfileCreate
from app.services.candidate_document_service import CandidateDocumentService
from app.services.candidate_profile_service import CandidateProfileService
from sqlalchemy import select


async def _create_profile(
    database_session,
    *,
    full_name: str = "API Test Candidate",
):
    profile_service = CandidateProfileService(database_session)

    return await profile_service.create_profile(
        CandidateProfileCreate(
            full_name=full_name,
        )
    )


async def _create_document(
    database_session,
    *,
    profile_id,
    extracted_text: str | None,
    extraction_status: str = "extracted",
):
    document_service = CandidateDocumentService(database_session)

    return await document_service.create_document(
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


@pytest.mark.asyncio
async def test_generate_evidence_endpoint_returns_201(
    async_client,
    database_session,
) -> None:
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
        f"/api/v1/candidate-profile/{profile.id}"
        f"/documents/{document.id}/evidence/generate"
    )

    assert response.status_code == 201

    body = response.json()

    assert len(body) == 3
    assert [item["evidence_type"] for item in body] == [
        "summary",
        "skill",
        "experience",
    ]


@pytest.mark.asyncio
async def test_generated_evidence_response_contains_provenance(
    async_client,
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        PROJECTS:
        Developed an automotive defect detection platform.
        """,
    )

    response = await async_client.post(
        f"/api/v1/candidate-profile/{profile.id}"
        f"/documents/{document.id}/evidence/generate"
    )

    assert response.status_code == 201

    body = response.json()

    assert len(body) == 1
    assert body[0]["source_type"] == "candidate_document"
    assert body[0]["source_id"] == str(document.id)
    assert body[0]["evidence_type"] == "project"


@pytest.mark.asyncio
async def test_generation_endpoint_persists_evidence(
    async_client,
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        SKILLS:
        Python and PyTorch.
        """,
    )

    response = await async_client.post(
        f"/api/v1/candidate-profile/{profile.id}"
        f"/documents/{document.id}/evidence/generate"
    )

    assert response.status_code == 201

    result = await database_session.execute(
        select(CandidateEvidence).where(
            CandidateEvidence.profile_id == profile.id,
            CandidateEvidence.source_id == document.id,
        )
    )

    persisted = list(result.scalars().all())

    assert len(persisted) == 1
    assert persisted[0].evidence_type == "skill"


@pytest.mark.asyncio
async def test_generation_endpoint_is_idempotent(
    async_client,
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        SKILLS:
        Python and PyTorch.

        EXPERIENCE:
        Built machine learning systems.
        """,
    )

    url = (
        f"/api/v1/candidate-profile/{profile.id}"
        f"/documents/{document.id}/evidence/generate"
    )

    first_response = await async_client.post(url)
    second_response = await async_client.post(url)

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    assert len(first_response.json()) == 2
    assert len(second_response.json()) == 2

    result = await database_session.execute(
        select(CandidateEvidence).where(
            CandidateEvidence.profile_id == profile.id,
            CandidateEvidence.source_id == document.id,
        )
    )

    persisted = list(result.scalars().all())

    assert len(persisted) == 2


@pytest.mark.asyncio
async def test_unknown_document_returns_404(
    async_client,
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    response = await async_client.post(
        f"/api/v1/candidate-profile/{profile.id}"
        f"/documents/{uuid4()}/evidence/generate"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Candidate document not found"


@pytest.mark.asyncio
async def test_cross_profile_document_access_returns_404(
    async_client,
    database_session,
) -> None:
    owner = await _create_profile(
        database_session,
        full_name="Owner",
    )
    other = await _create_profile(
        database_session,
        full_name="Other",
    )

    document = await _create_document(
        database_session,
        profile_id=owner.id,
        extracted_text="SKILLS:\nPython",
    )

    response = await async_client.post(
        f"/api/v1/candidate-profile/{other.id}"
        f"/documents/{document.id}/evidence/generate"
    )

    assert response.status_code == 404


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
    extraction_status: str,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SKILLS:\nPython",
        extraction_status=extraction_status,
    )

    response = await async_client.post(
        f"/api/v1/candidate-profile/{profile.id}"
        f"/documents/{document.id}/evidence/generate"
    )

    assert response.status_code == 409
    assert "not ready" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_missing_extracted_text_returns_422(
    async_client,
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text=None,
        extraction_status="extracted",
    )

    response = await async_client.post(
        f"/api/v1/candidate-profile/{profile.id}"
        f"/documents/{document.id}/evidence/generate"
    )

    assert response.status_code == 422
    assert "no usable extracted text" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_invalid_uuid_returns_422(
    async_client,
) -> None:
    response = await async_client.post(
        "/api/v1/candidate-profile/not-a-uuid"
        "/documents/not-a-uuid/evidence/generate"
    )

    assert response.status_code == 422