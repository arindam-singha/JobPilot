from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.models.candidate_document import CandidateDocument
from app.models.candidate_evidence import CandidateEvidence
from app.schemas.candidate_document import CandidateDocumentCreate
from app.schemas.candidate_profile import CandidateProfileCreate
from app.services.candidate_document_service import CandidateDocumentService
from app.services.candidate_profile_service import CandidateProfileService
from app.services.evidence_extraction_service import (
    EvidenceExtractionDocumentNotFoundError,
    EvidenceExtractionDocumentNotReadyError,
    EvidenceExtractionEmptyTextError,
    EvidenceExtractionService,
)


async def _create_profile(
    database_session,
    *,
    full_name: str = "Test Candidate",
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
    extraction_status: str,
    filename: str = "resume.pdf",
    content_type: str = "application/pdf",
):
    document_service = CandidateDocumentService(database_session)

    return await document_service.create_document(
        profile_id,
        CandidateDocumentCreate(
            filename=filename,
            content_type=content_type,
            storage_path=f"/tmp/{filename}",
            file_size=100,
            extracted_text=extracted_text,
            extraction_status=extraction_status,
            extraction_error=None,
        ),
    )


@pytest.mark.asyncio
async def test_build_document_chunks_returns_chunks_for_extracted_document(
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
        Python, PyTorch, ROS2, FastAPI and Docker.

        EXPERIENCE:
        Developed computer vision systems for manufacturing.
        """,
        extraction_status="extracted",
    )

    service = EvidenceExtractionService(
        database_session,
        max_chunk_characters=100,
        chunk_overlap_characters=10,
    )

    chunks = await service.build_document_chunks(
        profile.id,
        document.id,
    )

    assert len(chunks) == 3
    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert [chunk.section for chunk in chunks] == [
        "professional_summary",
        "technical_skills",
        "experience",
    ]


@pytest.mark.asyncio
async def test_build_document_chunks_preserves_source_order(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        SKILLS:
        Python

        EXPERIENCE:
        Lead Engineer

        EDUCATION:
        PhD in Robotics
        """,
        extraction_status="extracted",
    )

    service = EvidenceExtractionService(database_session)

    chunks = await service.build_document_chunks(
        profile.id,
        document.id,
    )

    assert [chunk.section for chunk in chunks] == [
        "skills",
        "experience",
        "education",
    ]

    assert [chunk.text for chunk in chunks] == [
        "Python",
        "Lead Engineer",
        "PhD in Robotics",
    ]


@pytest.mark.asyncio
async def test_build_document_chunks_uses_configured_chunk_size(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    long_text = (
        "Developed computer vision systems for automotive manufacturing. "
        "Built deep learning models for visual defect detection. "
        "Deployed inference services using Docker and Triton."
    )

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text=f"EXPERIENCE:\n{long_text}",
        extraction_status="extracted",
    )

    service = EvidenceExtractionService(
        database_session,
        max_chunk_characters=70,
        chunk_overlap_characters=10,
    )

    chunks = await service.build_document_chunks(
        profile.id,
        document.id,
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 70 for chunk in chunks)
    assert all(chunk.section == "experience" for chunk in chunks)


@pytest.mark.asyncio
async def test_build_document_chunks_is_deterministic(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        SKILLS:
        Python, PyTorch, FastAPI, PostgreSQL and Docker.

        EXPERIENCE:
        Built machine learning applications.
        """,
        extraction_status="extracted",
    )

    service = EvidenceExtractionService(
        database_session,
        max_chunk_characters=50,
        chunk_overlap_characters=10,
    )

    first = await service.build_document_chunks(
        profile.id,
        document.id,
    )

    second = await service.build_document_chunks(
        profile.id,
        document.id,
    )

    assert first == second


@pytest.mark.asyncio
async def test_unknown_document_is_rejected(
    database_session,
) -> None:
    profile = await _create_profile(database_session)
    service = EvidenceExtractionService(database_session)

    with pytest.raises(
        EvidenceExtractionDocumentNotFoundError,
        match="not found",
    ):
        await service.build_document_chunks(
            profile.id,
            uuid4(),
        )


@pytest.mark.asyncio
async def test_cross_profile_document_access_is_rejected(
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
        extraction_status="extracted",
    )

    service = EvidenceExtractionService(database_session)

    with pytest.raises(
        EvidenceExtractionDocumentNotFoundError,
        match="not found",
    ):
        await service.build_document_chunks(
            other.id,
            document.id,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        "uploaded",
        "extracting",
        "failed",
    ],
)
async def test_non_extracted_document_is_rejected(
    database_session,
    status: str,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SKILLS:\nPython",
        extraction_status=status,
    )

    service = EvidenceExtractionService(database_session)

    with pytest.raises(
        EvidenceExtractionDocumentNotReadyError,
        match="not ready",
    ):
        await service.build_document_chunks(
            profile.id,
            document.id,
        )


@pytest.mark.asyncio
async def test_missing_extracted_text_is_rejected(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text=None,
        extraction_status="extracted",
    )

    service = EvidenceExtractionService(database_session)

    with pytest.raises(
        EvidenceExtractionEmptyTextError,
        match="no usable extracted text",
    ):
        await service.build_document_chunks(
            profile.id,
            document.id,
        )


@pytest.mark.asyncio
async def test_whitespace_only_extracted_text_is_rejected(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="   \n\n\t ",
        extraction_status="extracted",
    )

    service = EvidenceExtractionService(database_session)

    with pytest.raises(
        EvidenceExtractionEmptyTextError,
        match="no usable extracted text",
    ):
        await service.build_document_chunks(
            profile.id,
            document.id,
        )


@pytest.mark.asyncio
async def test_service_does_not_create_candidate_evidence(
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
        Built computer vision systems.
        """,
        extraction_status="extracted",
    )

    before_result = await database_session.execute(
        select(func.count(CandidateEvidence.id))
    )
    before_count = before_result.scalar_one()

    service = EvidenceExtractionService(database_session)

    chunks = await service.build_document_chunks(
        profile.id,
        document.id,
    )

    after_result = await database_session.execute(
        select(func.count(CandidateEvidence.id))
    )
    after_count = after_result.scalar_one()

    assert chunks
    assert before_count == after_count


@pytest.mark.asyncio
async def test_service_does_not_modify_candidate_document(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    original_text = """
    SUMMARY:
    Robotics and artificial intelligence engineer.
    """

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text=original_text,
        extraction_status="extracted",
    )

    original_status = document.extraction_status
    original_error = document.extraction_error

    service = EvidenceExtractionService(database_session)

    await service.build_document_chunks(
        profile.id,
        document.id,
    )

    refreshed_document = await database_session.get(
        CandidateDocument,
        document.id,
    )

    assert refreshed_document is not None
    assert refreshed_document.extracted_text == original_text.strip()
    assert refreshed_document.extraction_status == original_status
    assert refreshed_document.extraction_error == original_error


def test_service_rejects_non_positive_maximum() -> None:
    with pytest.raises(
        ValueError,
        match="max_chunk_characters must be greater than zero",
    ):
        EvidenceExtractionService(
            None,  # type: ignore[arg-type]
            max_chunk_characters=0,
            chunk_overlap_characters=0,
        )


def test_service_rejects_negative_overlap() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to zero",
    ):
        EvidenceExtractionService(
            None,  # type: ignore[arg-type]
            max_chunk_characters=100,
            chunk_overlap_characters=-1,
        )


def test_service_rejects_overlap_equal_to_maximum() -> None:
    with pytest.raises(
        ValueError,
        match="must be smaller",
    ):
        EvidenceExtractionService(
            None,  # type: ignore[arg-type]
            max_chunk_characters=100,
            chunk_overlap_characters=100,
        )