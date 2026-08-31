from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.models.candidate_evidence import CandidateEvidence
from app.schemas.candidate_document import CandidateDocumentCreate
from app.schemas.candidate_profile import CandidateProfileCreate
from app.services.candidate_document_service import CandidateDocumentService
from app.services.candidate_evidence_generation_service import (
    CandidateEvidenceGenerationService,
)
from app.services.candidate_profile_service import CandidateProfileService
from app.services.evidence_extraction_service import (
    EvidenceExtractionDocumentNotFoundError,
    EvidenceExtractionDocumentNotReadyError,
)


async def _create_profile(database_session, name: str = "Test Candidate"):
    service = CandidateProfileService(database_session)

    return await service.create_profile(
        CandidateProfileCreate(full_name=name)
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


@pytest.mark.asyncio
async def test_generate_document_evidence_persists_records(
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

    service = CandidateEvidenceGenerationService(database_session)

    evidence = await service.generate_document_evidence(
        profile.id,
        document.id,
    )

    assert len(evidence) == 3

    result = await database_session.execute(
        select(CandidateEvidence).where(
            CandidateEvidence.profile_id == profile.id,
            CandidateEvidence.source_id == document.id,
        )
    )

    persisted = list(result.scalars().all())

    assert len(persisted) == 3


@pytest.mark.asyncio
async def test_generated_evidence_has_correct_types(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        SUMMARY:
        AI engineer.

        SKILLS:
        Python and PyTorch.

        PROJECTS:
        Built a defect detection platform.

        EDUCATION:
        PhD in Robotics.
        """,
    )

    service = CandidateEvidenceGenerationService(database_session)

    evidence = await service.generate_document_evidence(
        profile.id,
        document.id,
    )

    assert [item.evidence_type for item in evidence] == [
        "summary",
        "skill",
        "project",
        "education",
    ]


@pytest.mark.asyncio
async def test_generated_evidence_contains_document_provenance(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        EXPERIENCE:
        Lead Engineer at Example Company.
        """,
    )

    service = CandidateEvidenceGenerationService(database_session)

    evidence = await service.generate_document_evidence(
        profile.id,
        document.id,
    )

    assert len(evidence) == 1

    item = evidence[0]

    assert item.source_type == "candidate_document"
    assert item.source_id == document.id

    metadata = json.loads(item.metadata_json)

    assert metadata["chunk_index"] == 0
    assert metadata["section"] == "experience"
    assert metadata["start_offset"] >= 0
    assert metadata["end_offset"] > metadata["start_offset"]
    assert metadata["generation_method"] == (
        "deterministic_section_chunking"
    )


@pytest.mark.asyncio
async def test_generation_is_idempotent(
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

    service = CandidateEvidenceGenerationService(database_session)

    first = await service.generate_document_evidence(
        profile.id,
        document.id,
    )

    second = await service.generate_document_evidence(
        profile.id,
        document.id,
    )

    result = await database_session.execute(
        select(CandidateEvidence).where(
            CandidateEvidence.profile_id == profile.id,
            CandidateEvidence.source_type == "candidate_document",
            CandidateEvidence.source_id == document.id,
        )
    )

    persisted = list(result.scalars().all())

    assert len(first) == 2
    assert len(second) == 2
    assert len(persisted) == 2


@pytest.mark.asyncio
async def test_regeneration_replaces_old_document_evidence(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        SKILLS:
        Python.
        """,
    )

    service = CandidateEvidenceGenerationService(database_session)

    first = await service.generate_document_evidence(
        profile.id,
        document.id,
    )

    document.extracted_text = """
    SKILLS:
    Python and PyTorch.

    EXPERIENCE:
    Built computer vision systems.
    """
    await database_session.commit()

    second = await service.generate_document_evidence(
        profile.id,
        document.id,
    )

    result = await database_session.execute(
        select(CandidateEvidence).where(
            CandidateEvidence.profile_id == profile.id,
            CandidateEvidence.source_id == document.id,
        )
    )

    persisted = list(result.scalars().all())

    assert len(first) == 1
    assert len(second) == 2
    assert len(persisted) == 2
    assert all(item.id not in {old.id for old in first} for item in persisted)


@pytest.mark.asyncio
async def test_manual_evidence_is_not_deleted_during_generation(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        SKILLS:
        Python.
        """,
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

    service = CandidateEvidenceGenerationService(database_session)

    await service.generate_document_evidence(
        profile.id,
        document.id,
    )

    retained = await database_session.get(
        CandidateEvidence,
        manual_evidence.id,
    )

    assert retained is not None
    assert retained.source_type == "manual"


@pytest.mark.asyncio
async def test_generation_does_not_delete_other_document_evidence(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    first_document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SKILLS:\nPython",
    )

    second_document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="EDUCATION:\nPhD in Robotics",
    )

    service = CandidateEvidenceGenerationService(database_session)

    await service.generate_document_evidence(
        profile.id,
        first_document.id,
    )

    await service.generate_document_evidence(
        profile.id,
        second_document.id,
    )

    result = await database_session.execute(
        select(CandidateEvidence).where(
            CandidateEvidence.profile_id == profile.id,
        )
    )

    persisted = list(result.scalars().all())

    assert len(persisted) == 2
    assert {item.source_id for item in persisted} == {
        first_document.id,
        second_document.id,
    }


@pytest.mark.asyncio
async def test_cross_profile_generation_is_rejected(
    database_session,
) -> None:
    owner = await _create_profile(database_session, "Owner")
    other = await _create_profile(database_session, "Other")

    document = await _create_document(
        database_session,
        profile_id=owner.id,
        extracted_text="SKILLS:\nPython",
    )

    service = CandidateEvidenceGenerationService(database_session)

    with pytest.raises(EvidenceExtractionDocumentNotFoundError):
        await service.generate_document_evidence(
            other.id,
            document.id,
        )


@pytest.mark.asyncio
async def test_non_extracted_document_is_rejected(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SKILLS:\nPython",
        extraction_status="failed",
    )

    service = CandidateEvidenceGenerationService(database_session)

    with pytest.raises(EvidenceExtractionDocumentNotReadyError):
        await service.generate_document_evidence(
            profile.id,
            document.id,
        )


@pytest.mark.asyncio
async def test_unknown_section_maps_to_general_evidence(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="Candidate name and contact details.",
    )

    service = CandidateEvidenceGenerationService(database_session)

    evidence = await service.generate_document_evidence(
        profile.id,
        document.id,
    )

    assert len(evidence) == 1
    assert evidence[0].evidence_type == "general"
    assert evidence[0].title == "General candidate information — chunk 1"