from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.cv.models import TextChunk
from app.llm.fake_evidence_provider import (
    FakeEvidenceExtractionProvider,
)
from app.models.candidate_evidence import CandidateEvidence
from app.schemas.candidate_document import CandidateDocumentCreate
from app.schemas.candidate_profile import CandidateProfileCreate
from app.schemas.llm_evidence import (
    LlmChunkExtractionResult,
    LlmEvidenceItem,
)
from app.services.candidate_document_service import (
    CandidateDocumentService,
)
from app.services.candidate_profile_service import (
    CandidateProfileService,
)
from app.services.evidence_extraction_service import (
    EvidenceExtractionDocumentNotFoundError,
    EvidenceExtractionDocumentNotReadyError,
)
from app.services.llm_evidence_extraction_service import (
    LLM_SOURCE_TYPE,
    LlmEvidenceExtractionService,
    LlmEvidenceProviderFailureError,
)


async def _create_profile(
    database_session,
    *,
    full_name: str = "LLM Test Candidate",
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


@pytest.mark.asyncio
async def test_fake_provider_returns_structured_result() -> None:
    provider = FakeEvidenceExtractionProvider()

    chunk = TextChunk(
        index=0,
        section="skills",
        text="Python and PyTorch",
        start_offset=0,
        end_offset=18,
    )

    result = await provider.extract_evidence(chunk)

    assert len(result.evidence) == 1
    assert result.evidence[0].evidence_type == "skill"
    assert result.evidence[0].confidence == 1.0


@pytest.mark.asyncio
async def test_service_persists_llm_evidence(
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
    )

    service = LlmEvidenceExtractionService(
        database_session,
        FakeEvidenceExtractionProvider(),
    )

    evidence = await service.extract_document_evidence(
        profile.id,
        document.id,
    )

    assert len(evidence) == 2
    assert all(item.source_type == LLM_SOURCE_TYPE for item in evidence)
    assert all(item.source_id == document.id for item in evidence)


@pytest.mark.asyncio
async def test_service_stores_provider_provenance(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="PROJECTS:\nBuilt JobPilot.",
    )

    service = LlmEvidenceExtractionService(
        database_session,
        FakeEvidenceExtractionProvider(),
    )

    evidence = await service.extract_document_evidence(
        profile.id,
        document.id,
    )

    metadata = json.loads(evidence[0].metadata_json)

    assert metadata["provider"] == "fake"
    assert metadata["model"] == "fake-evidence-model"
    assert metadata["generation_method"] == "llm_structured_extraction"
    assert metadata["confidence"] == 1.0
    assert metadata["chunk_index"] == 0
    assert metadata["section"] == "projects"


@pytest.mark.asyncio
async def test_custom_fake_provider_response_is_persisted(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="EXPERIENCE:\nReduced inspection time.",
    )

    def responder(
        chunk: TextChunk,
    ) -> LlmChunkExtractionResult:
        return LlmChunkExtractionResult(
            evidence=[
                LlmEvidenceItem(
                    evidence_type="achievement",
                    title="Inspection time reduction",
                    content="Reduced inspection time significantly.",
                    confidence=0.91,
                    metadata={"metric_found": True},
                )
            ]
        )

    provider = FakeEvidenceExtractionProvider(responder=responder)

    service = LlmEvidenceExtractionService(
        database_session,
        provider,
    )

    evidence = await service.extract_document_evidence(
        profile.id,
        document.id,
    )

    assert len(evidence) == 1
    assert evidence[0].evidence_type == "achievement"
    assert evidence[0].title == "Inspection time reduction"

    metadata = json.loads(evidence[0].metadata_json)

    assert metadata["confidence"] == 0.91
    assert metadata["provider_metadata"] == {
        "metric_found": True,
    }


@pytest.mark.asyncio
async def test_empty_provider_result_creates_no_records(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SUMMARY:\nAI engineer.",
    )

    provider = FakeEvidenceExtractionProvider(
        responder=lambda chunk: LlmChunkExtractionResult(
            evidence=[]
        )
    )

    service = LlmEvidenceExtractionService(
        database_session,
        provider,
    )

    evidence = await service.extract_document_evidence(
        profile.id,
        document.id,
    )

    assert evidence == []

    result = await database_session.execute(
        select(CandidateEvidence).where(
            CandidateEvidence.profile_id == profile.id,
            CandidateEvidence.source_type == LLM_SOURCE_TYPE,
            CandidateEvidence.source_id == document.id,
        )
    )

    assert list(result.scalars().all()) == []


@pytest.mark.asyncio
async def test_llm_generation_is_idempotent(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="""
        SKILLS:
        Python.

        EXPERIENCE:
        Built AI systems.
        """,
    )

    service = LlmEvidenceExtractionService(
        database_session,
        FakeEvidenceExtractionProvider(),
    )

    first = await service.extract_document_evidence(
        profile.id,
        document.id,
    )

    second = await service.extract_document_evidence(
        profile.id,
        document.id,
    )

    result = await database_session.execute(
        select(CandidateEvidence).where(
            CandidateEvidence.profile_id == profile.id,
            CandidateEvidence.source_type == LLM_SOURCE_TYPE,
            CandidateEvidence.source_id == document.id,
        )
    )

    persisted = list(result.scalars().all())

    assert len(first) == 2
    assert len(second) == 2
    assert len(persisted) == 2


@pytest.mark.asyncio
async def test_llm_generation_preserves_manual_evidence(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SKILLS:\nPython.",
    )

    manual = CandidateEvidence(
        profile_id=profile.id,
        evidence_type="achievement",
        title="Manual evidence",
        content="Created manually.",
        source_type="manual",
        source_id=None,
        metadata_json=None,
    )

    database_session.add(manual)
    await database_session.commit()
    await database_session.refresh(manual)

    service = LlmEvidenceExtractionService(
        database_session,
        FakeEvidenceExtractionProvider(),
    )

    await service.extract_document_evidence(
        profile.id,
        document.id,
    )

    retained = await database_session.get(
        CandidateEvidence,
        manual.id,
    )

    assert retained is not None
    assert retained.source_type == "manual"


@pytest.mark.asyncio
async def test_llm_generation_preserves_deterministic_evidence(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SKILLS:\nPython.",
    )

    deterministic = CandidateEvidence(
        profile_id=profile.id,
        evidence_type="skill",
        title="Deterministic evidence",
        content="Python.",
        source_type="candidate_document",
        source_id=document.id,
        metadata_json=None,
    )

    database_session.add(deterministic)
    await database_session.commit()
    await database_session.refresh(deterministic)

    service = LlmEvidenceExtractionService(
        database_session,
        FakeEvidenceExtractionProvider(),
    )

    await service.extract_document_evidence(
        profile.id,
        document.id,
    )

    retained = await database_session.get(
        CandidateEvidence,
        deterministic.id,
    )

    assert retained is not None
    assert retained.source_type == "candidate_document"


@pytest.mark.asyncio
async def test_unknown_document_is_rejected(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    service = LlmEvidenceExtractionService(
        database_session,
        FakeEvidenceExtractionProvider(),
    )

    from uuid import uuid4

    with pytest.raises(
        EvidenceExtractionDocumentNotFoundError,
    ):
        await service.extract_document_evidence(
            profile.id,
            uuid4(),
        )


@pytest.mark.asyncio
async def test_non_extracted_document_is_rejected(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SKILLS:\nPython.",
        extraction_status="failed",
    )

    service = LlmEvidenceExtractionService(
        database_session,
        FakeEvidenceExtractionProvider(),
    )

    with pytest.raises(
        EvidenceExtractionDocumentNotReadyError,
    ):
        await service.extract_document_evidence(
            profile.id,
            document.id,
        )


@pytest.mark.asyncio
async def test_provider_failure_is_wrapped(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    document = await _create_document(
        database_session,
        profile_id=profile.id,
        extracted_text="SKILLS:\nPython.",
    )

    def failing_responder(
        chunk: TextChunk,
    ) -> LlmChunkExtractionResult:
        raise RuntimeError("provider unavailable")

    provider = FakeEvidenceExtractionProvider(
        responder=failing_responder
    )

    service = LlmEvidenceExtractionService(
        database_session,
        provider,
    )

    with pytest.raises(
        LlmEvidenceProviderFailureError,
        match="failed for chunk 0",
    ):
        await service.extract_document_evidence(
            profile.id,
            document.id,
        )