from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.llm.evidence_provider import EvidenceExtractionProviderError
from app.llm.evidence_provider_factory import (
    EvidenceProviderConfigurationError,
    create_evidence_provider,
)
from app.schemas.candidate_evidence import CandidateEvidenceRead
from app.services.evidence_extraction_service import (
    EvidenceExtractionDocumentNotFoundError,
    EvidenceExtractionDocumentNotReadyError,
    EvidenceExtractionEmptyTextError,
)
from app.services.llm_evidence_extraction_service import (
    LlmEvidenceExtractionService,
    LlmEvidenceProviderFailureError,
)


router = APIRouter(
    prefix="/api/v1/candidate-profile",
    tags=["llm-evidence-generation"],
)


@router.post(
    "/{profile_id}/documents/{document_id}/evidence/generate-llm",
    response_model=list[CandidateEvidenceRead],
    status_code=status.HTTP_201_CREATED,
)
async def generate_llm_candidate_evidence(
    profile_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[CandidateEvidenceRead]:
    try:
        provider = create_evidence_provider()
    except EvidenceProviderConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    service = LlmEvidenceExtractionService(
        db,
        provider,
    )

    try:
        evidence_records = await service.extract_document_evidence(
            profile_id,
            document_id,
        )
    except EvidenceExtractionDocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate document not found",
        ) from exc
    except EvidenceExtractionDocumentNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except EvidenceExtractionEmptyTextError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (
        LlmEvidenceProviderFailureError,
        EvidenceExtractionProviderError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Evidence extraction provider failed",
        ) from exc

    return [
        CandidateEvidenceRead.model_validate(record)
        for record in evidence_records
    ]