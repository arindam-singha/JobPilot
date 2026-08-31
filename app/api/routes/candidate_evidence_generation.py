from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.candidate_evidence import CandidateEvidenceRead
from app.services.candidate_evidence_generation_service import (
    CandidateEvidenceGenerationService,
)
from app.services.evidence_extraction_service import (
    EvidenceExtractionDocumentNotFoundError,
    EvidenceExtractionDocumentNotReadyError,
    EvidenceExtractionEmptyTextError,
)


router = APIRouter(
    prefix="/api/v1/candidate-profile",
    tags=["candidate-evidence-generation"],
)


@router.post(
    "/{profile_id}/documents/{document_id}/evidence/generate",
    response_model=list[CandidateEvidenceRead],
    status_code=status.HTTP_201_CREATED,
)
async def generate_candidate_evidence(
    profile_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[CandidateEvidenceRead]:
    """Generate deterministic evidence from an extracted candidate document."""

    service = CandidateEvidenceGenerationService(db)

    try:
        evidence_records = await service.generate_document_evidence(
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

    return [
        CandidateEvidenceRead.model_validate(record)
        for record in evidence_records
    ]