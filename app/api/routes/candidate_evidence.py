from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.candidate_evidence import (
    CandidateEvidenceCreate,
    CandidateEvidenceRead,
    CandidateEvidenceUpdate,
)
from app.services.candidate_evidence_service import (
    CandidateEvidenceNotFoundError,
    CandidateEvidenceService,
)
from app.services.candidate_profile_service import CandidateProfileNotFoundError

router = APIRouter(prefix="/api/v1/candidate-profile", tags=["candidate-evidence"])


@router.post(
    "/{profile_id}/evidence",
    response_model=CandidateEvidenceRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_evidence(
    profile_id: UUID,
    payload: CandidateEvidenceCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateEvidenceRead:
    service = CandidateEvidenceService(db)
    try:
        evidence = await service.create_evidence(profile_id, payload)
    except CandidateProfileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found",
        ) from exc

    return CandidateEvidenceRead.model_validate(evidence)


@router.get(
    "/{profile_id}/evidence",
    response_model=list[CandidateEvidenceRead],
)
async def list_evidence(
    profile_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[CandidateEvidenceRead]:
    service = CandidateEvidenceService(db)
    try:
        evidence_items = await service.list_evidence(profile_id)
    except CandidateProfileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found",
        ) from exc

    return [CandidateEvidenceRead.model_validate(item) for item in evidence_items]


@router.get(
    "/{profile_id}/evidence/{evidence_id}",
    response_model=CandidateEvidenceRead,
)
async def get_evidence(
    profile_id: UUID,
    evidence_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> CandidateEvidenceRead:
    service = CandidateEvidenceService(db)
    try:
        evidence = await service.get_evidence(profile_id, evidence_id)
    except CandidateEvidenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate evidence not found",
        ) from exc

    return CandidateEvidenceRead.model_validate(evidence)


@router.patch(
    "/{profile_id}/evidence/{evidence_id}",
    response_model=CandidateEvidenceRead,
)
async def update_evidence(
    profile_id: UUID,
    evidence_id: UUID,
    payload: CandidateEvidenceUpdate,
    db: AsyncSession = Depends(get_db),
) -> CandidateEvidenceRead:
    service = CandidateEvidenceService(db)
    try:
        evidence = await service.update_evidence(profile_id, evidence_id, payload)
    except CandidateEvidenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate evidence not found",
        ) from exc

    return CandidateEvidenceRead.model_validate(evidence)


@router.delete(
    "/{profile_id}/evidence/{evidence_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_evidence(
    profile_id: UUID,
    evidence_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    service = CandidateEvidenceService(db)
    try:
        await service.delete_evidence(profile_id, evidence_id)
    except CandidateEvidenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate evidence not found",
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
