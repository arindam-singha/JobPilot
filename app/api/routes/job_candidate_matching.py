from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.job_candidate_match import JobCandidateMatchRead
from app.services.job_candidate_matching_service import (
    JobCandidateMatchingService,
    MatchingCandidateEvidenceNotFoundError,
    MatchingCandidateProfileNotFoundError,
    MatchingJobNotFoundError,
    MatchingRequirementsNotFoundError,
)


router = APIRouter(
    prefix="/api/v1/jobs",
    tags=["job-candidate-matching"],
)


@router.post(
    "/{job_id}/match/{profile_id}",
    response_model=JobCandidateMatchRead,
    status_code=status.HTTP_200_OK,
)
async def match_candidate_to_job(
    job_id: UUID,
    profile_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> JobCandidateMatchRead:
    """Match one candidate profile against one job."""

    service = JobCandidateMatchingService(db)

    try:
        return await service.match(
            job_id=job_id,
            profile_id=profile_id,
        )
    except MatchingJobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        ) from exc
    except MatchingRequirementsNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Structured job requirements not found",
        ) from exc
    except MatchingCandidateProfileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found",
        ) from exc
    except MatchingCandidateEvidenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate evidence not found",
        ) from exc