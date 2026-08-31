from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.embeddings.embedding_provider_factory import (
    EmbeddingProviderConfigurationError,
    create_embedding_provider,
)
from app.schemas.hybrid_job_candidate_match import (
    HybridJobCandidateMatchRead,
)
from app.services.hybrid_job_candidate_matching_service import (
    HybridEmbeddedEvidenceNotFoundError,
    HybridJobCandidateMatchingService,
)
from app.services.job_candidate_matching_service import (
    MatchingCandidateEvidenceNotFoundError,
    MatchingCandidateProfileNotFoundError,
    MatchingJobNotFoundError,
    MatchingRequirementsNotFoundError,
)
from app.services.semantic_evidence_retrieval_service import (
    SemanticEvidenceProviderFailureError,
)


router = APIRouter(
    prefix="/api/v1/jobs",
    tags=["hybrid-job-candidate-matching"],
)


@router.post(
    "/{job_id}/hybrid-match/{profile_id}",
    response_model=HybridJobCandidateMatchRead,
    status_code=status.HTTP_200_OK,
)
async def hybrid_match_candidate_to_job(
    job_id: UUID,
    profile_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> HybridJobCandidateMatchRead:
    """Match a candidate to a job using deterministic and semantic signals."""

    try:
        embedding_provider = create_embedding_provider()
    except EmbeddingProviderConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    service = HybridJobCandidateMatchingService(
        db,
        embedding_provider,
    )

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
    except HybridEmbeddedEvidenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Compatible candidate evidence embeddings not found",
        ) from exc
    except SemanticEvidenceProviderFailureError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Embedding provider failed",
        ) from exc