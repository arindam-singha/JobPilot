from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.llm.job_requirements_provider_factory import (
    JobRequirementsProviderConfigurationError,
    create_job_requirements_provider,
)
from app.schemas.job_requirements import JobRequirementsRead
from app.services.job_requirements_extraction_service import (
    JobRequirementsExtractionService,
    JobRequirementsJobNotFoundError,
    JobRequirementsProviderFailureError,
)


router = APIRouter(
    prefix="/api/v1/jobs",
    tags=["job-requirements"],
)


@router.post(
    "/{job_id}/requirements/extract",
    response_model=JobRequirementsRead,
    status_code=status.HTTP_201_CREATED,
)
async def extract_job_requirements(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> JobRequirementsRead:
    """Extract and persist structured requirements for a job."""

    try:
        provider = create_job_requirements_provider()
    except JobRequirementsProviderConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    service = JobRequirementsExtractionService(
        db,
        provider,
    )

    try:
        return await service.extract_and_persist(job_id)
    except JobRequirementsJobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        ) from exc
    except JobRequirementsProviderFailureError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Job requirements provider failed",
        ) from exc


@router.get(
    "/{job_id}/requirements",
    response_model=JobRequirementsRead,
)
async def get_job_requirements(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> JobRequirementsRead:
    """Return previously extracted structured requirements."""

    # The provider is not used by GET, but the service requires one.
    # The fake provider keeps this operation free from external calls.
    from app.llm.fake_job_requirements_provider import (
        FakeJobRequirementsProvider,
    )

    service = JobRequirementsExtractionService(
        db,
        FakeJobRequirementsProvider(),
    )

    try:
        return await service.get_requirements(job_id)
    except JobRequirementsJobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc