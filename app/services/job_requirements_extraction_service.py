from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

# from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.job_requirements_provider import (
    JobRequirementsProvider,
    JobRequirementsProviderError,
)
from app.models.job import Job
from app.models.job_requirements import JobRequirements
from app.schemas.job_requirements import JobRequirementsRead


class JobRequirementsExtractionError(Exception):
    """Base exception for job-requirements extraction failures."""


class JobRequirementsJobNotFoundError(JobRequirementsExtractionError):
    """Raised when the supplied job does not exist."""


class JobRequirementsProviderFailureError(JobRequirementsExtractionError):
    """Raised when the configured provider fails."""


class JobRequirementsExtractionService:
    """Extract and persist structured requirements for a job."""

    def __init__(
        self,
        session: AsyncSession,
        provider: JobRequirementsProvider,
    ) -> None:
        self.session = session
        self.provider = provider

    async def extract_and_persist(
        self,
        job_id: UUID,
    ) -> JobRequirementsRead:
        job = await self._get_job(job_id)

        try:
            extracted = await self.provider.extract_requirements(
                title=job.title,
                company=job.company,
                location=job.location,
                description=job.description,
            )
        except JobRequirementsProviderError as exc:
            raise JobRequirementsProviderFailureError(
                f"Job requirements provider "
                f"{self.provider.provider_name!r} failed for job {job_id}"
            ) from exc
        except Exception as exc:
            raise JobRequirementsProviderFailureError(
                f"Job requirements provider "
                f"{self.provider.provider_name!r} failed for job {job_id}"
            ) from exc

        requirements = await self._get_existing_requirements(job_id)

        payload = {
            **extracted.model_dump(),
            "provider": self.provider.provider_name,
            "model_name": self.provider.model_name,
            "extraction_metadata": {
                "generation_method": "llm_structured_extraction",
                "provider": self.provider.provider_name,
                "model": self.provider.model_name,
            },
        }

        try:
            if requirements is None:
                requirements = JobRequirements(
                    job_id=job_id,
                    **payload,
                )
                self.session.add(requirements)
            else:
                for field_name, value in payload.items():
                    setattr(requirements, field_name, value)

                requirements.updated_at = datetime.now(UTC)

            await self.session.commit()
            await self.session.refresh(requirements)

        except Exception:
            await self.session.rollback()
            raise

        return JobRequirementsRead.model_validate(requirements)

    async def get_requirements(
        self,
        job_id: UUID,
    ) -> JobRequirementsRead:
        await self._get_job(job_id)

        requirements = await self._get_existing_requirements(job_id)

        if requirements is None:
            raise JobRequirementsJobNotFoundError(
                f"Structured requirements not found for job {job_id}"
            )

        return JobRequirementsRead.model_validate(requirements)

    async def _get_job(
        self,
        job_id: UUID,
    ) -> Job:
        result = await self.session.execute(
            select(Job).where(Job.id == job_id)
        )

        job = result.scalar_one_or_none()

        if job is None:
            raise JobRequirementsJobNotFoundError(
                f"Job {job_id} not found"
            )

        return job

    async def _get_existing_requirements(
        self,
        job_id: UUID,
    ) -> JobRequirements | None:
        result = await self.session.execute(
            select(JobRequirements).where(
                JobRequirements.job_id == job_id
            )
        )

        return result.scalar_one_or_none()