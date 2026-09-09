from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.candidate_profile import CandidateProfile
from app.models.job import Job
from app.models.job_application import JobApplication
from app.schemas.job_application import (
    JobApplicationCreate,
    JobApplicationRead,
    JobApplicationStatus,
    JobApplicationUpdate,
)


class JobApplicationError(Exception):
    """Base exception for application tracking failures."""


class JobApplicationNotFoundError(JobApplicationError):
    """Raised when an application does not exist."""


class JobApplicationReferenceNotFoundError(JobApplicationError):
    """Raised when the referenced job or profile does not exist."""


class DuplicateJobApplicationError(JobApplicationError):
    """Raised when an application already exists for the job and profile."""


class JobApplicationService:
    """Create and manage tracked job applications."""

    async def create(
        self,
        session: AsyncSession,
        payload: JobApplicationCreate,
    ) -> JobApplicationRead:
        await self._validate_references(
            session,
            job_id=payload.job_id,
            profile_id=payload.profile_id,
        )

        existing = await session.scalar(
            select(JobApplication).where(
                JobApplication.job_id == payload.job_id,
                JobApplication.profile_id == payload.profile_id,
            )
        )
        if existing is not None:
            raise DuplicateJobApplicationError(
                "An application already exists for this job and profile"
            )

        applied_at = payload.applied_at
        if (
            payload.status == JobApplicationStatus.APPLIED
            and applied_at is None
        ):
            applied_at = datetime.now(timezone.utc)

        application = JobApplication(
            job_id=payload.job_id,
            profile_id=payload.profile_id,
            status=payload.status.value,
            resume_snapshot=payload.resume_snapshot,
            cover_letter_snapshot=payload.cover_letter_snapshot,
            application_url=payload.application_url,
            notes=payload.notes,
            applied_at=applied_at,
        )

        session.add(application)

        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise DuplicateJobApplicationError(
                "An application already exists for this job and profile"
            ) from exc

        await session.refresh(application)
        return JobApplicationRead.model_validate(application)

    async def list(
        self,
        session: AsyncSession,
        *,
        profile_id: UUID | None = None,
        status: JobApplicationStatus | None = None,
    ) -> list[JobApplicationRead]:
        query = select(JobApplication)

        if profile_id is not None:
            query = query.where(
                JobApplication.profile_id == profile_id
            )

        if status is not None:
            query = query.where(
                JobApplication.status == status.value
            )

        query = query.order_by(JobApplication.created_at.desc())

        result = await session.scalars(query)

        return [
            JobApplicationRead.model_validate(application)
            for application in result.all()
        ]

    async def get(
        self,
        session: AsyncSession,
        application_id: UUID,
    ) -> JobApplicationRead:
        application = await session.get(
            JobApplication,
            application_id,
        )

        if application is None:
            raise JobApplicationNotFoundError(
                "Job application not found"
            )

        return JobApplicationRead.model_validate(application)

    async def update(
        self,
        session: AsyncSession,
        application_id: UUID,
        payload: JobApplicationUpdate,
    ) -> JobApplicationRead:
        application = await session.get(
            JobApplication,
            application_id,
        )

        if application is None:
            raise JobApplicationNotFoundError(
                "Job application not found"
            )

        changes = payload.model_dump(exclude_unset=True)

        status_value = changes.pop("status", None)
        if status_value is not None:
            application.status = status_value.value

            if (
                status_value == JobApplicationStatus.APPLIED
                and "applied_at" not in changes
                and application.applied_at is None
            ):
                application.applied_at = datetime.now(timezone.utc)

        for field, value in changes.items():
            setattr(application, field, value)

        await session.commit()
        await session.refresh(application)

        return JobApplicationRead.model_validate(application)

    async def delete(
        self,
        session: AsyncSession,
        application_id: UUID,
    ) -> None:
        application = await session.get(
            JobApplication,
            application_id,
        )

        if application is None:
            raise JobApplicationNotFoundError(
                "Job application not found"
            )

        await session.delete(application)
        await session.commit()

    async def _validate_references(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        profile_id: UUID,
    ) -> None:
        job = await session.get(Job, job_id)
        if job is None:
            raise JobApplicationReferenceNotFoundError(
                "Referenced job not found"
            )

        profile = await session.get(
            CandidateProfile,
            profile_id,
        )
        if profile is None:
            raise JobApplicationReferenceNotFoundError(
                "Referenced candidate profile not found"
            )