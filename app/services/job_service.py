from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import Job
from app.schemas.job import JobCreate, JobRead


class JobService:
    """Application service boundary for job records."""

    async def create_job(
        self,
        session: AsyncSession,
        payload: JobCreate,
    ) -> JobRead:
        """Create and persist a new job."""

        job = Job(
            title=payload.title,
            company=payload.company,
            location=payload.location,
            job_url=payload.job_url,
            description=payload.description,
            source=payload.source,
        )

        session.add(job)

        await session.commit()
        await session.refresh(job)

        return JobRead.model_validate(job)

    async def get_job_by_url(
        self,
        session: AsyncSession,
        job_url: str,
    ) -> JobRead | None:
        """Return a job matching the supplied URL, if one exists."""

        result = await session.execute(
            select(Job).where(Job.job_url == job_url)
        )

        job = result.scalar_one_or_none()

        if job is None:
            return None

        return JobRead.model_validate(job)

    async def list_jobs(
        self,
        session: AsyncSession,
    ) -> list[JobRead]:
        """Return all jobs ordered by creation time."""

        result = await session.execute(
            select(Job).order_by(Job.created_at.desc())
        )

        jobs = result.scalars().all()

        return [JobRead.model_validate(job) for job in jobs]

    async def get_job(
        self,
        session: AsyncSession,
        job_id: UUID,
    ) -> JobRead:
        """Return a job by ID or raise 404."""

        result = await session.execute(
            select(Job).where(Job.id == job_id)
        )

        job = result.scalar_one_or_none()

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found",
            )

        return JobRead.model_validate(job)