from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.job import JobCreate, JobIngestRequest, JobRead
from app.services.job_service import JobService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobIngestionResult:
    """Result of a job ingestion operation."""

    job: JobRead
    created: bool


class JobIngestionService:
    """Service responsible for validating and normalizing job ingestion."""

    def __init__(self) -> None:
        self.job_service = JobService()

    @staticmethod
    def identify_source(job_url: str) -> str:
        """Identify the job source from the URL hostname."""

        parsed = urlparse(job_url)
        hostname = (parsed.hostname or "").lower()

        if not hostname:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid job URL",
            )

        if hostname == "linkedin.com" or hostname.endswith(".linkedin.com"):
            return "linkedin"

        if hostname == "indeed.com" or hostname.endswith(".indeed.com"):
            return "indeed"

        if hostname == "naukri.com" or hostname.endswith(".naukri.com"):
            return "naukri"

        if hostname == "glassdoor.com" or hostname.endswith(".glassdoor.com"):
            return "glassdoor"

        if hostname == "greenhouse.io" or hostname.endswith(".greenhouse.io"):
            return "greenhouse"

        if hostname == "lever.co" or hostname.endswith(".lever.co"):
            return "lever"

        return "other"

    @staticmethod
    def validate_url(job_url: str) -> str:
        """Validate and normalize a job URL."""

        normalized_url = job_url.strip()
        parsed = urlparse(normalized_url)

        if parsed.scheme not in {"http", "https"}:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Job URL must use HTTP or HTTPS",
            )

        if not parsed.netloc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid job URL",
            )

        return normalized_url.rstrip("/")

    async def ingest_job(
        self,
        session: AsyncSession,
        payload: JobIngestRequest,
    ) -> JobIngestionResult:
        """Validate, normalize, deduplicate, and persist a job."""

        logger.info("Job ingestion started")

        job_url = self.validate_url(payload.job_url)
        source = self.identify_source(job_url)

        logger.info(
            "Job source identified: source=%s",
            source,
        )

        existing_job = await self.job_service.get_job_by_url(
            session,
            job_url,
        )

        if existing_job is not None:
            logger.info(
                "Duplicate job detected: source=%s",
                source,
            )

            return JobIngestionResult(
                job=existing_job,
                created=False,
            )

        job_payload = JobCreate(
            title=payload.title.strip(),
            company=payload.company.strip(),
            location=payload.location.strip()
            if payload.location
            else None,
            job_url=job_url,
            description=payload.job_description.strip(),
            source=source,
        )

        try:
            created_job = await self.job_service.create_job(
                session,
                job_payload,
            )
        except SQLAlchemyError as exc:
            await session.rollback()

            logger.exception(
                "Job ingestion failed while saving job: source=%s",
                source,
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to ingest job",
            ) from exc

        logger.info(
            "Job created successfully: job_id=%s source=%s",
            created_job.id,
            source,
        )

        return JobIngestionResult(
            job=created_job,
            created=True,
        )