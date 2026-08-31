from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.job import JobCreate, JobIngestRequest, JobRead
from app.services.job_ingestion_service import JobIngestionService
from app.services.job_service import JobService

router = APIRouter(prefix="/api/v1")


@router.post(
    "/jobs/ingest",
    response_model=JobRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_200_OK: {
            "model": JobRead,
            "description": "Job already exists for the supplied URL",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Failed to ingest job",
        },
    },
)

async def ingest_job(
    payload: JobIngestRequest,
    response: Response,
    session: AsyncSession = Depends(get_db),
) -> JobRead:
    service = JobIngestionService()

    result = await service.ingest_job(
        session,
        payload,
    )

    if not result.created:
        response.status_code = status.HTTP_200_OK

    return result.job


@router.post(
    "/jobs",
    response_model=JobRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_job(
    payload: JobCreate,
    session: AsyncSession = Depends(get_db),
) -> JobRead:
    service = JobService()

    return await service.create_job(
        session,
        payload,
    )


@router.get(
    "/jobs",
    response_model=list[JobRead],
)
async def list_jobs(
    session: AsyncSession = Depends(get_db),
) -> list[JobRead]:
    service = JobService()

    return await service.list_jobs(session)


@router.get(
    "/jobs/{job_id}",
    response_model=JobRead,
)
async def get_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_db),
) -> JobRead:
    service = JobService()

    return await service.get_job(
        session,
        job_id,
    )