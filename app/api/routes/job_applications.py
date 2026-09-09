from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.job_application import (
    JobApplicationCreate,
    JobApplicationRead,
    JobApplicationStatus,
    JobApplicationUpdate,
)
from app.services.job_application_service import (
    DuplicateJobApplicationError,
    JobApplicationNotFoundError,
    JobApplicationReferenceNotFoundError,
    JobApplicationService,
)


router = APIRouter(
    prefix="/api/v1/applications",
    tags=["applications"],
)


@router.post(
    "",
    response_model=JobApplicationRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_application(
    payload: JobApplicationCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JobApplicationRead:
    try:
        return await JobApplicationService().create(db, payload)
    except JobApplicationReferenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DuplicateJobApplicationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.get(
    "",
    response_model=list[JobApplicationRead],
)
async def list_applications(
    db: Annotated[AsyncSession, Depends(get_db)],
    profile_id: UUID | None = None,
    status_filter: JobApplicationStatus | None = None,
) -> list[JobApplicationRead]:
    return await JobApplicationService().list(
        db,
        profile_id=profile_id,
        status=status_filter,
    )


@router.get(
    "/{application_id}",
    response_model=JobApplicationRead,
)
async def get_application(
    application_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JobApplicationRead:
    try:
        return await JobApplicationService().get(
            db,
            application_id,
        )
    except JobApplicationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.patch(
    "/{application_id}",
    response_model=JobApplicationRead,
)
async def update_application(
    application_id: UUID,
    payload: JobApplicationUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JobApplicationRead:
    try:
        return await JobApplicationService().update(
            db,
            application_id,
            payload,
        )
    except JobApplicationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.delete(
    "/{application_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_application(
    application_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    try:
        await JobApplicationService().delete(
            db,
            application_id,
        )
    except JobApplicationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)