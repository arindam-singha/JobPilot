from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.embeddings.embedding_provider_factory import (
    EmbeddingProviderConfigurationError,
    create_embedding_provider,
)
from app.llm.resume_provider_factory import (
    ResumeProviderConfigurationError,
    create_resume_generation_provider,
)
from app.schemas.tailored_resume_record import (
    TailoredResumeRecordList,
    TailoredResumeRecordRead,
)
from app.services.hybrid_job_candidate_matching_service import (
    HybridEmbeddedEvidenceNotFoundError,
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
from app.services.tailored_resume_generation_service import (
    TailoredResumeGenerationService,
    TailoredResumeJobNotFoundError,
    TailoredResumeProfileNotFoundError,
    TailoredResumeProviderFailureError,
    TailoredResumeValidationError,
)
from app.services.tailored_resume_grounding_service import (
    TailoredResumeEvidenceIntegrityError,
    TailoredResumeEvidenceNotFoundError,
)
from app.services.tailored_resume_markdown_renderer import (
    TailoredResumeMarkdownRenderer,
    TailoredResumeMarkdownRenderingError,
)
from app.services.tailored_resume_pdf_renderer import (
    TailoredResumePdfRenderer,
    TailoredResumePdfRenderingError,
)
from app.services.tailored_resume_service import (
    TailoredResumeNotFoundError,
    TailoredResumeService,
)


router = APIRouter(prefix="/api/v1/jobs", tags=["tailored-resumes"])


@router.post(
    "/{job_id}/tailored-resumes/{profile_id}",
    response_model=TailoredResumeRecordRead,
    status_code=status.HTTP_201_CREATED,
)
async def generate_tailored_resume(
    job_id: UUID,
    profile_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TailoredResumeRecordRead:
    """Generate and persist a grounded tailored resume."""

    try:
        embedding_provider = create_embedding_provider()
        resume_provider = create_resume_generation_provider()
    except (
        EmbeddingProviderConfigurationError,
        ResumeProviderConfigurationError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    generation_service = TailoredResumeGenerationService(
        db,
        embedding_provider,
        resume_provider,
    )
    service = TailoredResumeService(db, generation_service)

    try:
        return await service.generate_and_save(
            job_id=job_id,
            profile_id=profile_id,
        )
    except (
        TailoredResumeJobNotFoundError,
        MatchingJobNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        ) from exc
    except MatchingRequirementsNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Structured job requirements not found",
        ) from exc
    except (
        TailoredResumeProfileNotFoundError,
        MatchingCandidateProfileNotFoundError,
    ) as exc:
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
    except TailoredResumeEvidenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No sufficiently relevant evidence was found "
                "for resume generation"
            ),
        ) from exc
    except (
        TailoredResumeEvidenceIntegrityError,
        TailoredResumeValidationError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Generated resume failed provenance validation",
        ) from exc
    except (
        SemanticEvidenceProviderFailureError,
        TailoredResumeProviderFailureError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Resume generation provider failed",
        ) from exc


@router.get(
    "/{job_id}/tailored-resumes/{profile_id}",
    response_model=TailoredResumeRecordList,
    status_code=status.HTTP_200_OK,
)
async def list_tailored_resumes(
    job_id: UUID,
    profile_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> TailoredResumeRecordList:
    """List persisted resumes for one job and profile."""

    return await TailoredResumeService(db).list_resumes(
        job_id=job_id,
        profile_id=profile_id,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{job_id}/tailored-resumes/{profile_id}/{resume_id}",
    response_model=TailoredResumeRecordRead,
    status_code=status.HTTP_200_OK,
)
async def get_tailored_resume(
    job_id: UUID,
    profile_id: UUID,
    resume_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TailoredResumeRecordRead:
    """Retrieve one scoped persisted tailored resume."""

    try:
        return await TailoredResumeService(db).get_resume(
            resume_id=resume_id,
            job_id=job_id,
            profile_id=profile_id,
        )
    except TailoredResumeNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tailored resume not found",
        ) from exc


@router.get(
    "/{job_id}/tailored-resumes/{profile_id}/{resume_id}/markdown",
    response_class=PlainTextResponse,
    status_code=status.HTTP_200_OK,
)
async def get_tailored_resume_markdown(
    job_id: UUID,
    profile_id: UUID,
    resume_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PlainTextResponse:
    """Render a persisted tailored resume as ATS Markdown."""

    try:
        record = await TailoredResumeService(db).get_resume(
            resume_id=resume_id,
            job_id=job_id,
            profile_id=profile_id,
        )
    except TailoredResumeNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tailored resume not found",
        ) from exc

    try:
        markdown = TailoredResumeMarkdownRenderer().render(
            record.structured_content
        )
    except TailoredResumeMarkdownRenderingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Tailored resume could not be rendered",
        ) from exc

    return PlainTextResponse(content=markdown, media_type="text/markdown")


@router.get(
    "/{job_id}/tailored-resumes/{profile_id}/{resume_id}/pdf",
    response_class=Response,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "content": {"application/pdf": {}},
            "description": "ATS-friendly tailored resume PDF",
        }
    },
)
async def get_tailored_resume_pdf(
    job_id: UUID,
    profile_id: UUID,
    resume_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Render a persisted tailored resume as an ATS PDF."""

    try:
        record = await TailoredResumeService(db).get_resume(
            resume_id=resume_id,
            job_id=job_id,
            profile_id=profile_id,
        )
    except TailoredResumeNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tailored resume not found",
        ) from exc

    try:
        pdf_bytes = TailoredResumePdfRenderer().render(
            record.structured_content
        )
    except TailoredResumePdfRenderingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Tailored resume PDF could not be rendered",
        ) from exc

    filename = f"tailored-resume-{record.id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )
