from __future__ import annotations

import base64
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.embeddings.embedding_provider_factory import create_embedding_provider
from app.llm.cover_letter_provider_factory import create_cover_letter_generation_provider
from app.llm.job_requirements_provider_factory import create_job_requirements_provider
from app.llm.resume_provider_factory import create_resume_generation_provider
from app.schemas.application_package import ApplicationPackageCreate, ApplicationPackageRead
from app.services.candidate_evidence_embedding_service import (
    CandidateEvidenceEmbeddingService,
)
from app.services.cover_letter_generation_service import CoverLetterGenerationService
from app.services.cover_letter_markdown_renderer import CoverLetterMarkdownRenderer
from app.services.cover_letter_pdf_renderer import CoverLetterPdfRenderer
from app.services.hybrid_job_candidate_matching_service import (
    HybridJobCandidateMatchingService,
)
from app.services.job_ingestion_service import JobIngestionService
from app.services.job_requirements_extraction_service import JobRequirementsExtractionService
from app.services.tailored_resume_generation_service import TailoredResumeGenerationService
from app.services.tailored_resume_markdown_renderer import TailoredResumeMarkdownRenderer
from app.services.tailored_resume_pdf_renderer import TailoredResumePdfRenderer
from app.services.tailored_resume_html_renderer import (
    TailoredResumeHtmlRenderer,
)

router = APIRouter(prefix="/api/v1/application-packages", tags=["application-packages"])


@router.post("", response_model=ApplicationPackageRead, status_code=status.HTTP_201_CREATED)
async def generate_application_package(
    payload: ApplicationPackageCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApplicationPackageRead:
    """Generate a complete personal application package in one request."""

    embedding_provider = create_embedding_provider()
    try:
        job_result = await JobIngestionService().ingest_job(db, payload)
        evidence = await CandidateEvidenceEmbeddingService(
            session=db,
            provider=embedding_provider,
        ).embed_profile_evidence(payload.profile_id, force=False)
        if not evidence:
            raise HTTPException(status_code=409, detail="Candidate profile has no evidence")

        requirements = await JobRequirementsExtractionService(
            db,
            create_job_requirements_provider(),
        ).extract_and_persist(job_result.job.id)
        match = await HybridJobCandidateMatchingService(
            db,
            embedding_provider,
        ).match(job_id=job_result.job.id, profile_id=payload.profile_id)
        resume = await TailoredResumeGenerationService(
            db,
            embedding_provider,
            create_resume_generation_provider(),
        ).generate(job_id=job_result.job.id, profile_id=payload.profile_id)
        cover_letter = await CoverLetterGenerationService(
            db,
            embedding_provider,
            create_cover_letter_generation_provider(),
        ).generate(job_id=job_result.job.id, profile_id=payload.profile_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc) or exc.__class__.__name__,
        ) from exc
    
    resume_html = TailoredResumeHtmlRenderer().render(resume)
    resume_markdown = TailoredResumeMarkdownRenderer().render(resume)
    cover_letter_markdown = CoverLetterMarkdownRenderer().render(cover_letter)
    resume_pdf = TailoredResumePdfRenderer().render(resume)
    cover_letter_pdf = CoverLetterPdfRenderer().render(cover_letter)
    return ApplicationPackageRead(
        job=job_result.job,
        requirements=requirements,
        match=match,
        resume=resume,
        cover_letter=cover_letter,
        resume_html=resume_html,
        resume_markdown=resume_markdown,
        cover_letter_markdown=cover_letter_markdown,
        resume_pdf_base64=base64.b64encode(resume_pdf).decode("ascii"),
        cover_letter_pdf_base64=base64.b64encode(
            cover_letter_pdf
        ).decode("ascii"),
    )
