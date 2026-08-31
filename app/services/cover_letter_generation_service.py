from __future__ import annotations

from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.embedding_provider import EmbeddingProvider
from app.llm.cover_letter_provider import (
    CoverLetterGenerationProvider,
    CoverLetterGenerationProviderError,
)
from app.models.candidate_profile import CandidateProfile
from app.models.job import Job
from app.schemas.cover_letter import (
    CoverLetterGenerationContext,
    TailoredCoverLetterDraft,
)
from app.schemas.tailored_resume import ResumeHeader
from app.services.cover_letter_grounding_validator import (
    CoverLetterGroundingValidationError,
    CoverLetterGroundingValidator,
)
from app.services.tailored_resume_grounding_service import TailoredResumeGroundingService


class CoverLetterGenerationError(Exception):
    """Base error for grounded cover-letter generation."""


class CoverLetterJobNotFoundError(CoverLetterGenerationError):
    pass


class CoverLetterProfileNotFoundError(CoverLetterGenerationError):
    pass


class CoverLetterProviderFailureError(CoverLetterGenerationError):
    pass


class CoverLetterValidationError(CoverLetterGenerationError):
    pass


class CoverLetterGenerationService:
    """Generate a structured letter from the Phase 9 grounding bundle."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        cover_letter_provider: CoverLetterGenerationProvider,
        *,
        grounding_service: TailoredResumeGroundingService | None = None,
        validator: CoverLetterGroundingValidator | None = None,
    ) -> None:
        self.session = session
        self.provider = cover_letter_provider
        self.grounding_service = grounding_service or TailoredResumeGroundingService(
            session, embedding_provider
        )
        self.validator = validator or CoverLetterGroundingValidator()

    async def generate(self, *, job_id: UUID, profile_id: UUID) -> TailoredCoverLetterDraft:
        job = await self._get_job(job_id)
        profile = await self._get_profile(profile_id)
        grounding = await self.grounding_service.build_bundle(
            job_id=job_id,
            profile_id=profile_id,
        )
        header = ResumeHeader(
            full_name=profile.full_name,
            email=profile.email,
            phone=profile.phone,
            location=profile.location,
            linkedin_url=profile.linkedin_url,
            github_url=profile.github_url,
            portfolio_url=profile.portfolio_url,
        )
        context = CoverLetterGenerationContext(
            job_title=job.title,
            company=job.company,
            location=job.location,
            job_description=job.description,
            header=header,
            grounding=grounding,
        )
        try:
            content = await self.provider.generate_cover_letter(context)
            draft = TailoredCoverLetterDraft(
                job_id=job.id,
                profile_id=profile.id,
                header=header,
                job_title=job.title,
                company=job.company,
                evidence_catalog=grounding.selected_evidence,
                generator_provider=self.provider.provider_name,
                generator_model=self.provider.model_name,
                **content.model_dump(),
            )
            self.validator.validate(draft)
            return draft
        except CoverLetterGenerationProviderError as exc:
            raise CoverLetterProviderFailureError("Cover-letter provider failed") from exc
        except (ValidationError, CoverLetterGroundingValidationError) as exc:
            raise CoverLetterValidationError("Cover letter failed grounding validation") from exc

    async def _get_job(self, job_id: UUID) -> Job:
        job = (await self.session.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
        if job is None:
            raise CoverLetterJobNotFoundError("Job not found")
        return job

    async def _get_profile(self, profile_id: UUID) -> CandidateProfile:
        profile = (
            await self.session.execute(
                select(CandidateProfile).where(CandidateProfile.id == profile_id)
            )
        ).scalar_one_or_none()
        if profile is None:
            raise CoverLetterProfileNotFoundError("Candidate profile not found")
        return profile
