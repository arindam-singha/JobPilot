from __future__ import annotations

from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.embeddings.embedding_provider import EmbeddingProvider
from app.llm.resume_provider import (
    ResumeGenerationProvider,
    ResumeGenerationProviderError,
)
from app.models.candidate_profile import CandidateProfile
from app.models.job import Job
from app.schemas.tailored_resume import (
    ResumeGenerationContext,
    ResumeHeader,
    TailoredResumeDraft,
)
from app.services.tailored_resume_grounding_service import (
    TailoredResumeGroundingService,
)


class TailoredResumeGenerationError(Exception):
    """Base exception for tailored-resume generation failures."""


class TailoredResumeJobNotFoundError(
    TailoredResumeGenerationError
):
    """Raised when the requested job does not exist."""


class TailoredResumeProfileNotFoundError(
    TailoredResumeGenerationError
):
    """Raised when the requested candidate profile does not exist."""


class TailoredResumeProviderFailureError(
    TailoredResumeGenerationError
):
    """Raised when the configured generation provider fails."""


class TailoredResumeValidationError(
    TailoredResumeGenerationError
):
    """Raised when generated content violates grounding constraints."""


class TailoredResumeGenerationService:
    """Orchestrate grounded structured resume generation."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        resume_provider: ResumeGenerationProvider,
        *,
        grounding_service: TailoredResumeGroundingService | None = None,
    ) -> None:
        self.session = session
        self.embedding_provider = embedding_provider
        self.resume_provider = resume_provider

        self.grounding_service = grounding_service or (
            TailoredResumeGroundingService(
                session,
                embedding_provider,
            )
        )

    async def generate(
        self,
        *,
        job_id: UUID,
        profile_id: UUID,
    ) -> TailoredResumeDraft:
        job = await self._get_job(job_id)
        profile = await self._get_profile(profile_id)

        grounding = await self.grounding_service.build_bundle(
            job_id=job_id,
            profile_id=profile_id,
        )

        context = ResumeGenerationContext(
            job_title=job.title,
            company=job.company,
            location=job.location,
            job_description=job.description,
            header=self._build_header(profile),
            grounding=grounding,
        )

        try:
            generated_content = (
                await self.resume_provider.generate_resume(
                    context
                )
            )
        except ResumeGenerationProviderError as exc:
            raise TailoredResumeProviderFailureError(
                "Resume generation provider failed"
            ) from exc

        try:
            return TailoredResumeDraft(
                job_id=job.id,
                profile_id=profile.id,
                header=context.header,
                target_title=generated_content.target_title,
                professional_summary=(
                    generated_content.professional_summary
                ),
                sections=generated_content.sections,
                skills=generated_content.skills,
                evidence_catalog=grounding.selected_evidence,
                generator_provider=(
                    self.resume_provider.provider_name
                ),
                generator_model=self.resume_provider.model_name,
            )
        except ValidationError as exc:
            raise TailoredResumeValidationError(
                "Generated resume failed provenance validation"
            ) from exc

    async def _get_job(
        self,
        job_id: UUID,
    ) -> Job:
        result = await self.session.execute(
            select(Job).where(Job.id == job_id)
        )

        job = result.scalar_one_or_none()

        if job is None:
            raise TailoredResumeJobNotFoundError(
                "Job not found"
            )

        return job

    # async def _get_profile(
    #     self,
    #     profile_id: UUID,
    # ) -> CandidateProfile:
    #     result = await self.session.execute(
    #         select(CandidateProfile).where(
    #             CandidateProfile.id == profile_id
    #         )
    #     )

    #     profile = result.scalar_one_or_none()

    #     if profile is None:
    #         raise TailoredResumeProfileNotFoundError(
    #             "Candidate profile not found"
    #         )

    #     return profile

    async def _get_profile(
        self,
        profile_id: UUID,
    ) -> CandidateProfile:
        result = await self.session.execute(
            select(CandidateProfile)
            .where(CandidateProfile.id == profile_id)
            .options(
                selectinload(CandidateProfile.experiences),
                selectinload(CandidateProfile.skills),
                selectinload(CandidateProfile.educations),
                selectinload(CandidateProfile.projects),
                selectinload(CandidateProfile.publications),
                selectinload(CandidateProfile.certifications),
                selectinload(CandidateProfile.achievements),
            )
        )

        profile = result.scalar_one_or_none()

        if profile is None:
            raise TailoredResumeProfileNotFoundError(
                "Candidate profile not found"
            )

        return profile

    @staticmethod
    def _build_header(
        profile: CandidateProfile,
    ) -> ResumeHeader:
        return ResumeHeader(
            full_name=profile.full_name,
            email=profile.email,
            phone=profile.phone,
            location=profile.location,
            linkedin_url=profile.linkedin_url,
            github_url=profile.github_url,
            portfolio_url=profile.portfolio_url,
        )