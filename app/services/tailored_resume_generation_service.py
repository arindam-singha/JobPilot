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
    ResumeAchievementEntry,
    ResumeCertificationEntry,
    ResumeEducationEntry,
    ResumeExperienceEntry,
    ResumeGenerationContext,
    ResumeHeader,
    ResumeProjectEntry,
    ResumePublicationEntry,
    ResumeSkillGroup,
    TailoredResumeDraft,
)
from app.services.resume_claim_validator import (
    ResumeClaimValidationError,
    ResumeClaimValidator,
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
        claim_validator: ResumeClaimValidator | None = None,
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
        self.claim_validator = claim_validator or ResumeClaimValidator()

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
            self.claim_validator.validate(
                generated_content,
                context,
            )
        except ResumeGenerationProviderError as exc:
            raise TailoredResumeProviderFailureError(
                "Resume generation provider failed"
            ) from exc
        except ResumeClaimValidationError as exc:
            raise TailoredResumeValidationError(
                "Generated resume failed provenance validation: "
                f"{exc}"
            ) from exc

        try:
            return TailoredResumeDraft(
                job_id=job.id,
                profile_id=profile.id,
                header=context.header,
                target_title=job.title,
                verified_summary=profile.professional_summary,
                professional_summary=(
                    generated_content.professional_summary
                ),
                sections=generated_content.sections,
                skills=generated_content.skills,
                skill_groups=self._build_skill_groups(profile),
                experiences=self._build_experiences(profile),
                projects=self._build_projects(profile),
                education=self._build_education(profile),
                publications=self._build_publications(profile),
                certifications=self._build_certifications(profile),
                achievements=self._build_achievements(profile),
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

    @staticmethod
    def _iso_date(value) -> str | None:
        if value is None:
            return None
        return value.isoformat()

    @staticmethod
    def _split_bullets(*values: str | None) -> list[str]:
        bullets: list[str] = []

        for value in values:
            if not value:
                continue

            normalized = value.replace("\r\n", "\n").replace("\r", "\n")

            for line in normalized.split("\n"):
                cleaned = line.strip().lstrip("-•* ").strip()
                if cleaned and cleaned not in bullets:
                    bullets.append(cleaned)

        return bullets

    @staticmethod
    def _build_skill_groups(
        profile: CandidateProfile,
    ) -> list[ResumeSkillGroup]:
        grouped: dict[str, list[str]] = {}

        for skill in profile.skills:
            name = skill.name.strip()
            if not name:
                continue

            category = (
                skill.category.strip()
                if skill.category and skill.category.strip()
                else "Technical Skills"
            )

            names = grouped.setdefault(category, [])

            if name.casefold() not in {
                existing.casefold() for existing in names
            }:
                names.append(name)

        return [
            ResumeSkillGroup(
                category=category,
                skills=sorted(names, key=str.casefold),
            )
            for category, names in sorted(
                grouped.items(),
                key=lambda item: item[0].casefold(),
            )
        ]

    @classmethod
    def _build_experiences(
        cls,
        profile: CandidateProfile,
    ) -> list[ResumeExperienceEntry]:
        ordered = sorted(
            profile.experiences,
            key=lambda item: (
                item.start_date is not None,
                item.start_date,
                item.is_current,
            ),
            reverse=True,
        )

        return [
            ResumeExperienceEntry(
                role=item.role,
                company=item.company,
                location=item.location,
                start_date=cls._iso_date(item.start_date),
                end_date=cls._iso_date(item.end_date),
                is_current=item.is_current,
                bullets=cls._split_bullets(
                    item.description,
                    item.achievements,
                ),
            )
            for item in ordered
        ]

    @staticmethod
    def _build_projects(
        profile: CandidateProfile,
    ) -> list[ResumeProjectEntry]:
        return [
            ResumeProjectEntry(
                name=item.name,
                technologies=item.technologies,
                description=item.description,
                achievements=item.achievements,
                project_url=(
                    str(item.project_url)
                    if item.project_url
                    else None
                ),
            )
            for item in profile.projects
        ]

    @classmethod
    def _build_education(
        cls,
        profile: CandidateProfile,
    ) -> list[ResumeEducationEntry]:
        ordered = sorted(
            profile.educations,
            key=lambda item: (
                item.end_date is not None,
                item.end_date,
                item.start_date is not None,
                item.start_date,
            ),
            reverse=True,
        )

        return [
            ResumeEducationEntry(
                institution=item.institution,
                degree=item.degree,
                field_of_study=item.field_of_study,
                location=item.location,
                start_date=cls._iso_date(item.start_date),
                end_date=cls._iso_date(item.end_date),
                description=item.description,
            )
            for item in ordered
        ]

    @classmethod
    def _build_publications(
        cls,
        profile: CandidateProfile,
    ) -> list[ResumePublicationEntry]:
        ordered = sorted(
            profile.publications,
            key=lambda item: (
                item.publication_date is not None,
                item.publication_date,
            ),
            reverse=True,
        )

        return [
            ResumePublicationEntry(
                title=item.title,
                venue=item.venue,
                publication_date=cls._iso_date(
                    item.publication_date
                ),
                url=str(item.url) if item.url else None,
                description=item.description,
            )
            for item in ordered
        ]

    @classmethod
    def _build_certifications(
        cls,
        profile: CandidateProfile,
    ) -> list[ResumeCertificationEntry]:
        return [
            ResumeCertificationEntry(
                name=item.name,
                issuing_organization=item.issuing_organization,
                issue_date=cls._iso_date(item.issue_date),
                expiry_date=cls._iso_date(item.expiry_date),
                credential_id=item.credential_id,
                credential_url=(
                    str(item.credential_url)
                    if item.credential_url
                    else None
                ),
            )
            for item in profile.certifications
        ]

    @classmethod
    def _build_achievements(
        cls,
        profile: CandidateProfile,
    ) -> list[ResumeAchievementEntry]:
        ordered = sorted(
            profile.achievements,
            key=lambda item: (
                item.date is not None,
                item.date,
            ),
            reverse=True,
        )

        return [
            ResumeAchievementEntry(
                title=item.title,
                description=item.description,
                date=cls._iso_date(item.date),
            )
            for item in ordered
        ]

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
