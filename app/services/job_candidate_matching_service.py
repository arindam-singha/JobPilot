from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.matching.text_matching import (
    combine_evidence_text,
    contains_normalized_phrase,
    token_coverage,
)
from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import CandidateProfile
from app.models.job import Job
from app.models.job_requirements import JobRequirements
from app.schemas.job_candidate_match import (
    ExperienceYearsMatch,
    JobCandidateMatchRead,
    MatchCategoryResult,
    RequirementMatch,
    SupportingEvidence,
)


class JobCandidateMatchingError(Exception):
    """Base exception for deterministic candidate matching."""


class MatchingJobNotFoundError(JobCandidateMatchingError):
    """Raised when the supplied job does not exist."""


class MatchingRequirementsNotFoundError(JobCandidateMatchingError):
    """Raised when structured requirements do not exist for a job."""


class MatchingCandidateProfileNotFoundError(JobCandidateMatchingError):
    """Raised when the supplied candidate profile does not exist."""


class MatchingCandidateEvidenceNotFoundError(JobCandidateMatchingError):
    """Raised when the candidate has no evidence to match."""


@dataclass(frozen=True, slots=True)
class CategoryDefinition:
    name: str
    requirements: list[str]
    evidence_types: set[str]
    base_weight: float


class JobCandidateMatchingService:
    """Compare structured job requirements with candidate evidence."""

    REQUIRED_SKILLS_WEIGHT = 0.40
    REQUIRED_EXPERIENCE_WEIGHT = 0.30
    EDUCATION_WEIGHT = 0.15
    PREFERRED_SKILLS_WEIGHT = 0.10
    CERTIFICATIONS_WEIGHT = 0.05

    EXACT_MATCH_SCORE = 1.0
    STRONG_TOKEN_MATCH_SCORE = 0.8
    MINIMUM_MATCH_SCORE = 0.6

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def match(
        self,
        *,
        job_id: UUID,
        profile_id: UUID,
    ) -> JobCandidateMatchRead:
        await self._ensure_job_exists(job_id)

        requirements = await self._get_requirements(job_id)
        profile = await self._get_profile(profile_id)
        evidence = await self._get_evidence(profile_id)

        if not evidence:
            raise MatchingCandidateEvidenceNotFoundError(
                f"Candidate profile {profile_id} has no evidence"
            )

        category_definitions = [
            CategoryDefinition(
                name="required_skills",
                requirements=requirements.required_skills,
                evidence_types={
                    "skill",
                    "experience",
                    "project",
                    "summary",
                    "general",
                },
                base_weight=self.REQUIRED_SKILLS_WEIGHT,
            ),
            CategoryDefinition(
                name="preferred_skills",
                requirements=requirements.preferred_skills,
                evidence_types={
                    "skill",
                    "experience",
                    "project",
                    "summary",
                    "general",
                },
                base_weight=self.PREFERRED_SKILLS_WEIGHT,
            ),
            CategoryDefinition(
                name="required_experience",
                requirements=requirements.required_experience,
                evidence_types={
                    "experience",
                    "project",
                    "achievement",
                    "summary",
                    "general",
                },
                base_weight=self.REQUIRED_EXPERIENCE_WEIGHT,
            ),
            CategoryDefinition(
                name="education",
                requirements=requirements.education_requirements,
                evidence_types={
                    "education",
                    "summary",
                    "general",
                },
                base_weight=self.EDUCATION_WEIGHT,
            ),
            CategoryDefinition(
                name="certifications",
                requirements=requirements.certifications,
                evidence_types={
                    "certification",
                    "skill",
                    "summary",
                    "general",
                },
                base_weight=self.CERTIFICATIONS_WEIGHT,
            ),
        ]

        category_results = {
            definition.name: self._match_category(
                definition=definition,
                evidence=evidence,
            )
            for definition in category_definitions
        }

        experience_years = self._match_experience_years(
            required_years=requirements.minimum_experience_years,
            candidate_years=profile.total_experience_years,
        )

        overall_score = self._calculate_overall_score(
            category_definitions=category_definitions,
            category_results=category_results,
            experience_years=experience_years,
        )

        all_requirement_results = [
            requirement_match
            for category_result in category_results.values()
            for requirement_match in category_result.requirements
        ]

        matched_requirements = [
            result.requirement
            for result in all_requirement_results
            if result.matched
        ]

        missing_requirements = [
            result.requirement
            for result in all_requirement_results
            if not result.matched
        ]

        if (
            experience_years.required_years is not None
            and experience_years.matched is False
        ):
            missing_requirements.append(
                f"Minimum {experience_years.required_years:g} "
                "years of experience"
            )

        return JobCandidateMatchRead(
            job_id=job_id,
            profile_id=profile_id,
            overall_score=round(overall_score, 2),
            required_skills=category_results["required_skills"],
            preferred_skills=category_results["preferred_skills"],
            required_experience=category_results[
                "required_experience"
            ],
            education=category_results["education"],
            certifications=category_results["certifications"],
            experience_years=experience_years,
            matched_requirements=matched_requirements,
            missing_requirements=missing_requirements,
            evidence_considered=len(evidence),
        )

    def _match_category(
        self,
        *,
        definition: CategoryDefinition,
        evidence: list[CandidateEvidence],
    ) -> MatchCategoryResult:
        requirement_results = [
            self._match_requirement(
                requirement=requirement,
                evidence=evidence,
                allowed_evidence_types=definition.evidence_types,
            )
            for requirement in definition.requirements
        ]

        if not requirement_results:
            category_score = 0.0
        else:
            category_score = (
                sum(item.score for item in requirement_results)
                / len(requirement_results)
                * 100
            )

        matched_count = sum(
            1 for item in requirement_results if item.matched
        )

        return MatchCategoryResult(
            category=definition.name,
            score=round(category_score, 2),
            total_requirements=len(requirement_results),
            matched_requirements=matched_count,
            requirements=requirement_results,
        )

    def _match_requirement(
        self,
        *,
        requirement: str,
        evidence: list[CandidateEvidence],
        allowed_evidence_types: set[str],
    ) -> RequirementMatch:
        supporting: list[SupportingEvidence] = []

        for item in evidence:
            if item.evidence_type not in allowed_evidence_types:
                continue

            searchable_text = combine_evidence_text(
                [
                    item.title,
                    item.content,
                ]
            )

            score = self._calculate_evidence_match_score(
                requirement=requirement,
                evidence_text=searchable_text,
            )

            if score < self.MINIMUM_MATCH_SCORE:
                continue

            supporting.append(
                SupportingEvidence(
                    evidence_id=item.id,
                    evidence_type=item.evidence_type,
                    title=item.title,
                    source_type=item.source_type,
                    score=score,
                )
            )

        supporting.sort(
            key=lambda item: (
                -item.score,
                item.title.casefold(),
                str(item.evidence_id),
            )
        )

        best_score = supporting[0].score if supporting else 0.0

        return RequirementMatch(
            requirement=requirement,
            matched=best_score >= self.MINIMUM_MATCH_SCORE,
            score=best_score,
            supporting_evidence=supporting[:5],
        )

    def _calculate_evidence_match_score(
        self,
        *,
        requirement: str,
        evidence_text: str,
    ) -> float:
        if contains_normalized_phrase(
            requirement,
            evidence_text,
        ):
            return self.EXACT_MATCH_SCORE

        coverage = token_coverage(
            requirement,
            evidence_text,
        )

        if coverage >= 1.0:
            return self.STRONG_TOKEN_MATCH_SCORE

        if coverage >= 0.75:
            return 0.7

        if coverage >= 0.5:
            return self.MINIMUM_MATCH_SCORE

        return 0.0

    def _match_experience_years(
        self,
        *,
        required_years: float | None,
        candidate_years: float | None,
    ) -> ExperienceYearsMatch:
        if required_years is None:
            return ExperienceYearsMatch(
                required_years=None,
                candidate_years=candidate_years,
                matched=None,
                score=None,
            )

        if candidate_years is None:
            return ExperienceYearsMatch(
                required_years=required_years,
                candidate_years=None,
                matched=False,
                score=0.0,
            )

        if candidate_years >= required_years:
            return ExperienceYearsMatch(
                required_years=required_years,
                candidate_years=candidate_years,
                matched=True,
                score=1.0,
            )

        proportional_score = (
            candidate_years / required_years
            if required_years > 0
            else 1.0
        )

        return ExperienceYearsMatch(
            required_years=required_years,
            candidate_years=candidate_years,
            matched=False,
            score=round(proportional_score, 4),
        )

    def _calculate_overall_score(
        self,
        *,
        category_definitions: list[CategoryDefinition],
        category_results: dict[str, MatchCategoryResult],
        experience_years: ExperienceYearsMatch,
    ) -> float:
        active_categories = [
            definition
            for definition in category_definitions
            if definition.requirements
        ]

        weighted_values: list[tuple[float, float]] = [
            (
                definition.base_weight,
                category_results[definition.name].score,
            )
            for definition in active_categories
        ]

        if experience_years.required_years is not None:
            weighted_values.append(
                (
                    self.REQUIRED_EXPERIENCE_WEIGHT,
                    (experience_years.score or 0.0) * 100,
                )
            )

        if not weighted_values:
            return 0.0

        total_weight = sum(weight for weight, _ in weighted_values)

        if total_weight <= 0:
            return 0.0

        return sum(
            weight * score
            for weight, score in weighted_values
        ) / total_weight

    async def _ensure_job_exists(
        self,
        job_id: UUID,
    ) -> None:
        job = await self.session.get(Job, job_id)

        if job is None:
            raise MatchingJobNotFoundError(
                f"Job {job_id} not found"
            )

    async def _get_requirements(
        self,
        job_id: UUID,
    ) -> JobRequirements:
        result = await self.session.execute(
            select(JobRequirements).where(
                JobRequirements.job_id == job_id
            )
        )

        requirements = result.scalar_one_or_none()

        if requirements is None:
            raise MatchingRequirementsNotFoundError(
                f"Structured requirements not found for job {job_id}"
            )

        return requirements

    async def _get_profile(
        self,
        profile_id: UUID,
    ) -> CandidateProfile:
        profile = await self.session.get(
            CandidateProfile,
            profile_id,
        )

        if profile is None:
            raise MatchingCandidateProfileNotFoundError(
                f"Candidate profile {profile_id} not found"
            )

        return profile

    async def _get_evidence(
        self,
        profile_id: UUID,
    ) -> list[CandidateEvidence]:
        result = await self.session.execute(
            select(CandidateEvidence)
            .where(
                CandidateEvidence.profile_id == profile_id
            )
            .order_by(
                CandidateEvidence.created_at,
                CandidateEvidence.id,
            )
        )

        return list(result.scalars().all())