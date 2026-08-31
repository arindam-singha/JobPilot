from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.embedding_provider import EmbeddingProvider
from app.models.candidate_evidence import CandidateEvidence
from app.schemas.hybrid_job_candidate_match import (
    HybridCategoryResult,
    HybridJobCandidateMatchRead,
    HybridRequirementMatch,
    HybridSupportingEvidence,
)
from app.schemas.job_candidate_match import (
    JobCandidateMatchRead,
    MatchCategoryResult,
    RequirementMatch,
)
from app.schemas.semantic_evidence import SemanticEvidenceMatch
from app.services.job_candidate_matching_service import (
    JobCandidateMatchingService,
)
from app.services.semantic_evidence_retrieval_service import (
    SemanticEvidenceNotFoundError,
    SemanticEvidenceRetrievalService,
)


class HybridJobCandidateMatchingError(Exception):
    """Base exception for hybrid matching errors."""


class HybridEmbeddedEvidenceNotFoundError(HybridJobCandidateMatchingError):
    """Raised when no compatible embedded evidence exists."""


@dataclass(frozen=True, slots=True)
class HybridCategoryDefinition:
    name: str
    evidence_types: tuple[str, ...]
    base_weight: float


class HybridJobCandidateMatchingService:
    """Combine deterministic and semantic candidate matching."""

    DETERMINISTIC_WEIGHT = 0.60
    SEMANTIC_WEIGHT = 0.40
    MINIMUM_MATCH_SCORE = 0.35

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.session = session
        self.embedding_provider = embedding_provider

        self.deterministic_service = JobCandidateMatchingService(session)

        self.semantic_service = SemanticEvidenceRetrievalService(
            session,
            embedding_provider,
        )

    async def match(
        self,
        *,
        job_id: UUID,
        profile_id: UUID,
    ) -> HybridJobCandidateMatchRead:
        deterministic_result = await self.deterministic_service.match(
            job_id=job_id,
            profile_id=profile_id,
        )

        await self._ensure_compatible_embeddings_exist(profile_id)

        definitions = [
            HybridCategoryDefinition(
                name="required_skills",
                evidence_types=(
                    "skill",
                    "experience",
                    "project",
                    "summary",
                    "general",
                ),
                base_weight=(JobCandidateMatchingService.REQUIRED_SKILLS_WEIGHT),
            ),
            HybridCategoryDefinition(
                name="preferred_skills",
                evidence_types=(
                    "skill",
                    "experience",
                    "project",
                    "summary",
                    "general",
                ),
                base_weight=(JobCandidateMatchingService.PREFERRED_SKILLS_WEIGHT),
            ),
            HybridCategoryDefinition(
                name="required_experience",
                evidence_types=(
                    "experience",
                    "project",
                    "achievement",
                    "summary",
                    "general",
                ),
                base_weight=(JobCandidateMatchingService.REQUIRED_EXPERIENCE_WEIGHT),
            ),
            HybridCategoryDefinition(
                name="education",
                evidence_types=(
                    "education",
                    "summary",
                    "general",
                ),
                base_weight=(JobCandidateMatchingService.EDUCATION_WEIGHT),
            ),
            HybridCategoryDefinition(
                name="certifications",
                evidence_types=(
                    "certification",
                    "skill",
                    "summary",
                    "general",
                ),
                base_weight=(JobCandidateMatchingService.CERTIFICATIONS_WEIGHT),
            ),
        ]

        category_results: dict[str, HybridCategoryResult] = {}

        for definition in definitions:
            deterministic_category = getattr(
                deterministic_result,
                definition.name,
            )

            category_results[definition.name] = await self._build_hybrid_category(
                profile_id=profile_id,
                definition=definition,
                deterministic_category=(deterministic_category),
            )

        overall_score = self._calculate_overall_score(
            definitions=definitions,
            categories=category_results,
            deterministic_result=deterministic_result,
        )

        semantic_overall_score = self._calculate_semantic_overall_score(
            definitions=definitions,
            categories=category_results,
        )

        all_requirements = [
            requirement
            for category in category_results.values()
            for requirement in category.requirements
        ]

        matched_requirements = [
            item.requirement for item in all_requirements if item.matched
        ]

        missing_requirements = [
            item.requirement for item in all_requirements if not item.matched
        ]

        if (
            deterministic_result.experience_years.required_years is not None
            and deterministic_result.experience_years.matched is False
        ):
            missing_requirements.append(
                f"Minimum "
                f"{deterministic_result.experience_years.required_years:g} "
                "years of experience"
            )

        return HybridJobCandidateMatchRead(
            job_id=job_id,
            profile_id=profile_id,
            overall_score=round(overall_score, 2),
            deterministic_overall_score=(deterministic_result.overall_score),
            semantic_overall_score=round(
                semantic_overall_score,
                2,
            ),
            required_skills=category_results["required_skills"],
            preferred_skills=category_results["preferred_skills"],
            required_experience=category_results["required_experience"],
            education=category_results["education"],
            certifications=category_results["certifications"],
            experience_years=(deterministic_result.experience_years),
            matched_requirements=matched_requirements,
            missing_requirements=missing_requirements,
            evidence_considered=(deterministic_result.evidence_considered),
            embedding_provider=(self.embedding_provider.provider_name),
            embedding_model=(self.embedding_provider.model_name),
            deterministic_weight=(self.DETERMINISTIC_WEIGHT),
            semantic_weight=self.SEMANTIC_WEIGHT,
        )

    async def _build_hybrid_category(
        self,
        *,
        profile_id: UUID,
        definition: HybridCategoryDefinition,
        deterministic_category: MatchCategoryResult,
    ) -> HybridCategoryResult:
        hybrid_requirements: list[HybridRequirementMatch] = []

        for deterministic_requirement in deterministic_category.requirements:
            semantic_matches = await self._semantic_matches(
                profile_id=profile_id,
                requirement=(deterministic_requirement.requirement),
                evidence_types=definition.evidence_types,
            )

            hybrid_requirements.append(
                self._combine_requirement_scores(
                    deterministic=(deterministic_requirement),
                    semantic_matches=semantic_matches,
                )
            )

        if hybrid_requirements:
            category_score = (
                sum(item.hybrid_score for item in hybrid_requirements)
                / len(hybrid_requirements)
                * 100
            )
        else:
            category_score = 0.0

        return HybridCategoryResult(
            category=definition.name,
            score=round(category_score, 2),
            total_requirements=len(hybrid_requirements),
            matched_requirements=sum(1 for item in hybrid_requirements if item.matched),
            requirements=hybrid_requirements,
        )

    async def _semantic_matches(
        self,
        *,
        profile_id: UUID,
        requirement: str,
        evidence_types: tuple[str, ...],
    ) -> list[SemanticEvidenceMatch]:
        try:
            result = await self.semantic_service.search(
                profile_id=profile_id,
                query=requirement,
                top_k=5,
                minimum_similarity=0.0,
                evidence_types=evidence_types,
            )
        except SemanticEvidenceNotFoundError:
            return []

        return result.matches

    def _combine_requirement_scores(
        self,
        *,
        deterministic: RequirementMatch,
        semantic_matches: list[SemanticEvidenceMatch],
    ) -> HybridRequirementMatch:
        semantic_by_id = {match.evidence_id: match for match in semantic_matches}

        deterministic_by_id = {
            item.evidence_id: item for item in deterministic.supporting_evidence
        }

        all_ids = set(semantic_by_id) | set(deterministic_by_id)

        supporting: list[HybridSupportingEvidence] = []

        for evidence_id in all_ids:
            semantic = semantic_by_id.get(evidence_id)
            deterministic_evidence = deterministic_by_id.get(evidence_id)

            deterministic_score = (
                deterministic_evidence.score
                if deterministic_evidence is not None
                else 0.0
            )

            semantic_score = semantic.similarity if semantic is not None else 0.0

            hybrid_score = self._hybrid_score(
                deterministic_score,
                semantic_score,
            )

            if deterministic_evidence is not None:
                evidence_type = deterministic_evidence.evidence_type
                title = deterministic_evidence.title
                source_type = deterministic_evidence.source_type
            else:
                assert semantic is not None
                evidence_type = semantic.evidence_type
                title = semantic.title
                source_type = semantic.source_type

            supporting.append(
                HybridSupportingEvidence(
                    evidence_id=evidence_id,
                    evidence_type=evidence_type,
                    title=title,
                    source_type=source_type,
                    deterministic_score=round(
                        deterministic_score,
                        6,
                    ),
                    semantic_score=round(
                        semantic_score,
                        6,
                    ),
                    hybrid_score=round(
                        hybrid_score,
                        6,
                    ),
                )
            )

        supporting.sort(
            key=lambda item: (
                -item.hybrid_score,
                -item.semantic_score,
                item.title.casefold(),
                str(item.evidence_id),
            )
        )

        semantic_score = max(
            (item.similarity for item in semantic_matches),
            default=0.0,
        )

        hybrid_score = self._hybrid_score(
            deterministic.score,
            semantic_score,
        )

        return HybridRequirementMatch(
            requirement=deterministic.requirement,
            matched=(
                max(
                    deterministic.score,
                    semantic_score,
                    hybrid_score,
                )
                >= self.MINIMUM_MATCH_SCORE
            ),
            deterministic_score=round(
                deterministic.score,
                6,
            ),
            semantic_score=round(
                semantic_score,
                6,
            ),
            hybrid_score=round(
                hybrid_score,
                6,
            ),
            supporting_evidence=supporting[:5],
        )

    def _calculate_overall_score(
        self,
        *,
        definitions: list[HybridCategoryDefinition],
        categories: dict[str, HybridCategoryResult],
        deterministic_result: JobCandidateMatchRead,
    ) -> float:
        weighted_values: list[tuple[float, float]] = []

        for definition in definitions:
            category = categories[definition.name]

            if category.total_requirements == 0:
                continue

            weighted_values.append(
                (
                    definition.base_weight,
                    category.score,
                )
            )

        experience_years = deterministic_result.experience_years

        if experience_years.required_years is not None:
            weighted_values.append(
                (
                    JobCandidateMatchingService.REQUIRED_EXPERIENCE_WEIGHT,
                    (experience_years.score or 0.0) * 100,
                )
            )

        if not weighted_values:
            return 0.0

        total_weight = sum(weight for weight, _ in weighted_values)

        return sum(weight * score for weight, score in weighted_values) / total_weight

    def _calculate_semantic_overall_score(
        self,
        *,
        definitions: list[HybridCategoryDefinition],
        categories: dict[str, HybridCategoryResult],
    ) -> float:
        values: list[tuple[float, float]] = []

        for definition in definitions:
            category = categories[definition.name]

            if not category.requirements:
                continue

            category_semantic_score = (
                sum(item.semantic_score for item in category.requirements)
                / len(category.requirements)
                * 100
            )

            values.append(
                (
                    definition.base_weight,
                    category_semantic_score,
                )
            )

        if not values:
            return 0.0

        total_weight = sum(weight for weight, _ in values)

        return sum(weight * score for weight, score in values) / total_weight

    @classmethod
    def _hybrid_score(
        cls,
        deterministic_score: float,
        semantic_score: float,
    ) -> float:
        return (
            deterministic_score * cls.DETERMINISTIC_WEIGHT
            + semantic_score * cls.SEMANTIC_WEIGHT
        )

    async def _ensure_compatible_embeddings_exist(
        self,
        profile_id: UUID,
    ) -> None:
        result = await self.session.execute(
            select(CandidateEvidence.id)
            .where(
                CandidateEvidence.profile_id == profile_id,
                CandidateEvidence.embedding.is_not(None),
                CandidateEvidence.embedding_provider
                == self.embedding_provider.provider_name,
                CandidateEvidence.embedding_model == self.embedding_provider.model_name,
            )
            .limit(1)
        )

        if result.scalar_one_or_none() is None:
            raise HybridEmbeddedEvidenceNotFoundError(
                f"Candidate profile {profile_id} has no " "compatible embedded evidence"
            )
