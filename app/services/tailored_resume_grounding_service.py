from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.embedding_provider import EmbeddingProvider
from app.models.candidate_evidence import CandidateEvidence
from app.schemas.hybrid_job_candidate_match import (
    HybridJobCandidateMatchRead,
    HybridSupportingEvidence,
)
from app.schemas.tailored_resume import (
    ResumeGroundingBundle,
    ResumeRequirementTrace,
    SelectedResumeEvidence,
)
from app.services.hybrid_job_candidate_matching_service import (
    HybridJobCandidateMatchingService,
)


class TailoredResumeGroundingError(Exception):
    """Base exception for tailored-resume grounding failures."""


class TailoredResumeEvidenceNotFoundError(TailoredResumeGroundingError):
    """Raised when hybrid matching yields no sufficiently relevant evidence."""


class TailoredResumeEvidenceIntegrityError(TailoredResumeGroundingError):
    """Raised when hybrid provenance cannot be resolved to stored evidence."""


@dataclass(frozen=True, slots=True)
class _EvidenceCandidate:
    evidence_id: UUID
    trace: ResumeRequirementTrace


class TailoredResumeGroundingService:
    """Build a factual evidence bundle from the authoritative hybrid match."""

    DEFAULT_MINIMUM_EVIDENCE_SCORE = 0.35
    DEFAULT_MAX_EVIDENCE_PER_REQUIREMENT = 3
    DEFAULT_MAX_TOTAL_EVIDENCE = 12
    CATEGORY_NAMES = (
        "required_skills",
        "preferred_skills",
        "required_experience",
        "education",
        "certifications",
    )

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        *,
        hybrid_matching_service: HybridJobCandidateMatchingService | None = None,
    ) -> None:
        self.session = session
        self.embedding_provider = embedding_provider
        self.hybrid_matching_service = hybrid_matching_service or (
            HybridJobCandidateMatchingService(session, embedding_provider)
        )

    async def build_bundle(
        self,
        *,
        job_id: UUID,
        profile_id: UUID,
        minimum_evidence_score: float = DEFAULT_MINIMUM_EVIDENCE_SCORE,
        max_evidence_per_requirement: int = DEFAULT_MAX_EVIDENCE_PER_REQUIREMENT,
        max_total_evidence: int = DEFAULT_MAX_TOTAL_EVIDENCE,
    ) -> ResumeGroundingBundle:
        self._validate_selection_arguments(
            minimum_evidence_score=minimum_evidence_score,
            max_evidence_per_requirement=max_evidence_per_requirement,
            max_total_evidence=max_total_evidence,
        )

        hybrid_result = await self.hybrid_matching_service.match(
            job_id=job_id,
            profile_id=profile_id,
        )
        candidates = self._collect_candidates(
            hybrid_result,
            minimum_evidence_score=minimum_evidence_score,
            max_evidence_per_requirement=max_evidence_per_requirement,
        )
        selected_ids, traces_by_id = self._rank_and_limit_candidates(
            candidates,
            max_total_evidence=max_total_evidence,
        )

        if not selected_ids:
            raise TailoredResumeEvidenceNotFoundError(
                "Hybrid matching found no evidence above the resume selection threshold"
            )

        evidence_by_id = await self._load_evidence(
            profile_id=profile_id,
            evidence_ids=selected_ids,
        )
        missing_ids = set(selected_ids) - set(evidence_by_id)
        if missing_ids:
            raise TailoredResumeEvidenceIntegrityError(
                "Hybrid matching referenced missing or cross-profile candidate evidence"
            )

        selected_evidence = [
            self._to_selected_evidence(
                evidence=evidence_by_id[evidence_id],
                traces=traces_by_id[evidence_id],
            )
            for evidence_id in selected_ids
        ]

        return ResumeGroundingBundle(
            job_id=hybrid_result.job_id,
            profile_id=hybrid_result.profile_id,
            hybrid_overall_score=hybrid_result.overall_score,
            matched_requirements=hybrid_result.matched_requirements,
            missing_requirements=hybrid_result.missing_requirements,
            selected_evidence=selected_evidence,
            embedding_provider=hybrid_result.embedding_provider,
            embedding_model=hybrid_result.embedding_model,
            minimum_evidence_score=minimum_evidence_score,
            max_evidence_per_requirement=max_evidence_per_requirement,
            max_total_evidence=max_total_evidence,
        )

    def _collect_candidates(
        self,
        result: HybridJobCandidateMatchRead,
        *,
        minimum_evidence_score: float,
        max_evidence_per_requirement: int,
    ) -> list[_EvidenceCandidate]:
        candidates: list[_EvidenceCandidate] = []

        for category_name in self.CATEGORY_NAMES:
            category = getattr(result, category_name)
            for requirement in category.requirements:
                if not requirement.matched:
                    continue

                eligible = sorted(
                    (
                        evidence
                        for evidence in requirement.supporting_evidence
                        if self._evidence_selection_score(evidence)
                        >= minimum_evidence_score
                    ),
                    key=lambda evidence: (
                        -self._evidence_selection_score(evidence),
                        evidence.title.casefold(),
                        str(evidence.evidence_id),
                    ),
                )[:max_evidence_per_requirement]

                for rank, evidence in enumerate(eligible, start=1):
                    candidates.append(
                        _EvidenceCandidate(
                            evidence_id=evidence.evidence_id,
                            trace=ResumeRequirementTrace(
                                category=category_name,
                                requirement=requirement.requirement,
                                deterministic_score=evidence.deterministic_score,
                                semantic_score=evidence.semantic_score,
                                hybrid_score=evidence.hybrid_score,
                                rank=rank,
                            ),
                        )
                    )

        return candidates

    @staticmethod
    def _rank_and_limit_candidates(
        candidates: list[_EvidenceCandidate],
        *,
        max_total_evidence: int,
    ) -> tuple[list[UUID], dict[UUID, list[ResumeRequirementTrace]]]:
        traces_by_id: dict[UUID, list[ResumeRequirementTrace]] = defaultdict(list)
        for candidate in candidates:
            traces_by_id[candidate.evidence_id].append(candidate.trace)

        for traces in traces_by_id.values():
            traces.sort(
                key=lambda trace: (
                    -TailoredResumeGroundingService._trace_selection_score(trace),
                    trace.category,
                    trace.requirement.casefold(),
                    trace.rank,
                )
            )

        selected_ids = sorted(
            traces_by_id,
            key=lambda evidence_id: (
                -max(
                    TailoredResumeGroundingService._trace_selection_score(trace)
                    for trace in traces_by_id[evidence_id]
                ),
                -len(traces_by_id[evidence_id]),
                str(evidence_id),
            ),
        )[:max_total_evidence]

        return selected_ids, traces_by_id

    async def _load_evidence(
        self,
        *,
        profile_id: UUID,
        evidence_ids: list[UUID],
    ) -> dict[UUID, CandidateEvidence]:
        statement = select(CandidateEvidence).where(
            CandidateEvidence.profile_id == profile_id,
            CandidateEvidence.id.in_(evidence_ids),
        )
        result = await self.session.execute(statement)
        return {evidence.id: evidence for evidence in result.scalars().all()}

    @staticmethod
    def _to_selected_evidence(
        *,
        evidence: CandidateEvidence,
        traces: list[ResumeRequirementTrace],
    ) -> SelectedResumeEvidence:
        return SelectedResumeEvidence(
            evidence_id=evidence.id,
            profile_id=evidence.profile_id,
            evidence_type=evidence.evidence_type,
            title=evidence.title,
            content=evidence.content,
            source_type=evidence.source_type,
            source_id=evidence.source_id,
            metadata_json=evidence.metadata_json,
            selection_score=max(
                TailoredResumeGroundingService._trace_selection_score(trace)
                for trace in traces
            ),
            requirement_traces=traces,
        )

    @staticmethod
    def _evidence_selection_score(
        evidence: HybridSupportingEvidence,
    ) -> float:
        return max(
            evidence.deterministic_score,
            evidence.semantic_score,
            evidence.hybrid_score,
        )

    @staticmethod
    def _trace_selection_score(
        trace: ResumeRequirementTrace,
    ) -> float:
        return max(
            trace.deterministic_score,
            trace.semantic_score,
            trace.hybrid_score,
        )

    @staticmethod
    def _validate_selection_arguments(
        *,
        minimum_evidence_score: float,
        max_evidence_per_requirement: int,
        max_total_evidence: int,
    ) -> None:
        if not 0.0 <= minimum_evidence_score <= 1.0:
            raise ValueError("minimum_evidence_score must be between 0 and 1")
        if max_evidence_per_requirement < 1:
            raise ValueError("max_evidence_per_requirement must be at least 1")
        if max_total_evidence < 1:
            raise ValueError("max_total_evidence must be at least 1")
