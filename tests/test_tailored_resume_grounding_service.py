from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from app.models.candidate_evidence import CandidateEvidence
from app.schemas.hybrid_job_candidate_match import (
    HybridCategoryResult,
    HybridJobCandidateMatchRead,
    HybridRequirementMatch,
    HybridSupportingEvidence,
)
from app.schemas.job_candidate_match import ExperienceYearsMatch
from app.services.tailored_resume_grounding_service import (
    TailoredResumeEvidenceIntegrityError,
    TailoredResumeEvidenceNotFoundError,
    TailoredResumeGroundingService,
)


@pytest.fixture(autouse=True)
def clean_database() -> None:
    """Override the suite-wide database fixture for stubbed service tests."""


class StubHybridService:
    def __init__(self, result: HybridJobCandidateMatchRead) -> None:
        self.result = result
        self.calls: list[tuple[UUID, UUID]] = []

    async def match(self, *, job_id: UUID, profile_id: UUID):
        self.calls.append((job_id, profile_id))
        return self.result


class StubScalars:
    def __init__(self, evidence: list[CandidateEvidence]) -> None:
        self.evidence = evidence

    def all(self) -> list[CandidateEvidence]:
        return self.evidence


class StubResult:
    def __init__(self, evidence: list[CandidateEvidence]) -> None:
        self.evidence = evidence

    def scalars(self) -> StubScalars:
        return StubScalars(self.evidence)


class StubSession:
    def __init__(self, evidence: list[CandidateEvidence]) -> None:
        self.evidence = evidence

    async def execute(self, statement):
        return StubResult(self.evidence)


class StubEmbeddingProvider:
    provider_name = "fake"
    model_name = "fake-model"
    dimensions = 768

    async def embed_text(self, text: str) -> list[float]:
        raise AssertionError("grounding service must delegate matching")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("grounding service must delegate matching")


def _supporting(
    evidence_id: UUID,
    *,
    score: float = 0.0,
    title: str,
    deterministic_score: float | None = None,
    semantic_score: float | None = None,
    hybrid_score: float | None = None,
) -> HybridSupportingEvidence:
    return HybridSupportingEvidence(
        evidence_id=evidence_id,
        evidence_type="project",
        title=title,
        source_type="candidate_project",
        deterministic_score=(
            score if deterministic_score is None else deterministic_score
        ),
        semantic_score=score if semantic_score is None else semantic_score,
        hybrid_score=score if hybrid_score is None else hybrid_score,
    )


def _requirement(
    name: str,
    evidence: list[HybridSupportingEvidence],
    *,
    matched: bool = True,
) -> HybridRequirementMatch:
    score = max((item.hybrid_score for item in evidence), default=0.0)
    return HybridRequirementMatch(
        requirement=name,
        matched=matched,
        deterministic_score=score,
        semantic_score=score,
        hybrid_score=score,
        supporting_evidence=evidence,
    )


def _category(
    name: str,
    requirements: list[HybridRequirementMatch] | None = None,
) -> HybridCategoryResult:
    items = requirements or []
    return HybridCategoryResult(
        category=name,
        score=100 if items else 0,
        total_requirements=len(items),
        matched_requirements=sum(item.matched for item in items),
        requirements=items,
    )


def _match_result(
    *,
    job_id: UUID,
    profile_id: UUID,
    required_skills: list[HybridRequirementMatch],
    required_experience: list[HybridRequirementMatch] | None = None,
) -> HybridJobCandidateMatchRead:
    matched = [
        item.requirement
        for item in [*required_skills, *(required_experience or [])]
        if item.matched
    ]
    return HybridJobCandidateMatchRead(
        job_id=job_id,
        profile_id=profile_id,
        overall_score=88,
        deterministic_overall_score=85,
        semantic_overall_score=92,
        required_skills=_category("required_skills", required_skills),
        preferred_skills=_category("preferred_skills"),
        required_experience=_category("required_experience", required_experience),
        education=_category("education"),
        certifications=_category("certifications"),
        experience_years=ExperienceYearsMatch(),
        matched_requirements=matched,
        missing_requirements=[],
        evidence_considered=5,
        embedding_provider="fake",
        embedding_model="fake-model",
    )


def _stored_evidence(
    *,
    evidence_id: UUID,
    profile_id: UUID,
    title: str,
) -> CandidateEvidence:
    return CandidateEvidence(
        id=evidence_id,
        profile_id=profile_id,
        evidence_type="project",
        title=title,
        content=f"Factual content for {title}",
        source_type="candidate_project",
        source_id=uuid4(),
        metadata_json='{"origin":"profile"}',
    )


def _service(
    result: HybridJobCandidateMatchRead,
    evidence: list[CandidateEvidence],
) -> tuple[TailoredResumeGroundingService, StubHybridService]:
    hybrid = StubHybridService(result)
    service = TailoredResumeGroundingService(
        StubSession(evidence),  # type: ignore[arg-type]
        StubEmbeddingProvider(),
        hybrid_matching_service=hybrid,  # type: ignore[arg-type]
    )
    return service, hybrid


@pytest.mark.asyncio
async def test_build_bundle_uses_hybrid_result_and_stored_content() -> None:
    job_id = uuid4()
    profile_id = uuid4()
    evidence_id = uuid4()
    match = _match_result(
        job_id=job_id,
        profile_id=profile_id,
        required_skills=[
            _requirement(
                "Python", [_supporting(evidence_id, score=0.9, title="Python")]
            )
        ],
    )
    stored = _stored_evidence(
        evidence_id=evidence_id,
        profile_id=profile_id,
        title="Authoritative Python project",
    )
    service, hybrid = _service(match, [stored])

    bundle = await service.build_bundle(job_id=job_id, profile_id=profile_id)

    assert hybrid.calls == [(job_id, profile_id)]
    assert bundle.selected_evidence[0].content == stored.content
    assert bundle.selected_evidence[0].source_id == stored.source_id
    assert bundle.selected_evidence[0].selection_score == 0.9
    assert bundle.selected_evidence[0].requirement_traces[0].requirement == "Python"


@pytest.mark.asyncio
async def test_selector_excludes_unmatched_and_below_threshold_evidence() -> None:
    job_id = uuid4()
    profile_id = uuid4()
    selected_id = uuid4()
    low_id = uuid4()
    unmatched_id = uuid4()
    match = _match_result(
        job_id=job_id,
        profile_id=profile_id,
        required_skills=[
            _requirement(
                "Python",
                [
                    _supporting(selected_id, score=0.95, title="Selected"),
                    _supporting(low_id, score=0.34, title="Low"),
                ],
            ),
            _requirement(
                "Kubernetes",
                [_supporting(unmatched_id, score=0.99, title="Unmatched")],
                matched=False,
            ),
        ],
    )
    stored = _stored_evidence(
        evidence_id=selected_id,
        profile_id=profile_id,
        title="Selected",
    )
    service, _ = _service(match, [stored])

    bundle = await service.build_bundle(job_id=job_id, profile_id=profile_id)

    assert [item.evidence_id for item in bundle.selected_evidence] == [selected_id]


@pytest.mark.asyncio
async def test_selector_accepts_semantic_only_evidence_at_threshold() -> None:
    job_id = uuid4()
    profile_id = uuid4()
    evidence_id = uuid4()
    match = _match_result(
        job_id=job_id,
        profile_id=profile_id,
        required_skills=[
            _requirement(
                "C++ and Python",
                [
                    _supporting(
                        evidence_id,
                        title="Programming",
                        deterministic_score=0.0,
                        semantic_score=0.65,
                        hybrid_score=0.26,
                    )
                ],
            )
        ],
    )
    stored = _stored_evidence(
        evidence_id=evidence_id,
        profile_id=profile_id,
        title="Programming",
    )
    service, _ = _service(match, [stored])

    bundle = await service.build_bundle(
        job_id=job_id,
        profile_id=profile_id,
    )

    assert [item.evidence_id for item in bundle.selected_evidence] == [evidence_id]
    assert bundle.selected_evidence[0].selection_score == 0.65
    assert bundle.minimum_evidence_score == 0.35


@pytest.mark.asyncio
async def test_selector_deduplicates_and_preserves_all_requirement_traces() -> None:
    job_id = uuid4()
    profile_id = uuid4()
    evidence_id = uuid4()
    match = _match_result(
        job_id=job_id,
        profile_id=profile_id,
        required_skills=[
            _requirement(
                "Python", [_supporting(evidence_id, score=0.9, title="Project")]
            ),
            _requirement(
                "FastAPI", [_supporting(evidence_id, score=0.8, title="Project")]
            ),
        ],
    )
    stored = _stored_evidence(
        evidence_id=evidence_id,
        profile_id=profile_id,
        title="Project",
    )
    service, _ = _service(match, [stored])

    bundle = await service.build_bundle(job_id=job_id, profile_id=profile_id)

    assert len(bundle.selected_evidence) == 1
    assert [
        trace.requirement for trace in bundle.selected_evidence[0].requirement_traces
    ] == [
        "Python",
        "FastAPI",
    ]


@pytest.mark.asyncio
async def test_selector_enforces_global_highest_score_limit() -> None:
    job_id = uuid4()
    profile_id = uuid4()
    ids = [uuid4(), uuid4(), uuid4()]
    scores = [0.7, 0.95, 0.8]
    match = _match_result(
        job_id=job_id,
        profile_id=profile_id,
        required_skills=[
            _requirement(
                "Python",
                [
                    _supporting(evidence_id, score=score, title=str(index))
                    for index, (evidence_id, score) in enumerate(
                        zip(ids, scores, strict=False)
                    )
                ],
            )
        ],
    )
    stored = [
        _stored_evidence(
            evidence_id=evidence_id, profile_id=profile_id, title=str(index)
        )
        for index, evidence_id in enumerate(ids)
    ]
    service, _ = _service(match, stored)

    bundle = await service.build_bundle(
        job_id=job_id,
        profile_id=profile_id,
        max_total_evidence=2,
    )

    assert [item.selection_score for item in bundle.selected_evidence] == [0.95, 0.8]


@pytest.mark.asyncio
async def test_no_relevant_evidence_is_rejected() -> None:
    job_id = uuid4()
    profile_id = uuid4()
    match = _match_result(
        job_id=job_id,
        profile_id=profile_id,
        required_skills=[
            _requirement(
                "Python",
                [_supporting(uuid4(), score=0.5, title="Too low")],
                matched=False,
            )
        ],
    )
    service, _ = _service(match, [])

    with pytest.raises(TailoredResumeEvidenceNotFoundError):
        await service.build_bundle(job_id=job_id, profile_id=profile_id)


@pytest.mark.asyncio
async def test_missing_stored_evidence_is_integrity_error() -> None:
    job_id = uuid4()
    profile_id = uuid4()
    match = _match_result(
        job_id=job_id,
        profile_id=profile_id,
        required_skills=[
            _requirement(
                "Python",
                [_supporting(uuid4(), score=0.9, title="Missing")],
            )
        ],
    )
    service, _ = _service(match, [])

    with pytest.raises(TailoredResumeEvidenceIntegrityError):
        await service.build_bundle(job_id=job_id, profile_id=profile_id)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minimum_evidence_score": -0.1}, "between 0 and 1"),
        ({"minimum_evidence_score": 1.1}, "between 0 and 1"),
        ({"max_evidence_per_requirement": 0}, "at least 1"),
        ({"max_total_evidence": 0}, "at least 1"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_selection_arguments_are_rejected(kwargs, message) -> None:
    job_id = uuid4()
    profile_id = uuid4()
    service, hybrid = _service(
        _match_result(job_id=job_id, profile_id=profile_id, required_skills=[]),
        [],
    )

    with pytest.raises(ValueError, match=message):
        await service.build_bundle(job_id=job_id, profile_id=profile_id, **kwargs)

    assert hybrid.calls == []
