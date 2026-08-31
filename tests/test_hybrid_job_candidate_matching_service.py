from __future__ import annotations

import math
from uuid import uuid4

import pytest

from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import CandidateProfile
from app.models.job import Job
from app.models.job_requirements import JobRequirements
from app.services.hybrid_job_candidate_matching_service import (
    HybridEmbeddedEvidenceNotFoundError,
    HybridJobCandidateMatchingService,
)


DIMENSIONS = 768


def _vector(
    first: float,
    second: float = 0.0,
) -> list[float]:
    vector = [0.0] * DIMENSIONS
    vector[0] = first
    vector[1] = second
    return vector


class FixedEmbeddingProvider:
    def __init__(
        self,
        vector: list[float],
    ) -> None:
        self._vector = vector

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "hybrid-test-model"

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:
        return list(self._vector)

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


async def _create_job(
    database_session,
) -> Job:
    job = Job(
        title="Senior Robotics Engineer",
        company="Example Robotics",
        location="Abu Dhabi",
        job_url=f"https://example.com/{uuid4()}",
        description="Robotics role",
        source="manual",
    )

    database_session.add(job)
    await database_session.commit()
    await database_session.refresh(job)

    return job


async def _create_profile(
    database_session,
    *,
    years: float | None = 8,
) -> CandidateProfile:
    profile = CandidateProfile(
        full_name="Hybrid Candidate",
        total_experience_years=years,
    )

    database_session.add(profile)
    await database_session.commit()
    await database_session.refresh(profile)

    return profile


async def _create_requirements(
    database_session,
    *,
    job_id,
    required_skills=None,
    required_experience=None,
    minimum_experience_years=None,
) -> JobRequirements:
    requirements = JobRequirements(
        job_id=job_id,
        required_skills=required_skills or [],
        preferred_skills=[],
        required_experience=(required_experience or []),
        responsibilities=[],
        education_requirements=[],
        certifications=[],
        domain_keywords=[],
        minimum_experience_years=(minimum_experience_years),
        provider="fake",
        model_name="fake-model",
        extraction_metadata={},
    )

    database_session.add(requirements)
    await database_session.commit()
    await database_session.refresh(requirements)

    return requirements


async def _create_evidence(
    database_session,
    *,
    profile_id,
    title: str,
    content: str,
    embedding: list[float] | None,
    evidence_type: str = "skill",
) -> CandidateEvidence:
    evidence = CandidateEvidence(
        profile_id=profile_id,
        evidence_type=evidence_type,
        title=title,
        content=content,
        source_type="manual",
        source_id=None,
        metadata_json=None,
        embedding=embedding,
        embedding_provider=("fake" if embedding is not None else None),
        embedding_model=("hybrid-test-model" if embedding is not None else None),
    )

    database_session.add(evidence)
    await database_session.commit()
    await database_session.refresh(evidence)

    return evidence


@pytest.mark.asyncio
async def test_exact_and_semantic_match_scores_one(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Python"],
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Python",
        content="Python development",
        embedding=_vector(1.0),
    )

    service = HybridJobCandidateMatchingService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    )

    result = await service.match(
        job_id=job.id,
        profile_id=profile.id,
    )

    requirement = result.required_skills.requirements[0]

    assert requirement.deterministic_score == 1.0
    assert requirement.semantic_score == 1.0
    assert requirement.hybrid_score == 1.0
    assert requirement.matched is True
    assert result.overall_score == 100


@pytest.mark.asyncio
async def test_semantic_match_can_support_non_exact_wording(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Visual inspection automation"],
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Defect detection",
        content=("Built an automated manufacturing " "defect-detection platform."),
        embedding=_vector(1.0),
        evidence_type="project",
    )

    result = await HybridJobCandidateMatchingService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    ).match(
        job_id=job.id,
        profile_id=profile.id,
    )

    requirement = result.required_skills.requirements[0]

    assert requirement.deterministic_score == 0.0
    assert requirement.semantic_score == 1.0
    assert requirement.hybrid_score == 0.4
    assert requirement.matched is True


@pytest.mark.asyncio
async def test_semantic_score_below_threshold_is_not_matched(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Visual inspection automation"],
    )

    semantic_similarity = 0.34
    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Unrelated evidence",
        content="Unrelated candidate evidence",
        embedding=_vector(
            semantic_similarity,
            math.sqrt(1 - semantic_similarity**2),
        ),
        evidence_type="project",
    )

    result = await HybridJobCandidateMatchingService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    ).match(
        job_id=job.id,
        profile_id=profile.id,
    )

    requirement = result.required_skills.requirements[0]

    assert requirement.deterministic_score == 0.0
    assert requirement.semantic_score == pytest.approx(semantic_similarity)
    assert requirement.hybrid_score == pytest.approx(0.136)
    assert requirement.matched is False


@pytest.mark.asyncio
async def test_combined_scores_use_sixty_forty_weights(
    database_session,
) -> None:
    service = HybridJobCandidateMatchingService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    )

    score = service._hybrid_score(
        deterministic_score=0.8,
        semantic_score=0.5,
    )

    assert score == pytest.approx(0.68)


@pytest.mark.asyncio
async def test_hybrid_supporting_evidence_contains_scores(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Python"],
    )

    evidence = await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Python evidence",
        content="Python",
        embedding=_vector(1.0),
    )

    result = await HybridJobCandidateMatchingService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    ).match(
        job_id=job.id,
        profile_id=profile.id,
    )

    supporting = result.required_skills.requirements[0].supporting_evidence[0]

    assert supporting.evidence_id == evidence.id
    assert supporting.deterministic_score == 1.0
    assert supporting.semantic_score == 1.0
    assert supporting.hybrid_score == 1.0


@pytest.mark.asyncio
async def test_experience_years_remain_part_of_score(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(
        database_session,
        years=3,
    )

    await _create_requirements(
        database_session,
        job_id=job.id,
        minimum_experience_years=6,
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Experience",
        content="Engineering experience",
        embedding=_vector(1.0),
        evidence_type="experience",
    )

    result = await HybridJobCandidateMatchingService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    ).match(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert result.experience_years.score == 0.5
    assert result.overall_score == 50
    assert "Minimum 6 years of experience" in result.missing_requirements


@pytest.mark.asyncio
async def test_missing_compatible_embeddings_is_rejected(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Python"],
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Python",
        content="Python",
        embedding=None,
    )

    service = HybridJobCandidateMatchingService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    )

    with pytest.raises(HybridEmbeddedEvidenceNotFoundError):
        await service.match(
            job_id=job.id,
            profile_id=profile.id,
        )


def test_hybrid_score_calculation() -> None:
    assert HybridJobCandidateMatchingService._hybrid_score(
        1.0,
        0.0,
    ) == pytest.approx(0.6)

    assert HybridJobCandidateMatchingService._hybrid_score(
        0.0,
        1.0,
    ) == pytest.approx(0.4)

    assert HybridJobCandidateMatchingService._hybrid_score(
        1.0,
        1.0,
    ) == pytest.approx(1.0)
