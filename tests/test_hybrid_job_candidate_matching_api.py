from __future__ import annotations

from uuid import uuid4

import pytest
from app.embeddings.embedding_provider_factory import (
    EmbeddingProviderConfigurationError,
)
from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import CandidateProfile
from app.models.job import Job
from app.models.job_requirements import JobRequirements

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
        vector: list[float] | None = None,
    ) -> None:
        self._vector = vector or _vector(1.0)

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-embedding-model"

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
        return [
            list(self._vector)
            for _ in texts
        ]
    
async def _create_job(
    database_session,
) -> Job:
    job = Job(
        title="Senior Robotics Engineer",
        company="Example Robotics",
        location="Abu Dhabi",
        job_url=f"https://example.com/jobs/{uuid4()}",
        description="Robotics engineering role.",
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
        full_name="Hybrid API Candidate",
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
    preferred_skills=None,
    required_experience=None,
    education_requirements=None,
    certifications=None,
    minimum_experience_years=None,
) -> JobRequirements:
    requirements = JobRequirements(
        job_id=job_id,
        required_skills=required_skills or [],
        preferred_skills=preferred_skills or [],
        required_experience=required_experience or [],
        responsibilities=[],
        education_requirements=education_requirements or [],
        certifications=certifications or [],
        domain_keywords=[],
        minimum_experience_years=minimum_experience_years,
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
    evidence_type: str,
    title: str,
    content: str,
    embedding: list[float] | None,
    provider: str | None = "fake",
    model: str | None = "fake-embedding-model",
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
        embedding_provider=provider if embedding is not None else None,
        embedding_model=model if embedding is not None else None,
    )

    database_session.add(evidence)
    await database_session.commit()
    await database_session.refresh(evidence)

    return evidence


def _match_url(job_id, profile_id) -> str:
    return (
        f"/api/v1/jobs/{job_id}"
        f"/hybrid-match/{profile_id}"
    )


# def _configure_fake_embeddings(
#     monkeypatch,
# ) -> None:
#     monkeypatch.setenv(
#         "EMBEDDING_PROVIDER",
#         "fake",
#     )
#     monkeypatch.setenv(
#         "OLLAMA_EMBEDDING_DIMENSIONS",
#         str(DIMENSIONS),
#     )

def _configure_fake_embeddings(
    monkeypatch,
    *,
    query_vector: list[float] | None = None,
) -> None:
    provider = FixedEmbeddingProvider(
        query_vector or _vector(1.0)
    )

    monkeypatch.setattr(
        "app.api.routes.hybrid_job_candidate_matching."
        "create_embedding_provider",
        lambda: provider,
    )

@pytest.mark.asyncio
async def test_hybrid_matching_endpoint_returns_200(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    _configure_fake_embeddings(monkeypatch)

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
        evidence_type="skill",
        title="Python",
        content="Python development",
        embedding=_vector(1.0),
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 200

    body = response.json()

    assert body["job_id"] == str(job.id)
    assert body["profile_id"] == str(profile.id)
    assert body["overall_score"] == 100
    assert body["deterministic_overall_score"] == 100
    assert body["semantic_overall_score"] == 100
    assert body["embedding_provider"] == "fake"
    assert body["embedding_model"] == "fake-embedding-model"
    assert body["deterministic_weight"] == 0.6
    assert body["semantic_weight"] == 0.4


@pytest.mark.asyncio
async def test_hybrid_response_contains_requirement_scores(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    _configure_fake_embeddings(monkeypatch)

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
        evidence_type="skill",
        title="Python skill",
        content="Python",
        embedding=_vector(1.0),
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 200

    requirement = (
        response.json()["required_skills"]["requirements"][0]
    )

    assert requirement["matched"] is True
    assert requirement["deterministic_score"] == 1.0
    assert requirement["semantic_score"] == 1.0
    assert requirement["hybrid_score"] == 1.0

    supporting = requirement["supporting_evidence"][0]

    assert supporting["evidence_id"] == str(evidence.id)
    assert supporting["deterministic_score"] == 1.0
    assert supporting["semantic_score"] == 1.0
    assert supporting["hybrid_score"] == 1.0


@pytest.mark.asyncio
async def test_hybrid_response_contains_missing_requirement(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    _configure_fake_embeddings(monkeypatch)

    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Kubernetes"],
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="skill",
        title="Python",
        content="Python",
        embedding=_vector(0.0, 1.0),
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 200

    body = response.json()

    assert "Kubernetes" in body["missing_requirements"]
    assert body["required_skills"]["requirements"][0]["matched"] is False


@pytest.mark.asyncio
async def test_hybrid_response_contains_experience_gap(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    _configure_fake_embeddings(monkeypatch)

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
        evidence_type="experience",
        title="Experience",
        content="Engineering experience",
        embedding=_vector(1.0),
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 200

    body = response.json()

    assert body["experience_years"]["required_years"] == 6
    assert body["experience_years"]["candidate_years"] == 3
    assert body["experience_years"]["matched"] is False
    assert body["experience_years"]["score"] == 0.5
    assert "Minimum 6 years of experience" in body[
        "missing_requirements"
    ]


@pytest.mark.asyncio
async def test_hybrid_matching_unknown_job_returns_404(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    _configure_fake_embeddings(monkeypatch)

    profile = await _create_profile(database_session)

    response = await async_client.post(
        _match_url(uuid4(), profile.id)
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


@pytest.mark.asyncio
async def test_hybrid_matching_missing_requirements_returns_404(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    _configure_fake_embeddings(monkeypatch)

    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Structured job requirements not found"
    )


@pytest.mark.asyncio
async def test_hybrid_matching_unknown_profile_returns_404(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    _configure_fake_embeddings(monkeypatch)

    job = await _create_job(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Python"],
    )

    response = await async_client.post(
        _match_url(job.id, uuid4())
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Candidate profile not found"
    )


@pytest.mark.asyncio
async def test_hybrid_matching_without_evidence_returns_404(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    _configure_fake_embeddings(monkeypatch)

    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Python"],
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Candidate evidence not found"
    )


@pytest.mark.asyncio
async def test_hybrid_matching_without_embeddings_returns_409(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    _configure_fake_embeddings(monkeypatch)

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
        evidence_type="skill",
        title="Python",
        content="Python",
        embedding=None,
        provider=None,
        model=None,
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Compatible candidate evidence embeddings not found"
    )


@pytest.mark.asyncio
async def test_invalid_embedding_configuration_returns_503(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    def raise_configuration_error():
        raise EmbeddingProviderConfigurationError(
            "Invalid embedding configuration"
        )

    monkeypatch.setattr(
        "app.api.routes.hybrid_job_candidate_matching."
        "create_embedding_provider",
        raise_configuration_error,
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Invalid embedding configuration"
    )


@pytest.mark.asyncio
async def test_invalid_job_uuid_returns_422(
    async_client,
) -> None:
    response = await async_client.post(
        f"/api/v1/jobs/not-a-uuid/hybrid-match/{uuid4()}"
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_profile_uuid_returns_422(
    async_client,
) -> None:
    response = await async_client.post(
        f"/api/v1/jobs/{uuid4()}/hybrid-match/not-a-uuid"
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_hybrid_response_contains_all_categories(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    _configure_fake_embeddings(monkeypatch)

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
        evidence_type="skill",
        title="Python",
        content="Python",
        embedding=_vector(1.0),
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 200

    body = response.json()

    assert "required_skills" in body
    assert "preferred_skills" in body
    assert "required_experience" in body
    assert "education" in body
    assert "certifications" in body
    assert "experience_years" in body


@pytest.mark.asyncio
async def test_health_endpoint_still_works(
    async_client,
) -> None:
    response = await async_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}