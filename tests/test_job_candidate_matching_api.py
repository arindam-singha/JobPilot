from __future__ import annotations

from uuid import uuid4

import pytest
from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import CandidateProfile
from app.models.job import Job
from app.models.job_requirements import JobRequirements


async def _create_job(database_session) -> Job:
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
        full_name="Test Candidate",
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
) -> CandidateEvidence:
    evidence = CandidateEvidence(
        profile_id=profile_id,
        evidence_type=evidence_type,
        title=title,
        content=content,
        source_type="manual",
        source_id=None,
        metadata_json=None,
    )

    database_session.add(evidence)
    await database_session.commit()
    await database_session.refresh(evidence)

    return evidence


def _match_url(job_id, profile_id) -> str:
    return f"/api/v1/jobs/{job_id}/match/{profile_id}"


@pytest.mark.asyncio
async def test_matching_endpoint_returns_200(
    async_client,
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Python", "ROS2"],
        preferred_skills=["Docker"],
        required_experience=["Robotics engineering"],
        education_requirements=["PhD"],
        minimum_experience_years=5,
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="skill",
        title="Technical skills",
        content="Python, ROS2 and Docker",
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="experience",
        title="Professional experience",
        content="Eight years of robotics engineering experience.",
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="education",
        title="Education",
        content="PhD in Robotics and Artificial Intelligence.",
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 200

    body = response.json()

    assert body["job_id"] == str(job.id)
    assert body["profile_id"] == str(profile.id)
    assert body["overall_score"] == 100
    assert body["evidence_considered"] == 3
    assert body["scoring_method"] == (
        "deterministic_phrase_and_token_matching"
    )


@pytest.mark.asyncio
async def test_matching_response_contains_category_scores(
    async_client,
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Python"],
        preferred_skills=["Docker"],
        required_experience=["Computer vision systems"],
        education_requirements=["PhD"],
        certifications=["AWS"],
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="skill",
        title="Skills",
        content="Python and Docker",
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
async def test_matching_response_contains_supporting_evidence(
    async_client,
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
        evidence_type="skill",
        title="Technical skills",
        content="Python and PyTorch",
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 200

    requirement = response.json()["required_skills"]["requirements"][0]

    assert requirement["matched"] is True
    assert requirement["score"] == 1.0
    assert requirement["supporting_evidence"][0]["evidence_id"] == str(
        evidence.id
    )
    assert requirement["supporting_evidence"][0]["title"] == (
        "Technical skills"
    )


@pytest.mark.asyncio
async def test_matching_response_contains_missing_requirements(
    async_client,
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Python", "Kubernetes"],
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="skill",
        title="Skills",
        content="Python",
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 200

    body = response.json()

    assert "Python" in body["matched_requirements"]
    assert "Kubernetes" in body["missing_requirements"]
    assert body["required_skills"]["score"] == 50


@pytest.mark.asyncio
async def test_matching_response_contains_experience_gap(
    async_client,
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
        minimum_experience_years=5,
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="experience",
        title="Experience",
        content="Worked as a robotics engineer.",
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 200

    body = response.json()

    assert body["experience_years"]["required_years"] == 5
    assert body["experience_years"]["candidate_years"] == 3
    assert body["experience_years"]["matched"] is False
    assert body["experience_years"]["score"] == 0.6
    assert "Minimum 5 years of experience" in body[
        "missing_requirements"
    ]


@pytest.mark.asyncio
async def test_unknown_job_returns_404(
    async_client,
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    response = await async_client.post(
        _match_url(uuid4(), profile.id)
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


@pytest.mark.asyncio
async def test_missing_requirements_returns_404(
    async_client,
    database_session,
) -> None:
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
async def test_unknown_profile_returns_404(
    async_client,
    database_session,
) -> None:
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
async def test_candidate_without_evidence_returns_404(
    async_client,
    database_session,
) -> None:
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
async def test_invalid_job_uuid_returns_422(
    async_client,
) -> None:
    response = await async_client.post(
        f"/api/v1/jobs/not-a-uuid/match/{uuid4()}"
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_profile_uuid_returns_422(
    async_client,
) -> None:
    response = await async_client.post(
        f"/api/v1/jobs/{uuid4()}/match/not-a-uuid"
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_empty_requirement_categories_are_returned(
    async_client,
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
        evidence_type="skill",
        title="Skills",
        content="Python",
    )

    response = await async_client.post(
        _match_url(job.id, profile.id)
    )

    assert response.status_code == 200

    body = response.json()

    assert body["preferred_skills"]["total_requirements"] == 0
    assert body["required_experience"]["total_requirements"] == 0
    assert body["education"]["total_requirements"] == 0
    assert body["certifications"]["total_requirements"] == 0


@pytest.mark.asyncio
async def test_health_endpoint_still_works_after_router_registration(
    async_client,
) -> None:
    response = await async_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}