from __future__ import annotations

from uuid import uuid4

import pytest
from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import CandidateProfile
from app.models.job import Job
from app.models.job_requirements import JobRequirements
from app.services.job_candidate_matching_service import (
    JobCandidateMatchingService,
    MatchingCandidateEvidenceNotFoundError,
    MatchingCandidateProfileNotFoundError,
    MatchingJobNotFoundError,
    MatchingRequirementsNotFoundError,
)


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


@pytest.mark.asyncio
async def test_exact_required_skill_match(
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
        content="Python, PyTorch and FastAPI",
    )

    service = JobCandidateMatchingService(database_session)

    result = await service.match(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert result.required_skills.score == 100
    assert result.required_skills.matched_requirements == 1

    requirement = result.required_skills.requirements[0]

    assert requirement.matched is True
    assert requirement.score == 1.0
    assert requirement.supporting_evidence[0].evidence_id == evidence.id


@pytest.mark.asyncio
async def test_missing_required_skill(
    database_session,
) -> None:
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
        title="Technical skills",
        content="Python and Docker",
    )

    result = await JobCandidateMatchingService(
        database_session
    ).match(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert result.required_skills.score == 0
    assert result.required_skills.requirements[0].matched is False
    assert "Kubernetes" in result.missing_requirements


@pytest.mark.asyncio
async def test_case_insensitive_skill_match(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["PYTORCH"],
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="skill",
        title="Skills",
        content="Experienced with PyTorch.",
    )

    result = await JobCandidateMatchingService(
        database_session
    ).match(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert result.required_skills.score == 100


@pytest.mark.asyncio
async def test_experience_years_match(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(
        database_session,
        years=8,
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
        title="Professional experience",
        content="Worked as a robotics engineer.",
    )

    result = await JobCandidateMatchingService(
        database_session
    ).match(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert result.experience_years.matched is True
    assert result.experience_years.score == 1.0
    assert result.overall_score == 100


@pytest.mark.asyncio
async def test_insufficient_experience_years_uses_proportional_score(
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
        evidence_type="experience",
        title="Professional experience",
        content="Worked as an engineer.",
    )

    result = await JobCandidateMatchingService(
        database_session
    ).match(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert result.experience_years.matched is False
    assert result.experience_years.score == 0.5
    assert result.overall_score == 50


@pytest.mark.asyncio
async def test_education_matches_only_relevant_evidence_types(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        education_requirements=[
            "PhD in Robotics",
        ],
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="education",
        title="Education",
        content="PhD in Robotics and Artificial Intelligence",
    )

    result = await JobCandidateMatchingService(
        database_session
    ).match(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert result.education.score == 100


@pytest.mark.asyncio
async def test_weight_redistribution_for_active_categories(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=[
            "Python",
            "Kubernetes",
        ],
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="skill",
        title="Skills",
        content="Python",
    )

    result = await JobCandidateMatchingService(
        database_session
    ).match(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert result.required_skills.score == 50
    assert result.overall_score == 50


@pytest.mark.asyncio
async def test_multiple_categories_use_configured_weights(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Python"],
        education_requirements=["PhD"],
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="skill",
        title="Skills",
        content="Python",
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="education",
        title="Education",
        content="Master of Science",
    )

    result = await JobCandidateMatchingService(
        database_session
    ).match(
        job_id=job.id,
        profile_id=profile.id,
    )

    expected = 100 * (
        JobCandidateMatchingService.REQUIRED_SKILLS_WEIGHT
        / (
            JobCandidateMatchingService.REQUIRED_SKILLS_WEIGHT
            + JobCandidateMatchingService.EDUCATION_WEIGHT
        )
    )

    assert result.overall_score == round(expected, 2)


@pytest.mark.asyncio
async def test_unknown_job_is_rejected(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    service = JobCandidateMatchingService(database_session)

    with pytest.raises(MatchingJobNotFoundError):
        await service.match(
            job_id=uuid4(),
            profile_id=profile.id,
        )


@pytest.mark.asyncio
async def test_missing_requirements_are_rejected(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    service = JobCandidateMatchingService(database_session)

    with pytest.raises(MatchingRequirementsNotFoundError):
        await service.match(
            job_id=job.id,
            profile_id=profile.id,
        )


@pytest.mark.asyncio
async def test_unknown_profile_is_rejected(
    database_session,
) -> None:
    job = await _create_job(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Python"],
    )

    service = JobCandidateMatchingService(database_session)

    with pytest.raises(MatchingCandidateProfileNotFoundError):
        await service.match(
            job_id=job.id,
            profile_id=uuid4(),
        )


@pytest.mark.asyncio
async def test_candidate_without_evidence_is_rejected(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Python"],
    )

    service = JobCandidateMatchingService(database_session)

    with pytest.raises(MatchingCandidateEvidenceNotFoundError):
        await service.match(
            job_id=job.id,
            profile_id=profile.id,
        )


@pytest.mark.asyncio
async def test_supporting_evidence_is_limited_to_five(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    await _create_requirements(
        database_session,
        job_id=job.id,
        required_skills=["Python"],
    )

    for index in range(7):
        await _create_evidence(
            database_session,
            profile_id=profile.id,
            evidence_type="skill",
            title=f"Python evidence {index}",
            content="Python",
        )

    result = await JobCandidateMatchingService(
        database_session
    ).match(
        job_id=job.id,
        profile_id=profile.id,
    )

    supporting = (
        result.required_skills
        .requirements[0]
        .supporting_evidence
    )

    assert len(supporting) == 5