from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import select

from app.llm.fake_job_requirements_provider import (
    FakeJobRequirementsProvider,
)
from app.models.job import Job
from app.models.job_requirements import JobRequirements
from app.schemas.job_requirements import JobRequirementsData
from app.services.job_requirements_extraction_service import (
    JobRequirementsExtractionService,
    JobRequirementsJobNotFoundError,
    JobRequirementsProviderFailureError,
)


async def _create_job(
    database_session,
    *,
    title: str = "Senior Robotics Engineer",
    company: str = "Example Robotics",
    location: str | None = "Abu Dhabi",
    description: str = (
        "We require Python, ROS2 and computer vision experience. "
        "Candidates must have at least 5 years of experience."
    ),
) -> Job:
    job = Job(
        title=title,
        company=company,
        location=location,
        job_url=f"https://example.com/jobs/{uuid4()}",
        description=description,
        source="manual",
    )

    database_session.add(job)
    await database_session.commit()
    await database_session.refresh(job)

    return job


@pytest.mark.asyncio
async def test_extract_and_persist_creates_requirements(
    database_session,
) -> None:
    job = await _create_job(database_session)

    service = JobRequirementsExtractionService(
        database_session,
        FakeJobRequirementsProvider(),
    )

    result = await service.extract_and_persist(job.id)

    assert result.job_id == job.id
    assert result.provider == "fake"
    assert result.model_name == "fake-job-requirements-model"
    assert "Python" in result.required_skills
    assert "ROS2" in result.required_skills
    assert result.minimum_experience_years == 5


@pytest.mark.asyncio
async def test_extract_and_persist_stores_record_in_database(
    database_session,
) -> None:
    job = await _create_job(database_session)

    service = JobRequirementsExtractionService(
        database_session,
        FakeJobRequirementsProvider(),
    )

    await service.extract_and_persist(job.id)

    query = await database_session.execute(
        select(JobRequirements).where(
            JobRequirements.job_id == job.id
        )
    )

    stored = query.scalar_one_or_none()

    assert stored is not None
    assert stored.job_id == job.id
    assert stored.provider == "fake"


@pytest.mark.asyncio
async def test_extraction_uses_job_fields(
    database_session,
) -> None:
    job = await _create_job(
        database_session,
        title="Custom Title",
        company="Custom Company",
        location="Remote",
        description="Custom description",
    )

    received: dict[str, object] = {}

    def responder(
        title: str,
        company: str,
        location: str | None,
        description: str,
    ) -> JobRequirementsData:
        received["title"] = title
        received["company"] = company
        received["location"] = location
        received["description"] = description

        return JobRequirementsData(
            required_skills=["Custom Skill"]
        )

    service = JobRequirementsExtractionService(
        database_session,
        FakeJobRequirementsProvider(
            responder=responder,
        ),
    )

    result = await service.extract_and_persist(job.id)

    assert received == {
        "title": "Custom Title",
        "company": "Custom Company",
        "location": "Remote",
        "description": "Custom description",
    }
    assert result.required_skills == ["Custom Skill"]


@pytest.mark.asyncio
async def test_extraction_is_idempotent(
    database_session,
) -> None:
    job = await _create_job(database_session)

    service = JobRequirementsExtractionService(
        database_session,
        FakeJobRequirementsProvider(),
    )

    first = await service.extract_and_persist(job.id)
    second = await service.extract_and_persist(job.id)

    query = await database_session.execute(
        select(JobRequirements).where(
            JobRequirements.job_id == job.id
        )
    )

    stored = list(query.scalars().all())

    assert first.id == second.id
    assert len(stored) == 1


@pytest.mark.asyncio
async def test_reextraction_updates_existing_record(
    database_session,
) -> None:
    job = await _create_job(database_session)

    first_provider = FakeJobRequirementsProvider(
        responder=lambda title, company, location, description: (
            JobRequirementsData(
                required_skills=["Python"],
                minimum_experience_years=3,
            )
        )
    )

    first_service = JobRequirementsExtractionService(
        database_session,
        first_provider,
    )

    first = await first_service.extract_and_persist(job.id)

    second_provider = FakeJobRequirementsProvider(
        responder=lambda title, company, location, description: (
            JobRequirementsData(
                required_skills=["Python", "ROS2"],
                minimum_experience_years=5,
            )
        )
    )

    second_service = JobRequirementsExtractionService(
        database_session,
        second_provider,
    )

    second = await second_service.extract_and_persist(job.id)

    assert first.id == second.id
    assert second.required_skills == ["Python", "ROS2"]
    assert second.minimum_experience_years == 5


@pytest.mark.asyncio
async def test_extraction_metadata_is_stored(
    database_session,
) -> None:
    job = await _create_job(database_session)

    service = JobRequirementsExtractionService(
        database_session,
        FakeJobRequirementsProvider(),
    )

    result = await service.extract_and_persist(job.id)

    assert result.extraction_metadata == {
        "generation_method": "llm_structured_extraction",
        "provider": "fake",
        "model": "fake-job-requirements-model",
    }


@pytest.mark.asyncio
async def test_get_requirements_returns_existing_record(
    database_session,
) -> None:
    job = await _create_job(database_session)

    service = JobRequirementsExtractionService(
        database_session,
        FakeJobRequirementsProvider(),
    )

    created = await service.extract_and_persist(job.id)
    retrieved = await service.get_requirements(job.id)

    assert retrieved.id == created.id
    assert retrieved.job_id == job.id


@pytest.mark.asyncio
async def test_unknown_job_is_rejected(
    database_session,
) -> None:
    service = JobRequirementsExtractionService(
        database_session,
        FakeJobRequirementsProvider(),
    )

    with pytest.raises(
        JobRequirementsJobNotFoundError,
        match="not found",
    ):
        await service.extract_and_persist(uuid4())


@pytest.mark.asyncio
async def test_get_requirements_rejects_unknown_job(
    database_session,
) -> None:
    service = JobRequirementsExtractionService(
        database_session,
        FakeJobRequirementsProvider(),
    )

    with pytest.raises(
        JobRequirementsJobNotFoundError,
        match="not found",
    ):
        await service.get_requirements(uuid4())


@pytest.mark.asyncio
async def test_get_requirements_rejects_missing_requirements(
    database_session,
) -> None:
    job = await _create_job(database_session)

    service = JobRequirementsExtractionService(
        database_session,
        FakeJobRequirementsProvider(),
    )

    with pytest.raises(
        JobRequirementsJobNotFoundError,
        match="Structured requirements not found",
    ):
        await service.get_requirements(job.id)


@pytest.mark.asyncio
async def test_provider_failure_is_wrapped(
    database_session,
) -> None:
    job = await _create_job(database_session)

    def failing_responder(
        title: str,
        company: str,
        location: str | None,
        description: str,
    ) -> JobRequirementsData:
        raise RuntimeError("provider unavailable")

    service = JobRequirementsExtractionService(
        database_session,
        FakeJobRequirementsProvider(
            responder=failing_responder,
        ),
    )

    with pytest.raises(
        JobRequirementsProviderFailureError,
        match="failed for job",
    ):
        await service.extract_and_persist(job.id)


@pytest.mark.asyncio
async def test_job_delete_cascades_requirements(
    database_session,
) -> None:
    job = await _create_job(database_session)

    service = JobRequirementsExtractionService(
        database_session,
        FakeJobRequirementsProvider(),
    )

    created = await service.extract_and_persist(job.id)

    await database_session.delete(job)
    await database_session.commit()

    remaining = await database_session.get(
        JobRequirements,
        created.id,
    )

    assert remaining is None