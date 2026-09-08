from __future__ import annotations

from uuid import uuid4

import pytest
from app.llm.fake_job_requirements_provider import (
    FakeJobRequirementsProvider,
)
from app.models.job import Job
from app.models.job_requirements import JobRequirements
from app.schemas.job_requirements import JobRequirementsData
from sqlalchemy import select


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


def _extract_url(job_id) -> str:
    return f"/api/v1/jobs/{job_id}/requirements/extract"


def _get_url(job_id) -> str:
    return f"/api/v1/jobs/{job_id}/requirements"


@pytest.mark.asyncio
async def test_extract_job_requirements_returns_201(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "fake",
    )

    job = await _create_job(database_session)

    response = await async_client.post(
        _extract_url(job.id)
    )

    assert response.status_code == 201

    body = response.json()

    assert body["job_id"] == str(job.id)
    assert body["provider"] == "fake"
    assert body["model_name"] == "fake-job-requirements-model"
    assert "Python" in body["required_skills"]
    assert "ROS2" in body["required_skills"]
    assert body["minimum_experience_years"] == 5


@pytest.mark.asyncio
async def test_extract_job_requirements_persists_record(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "fake",
    )

    job = await _create_job(database_session)

    response = await async_client.post(
        _extract_url(job.id)
    )

    assert response.status_code == 201

    result = await database_session.execute(
        select(JobRequirements).where(
            JobRequirements.job_id == job.id
        )
    )

    stored = result.scalar_one_or_none()

    assert stored is not None
    assert stored.provider == "fake"
    assert stored.job_id == job.id


@pytest.mark.asyncio
async def test_get_job_requirements_returns_existing_record(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "fake",
    )

    job = await _create_job(database_session)

    create_response = await async_client.post(
        _extract_url(job.id)
    )

    assert create_response.status_code == 201

    get_response = await async_client.get(
        _get_url(job.id)
    )

    assert get_response.status_code == 200
    assert get_response.json()["id"] == create_response.json()["id"]
    assert get_response.json()["job_id"] == str(job.id)


@pytest.mark.asyncio
async def test_repeated_extraction_is_idempotent(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "fake",
    )

    job = await _create_job(database_session)

    first = await async_client.post(
        _extract_url(job.id)
    )
    second = await async_client.post(
        _extract_url(job.id)
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] == second.json()["id"]

    result = await database_session.execute(
        select(JobRequirements).where(
            JobRequirements.job_id == job.id
        )
    )

    records = list(result.scalars().all())

    assert len(records) == 1


@pytest.mark.asyncio
async def test_extract_unknown_job_returns_404(
    async_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "fake",
    )

    response = await async_client.post(
        _extract_url(uuid4())
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


@pytest.mark.asyncio
async def test_get_unknown_job_returns_404(
    async_client,
) -> None:
    response = await async_client.get(
        _get_url(uuid4())
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_missing_requirements_returns_404(
    async_client,
    database_session,
) -> None:
    job = await _create_job(database_session)

    response = await async_client.get(
        _get_url(job.id)
    )

    assert response.status_code == 404
    assert "structured requirements not found" in (
        response.json()["detail"].lower()
    )


@pytest.mark.asyncio
async def test_invalid_job_uuid_returns_422(
    async_client,
) -> None:
    extract_response = await async_client.post(
        "/api/v1/jobs/not-a-uuid/requirements/extract"
    )

    get_response = await async_client.get(
        "/api/v1/jobs/not-a-uuid/requirements"
    )

    assert extract_response.status_code == 422
    assert get_response.status_code == 422


@pytest.mark.asyncio
async def test_provider_failure_returns_502(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    job = await _create_job(database_session)

    def failing_responder(
        title: str,
        company: str,
        location: str | None,
        description: str,
    ) -> JobRequirementsData:
        raise RuntimeError("Provider unavailable")

    provider = FakeJobRequirementsProvider(
        responder=failing_responder,
    )

    monkeypatch.setattr(
        "app.api.routes.job_requirements."
        "create_job_requirements_provider",
        lambda: provider,
    )

    response = await async_client.post(
        _extract_url(job.id)
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Job requirements provider failed"
    )


@pytest.mark.asyncio
async def test_invalid_provider_configuration_returns_503(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    from app.llm.job_requirements_provider_factory import (
        JobRequirementsProviderConfigurationError,
    )

    job = await _create_job(database_session)

    def raise_configuration_error():
        raise JobRequirementsProviderConfigurationError(
            "Invalid provider configuration"
        )

    monkeypatch.setattr(
        "app.api.routes.job_requirements."
        "create_job_requirements_provider",
        raise_configuration_error,
    )

    response = await async_client.post(
        _extract_url(job.id)
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Invalid provider configuration"
    )


@pytest.mark.asyncio
async def test_extraction_response_contains_metadata(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JOB_REQUIREMENTS_LLM_PROVIDER",
        "fake",
    )

    job = await _create_job(database_session)

    response = await async_client.post(
        _extract_url(job.id)
    )

    assert response.status_code == 201

    metadata = response.json()["extraction_metadata"]

    assert metadata == {
        "generation_method": "llm_structured_extraction",
        "provider": "fake",
        "model": "fake-job-requirements-model",
    }