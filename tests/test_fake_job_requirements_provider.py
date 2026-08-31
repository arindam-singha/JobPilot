from __future__ import annotations

import pytest

from app.llm.fake_job_requirements_provider import (
    FakeJobRequirementsProvider,
)
from app.schemas.job_requirements import JobRequirementsData


@pytest.mark.asyncio
async def test_fake_provider_has_stable_identity() -> None:
    provider = FakeJobRequirementsProvider()

    assert provider.provider_name == "fake"
    assert provider.model_name == "fake-job-requirements-model"


@pytest.mark.asyncio
async def test_fake_provider_extracts_known_skills() -> None:
    provider = FakeJobRequirementsProvider()

    result = await provider.extract_requirements(
        title="Senior Robotics Engineer",
        company="Example Robotics",
        location="Abu Dhabi",
        description=(
            "We require Python, ROS2, Docker, computer vision "
            "and deep learning experience."
        ),
    )

    assert result.required_skills == [
        "Python",
        "Docker",
        "ROS2",
        "Computer Vision",
        "Deep Learning",
    ]


@pytest.mark.asyncio
async def test_fake_provider_extracts_experience_categories() -> None:
    provider = FakeJobRequirementsProvider()

    result = await provider.extract_requirements(
        title="Machine Learning Engineer",
        company="Example AI",
        location=None,
        description=(
            "Build production AI and machine learning systems "
            "for manufacturing applications."
        ),
    )

    assert "Machine-learning system development" in (
        result.required_experience
    )
    assert "Production AI systems" in result.required_experience
    assert "Manufacturing applications" in (
        result.required_experience
    )


@pytest.mark.asyncio
async def test_fake_provider_extracts_minimum_experience() -> None:
    provider = FakeJobRequirementsProvider()

    result = await provider.extract_requirements(
        title="Senior AI Engineer",
        company="Example",
        location=None,
        description=(
            "Candidates must have at least 5 years of experience "
            "with Python and machine learning."
        ),
    )

    assert result.minimum_experience_years == 5


@pytest.mark.asyncio
async def test_fake_provider_extracts_decimal_experience() -> None:
    provider = FakeJobRequirementsProvider()

    result = await provider.extract_requirements(
        title="AI Engineer",
        company="Example",
        location=None,
        description=(
            "Minimum of 3.5 years of experience is required."
        ),
    )

    assert result.minimum_experience_years == 3.5


@pytest.mark.asyncio
async def test_fake_provider_returns_empty_lists_for_unknown_text() -> None:
    provider = FakeJobRequirementsProvider()

    result = await provider.extract_requirements(
        title="Specialist",
        company="Example",
        location=None,
        description="Perform assigned duties.",
    )

    assert result.required_skills == []
    assert result.required_experience == []
    assert result.domain_keywords == []
    assert result.minimum_experience_years is None


@pytest.mark.asyncio
async def test_custom_responder_is_used() -> None:
    def responder(
        title: str,
        company: str,
        location: str | None,
        description: str,
    ) -> JobRequirementsData:
        assert title == "Custom Role"
        assert company == "Custom Company"
        assert location == "Remote"
        assert description == "Custom description"

        return JobRequirementsData(
            required_skills=["Custom Skill"],
            preferred_skills=["Preferred Skill"],
            minimum_experience_years=7,
        )

    provider = FakeJobRequirementsProvider(
        responder=responder,
    )

    result = await provider.extract_requirements(
        title="Custom Role",
        company="Custom Company",
        location="Remote",
        description="Custom description",
    )

    assert result.required_skills == ["Custom Skill"]
    assert result.preferred_skills == ["Preferred Skill"]
    assert result.minimum_experience_years == 7