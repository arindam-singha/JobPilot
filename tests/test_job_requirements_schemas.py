from __future__ import annotations

import pytest
from app.schemas.job_requirements import (
    JobRequirementsCreate,
    JobRequirementsData,
)
from pydantic import ValidationError


def test_job_requirements_data_accepts_valid_values() -> None:
    data = JobRequirementsData(
        required_skills=["Python", "ROS2"],
        preferred_skills=["Isaac Sim"],
        required_experience=["Computer vision systems"],
        responsibilities=["Develop robotics applications"],
        education_requirements=["Master's degree"],
        certifications=["AWS certification"],
        domain_keywords=["robotics"],
        minimum_experience_years=5,
    )

    assert data.required_skills == ["Python", "ROS2"]
    assert data.minimum_experience_years == 5


def test_string_lists_are_trimmed_and_deduplicated() -> None:
    data = JobRequirementsData(
        required_skills=[
            " Python ",
            "python",
            "",
            "   ",
            "ROS2",
            "ros2",
        ]
    )

    assert data.required_skills == [
        "Python",
        "ROS2",
    ]


def test_normalization_preserves_first_value_casing() -> None:
    data = JobRequirementsData(
        domain_keywords=[
            "Computer Vision",
            "computer vision",
        ]
    )

    assert data.domain_keywords == [
        "Computer Vision",
    ]


def test_negative_minimum_experience_is_rejected() -> None:
    with pytest.raises(ValidationError):
        JobRequirementsData(
            minimum_experience_years=-1,
        )


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        JobRequirementsData.model_validate(
            {
                "required_skills": ["Python"],
                "unexpected_field": "invalid",
            }
        )


def test_job_requirements_create_normalizes_provider_and_model() -> None:
    data = JobRequirementsCreate(
        required_skills=["Python"],
        provider=" ollama ",
        model_name=" qwen2.5:7b ",
    )

    assert data.provider == "ollama"
    assert data.model_name == "qwen2.5:7b"


@pytest.mark.parametrize(
    "field_name",
    [
        "provider",
        "model_name",
    ],
)
def test_job_requirements_create_rejects_empty_provider_fields(
    field_name: str,
) -> None:
    payload = {
        "provider": "fake",
        "model_name": "fake-model",
    }
    payload[field_name] = "   "

    with pytest.raises(ValidationError):
        JobRequirementsCreate.model_validate(payload)