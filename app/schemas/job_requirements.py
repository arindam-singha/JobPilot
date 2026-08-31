from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobRequirementsData(BaseModel):
    """Validated structured requirements extracted from a job description."""

    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    required_experience: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    domain_keywords: list[str] = Field(default_factory=list)
    minimum_experience_years: float | None = Field(
        default=None,
        ge=0,
    )

    @field_validator(
        "required_skills",
        "preferred_skills",
        "required_experience",
        "responsibilities",
        "education_requirements",
        "certifications",
        "domain_keywords",
    )
    @classmethod
    def normalize_string_lists(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for value in values:
            cleaned = value.strip()

            if not cleaned:
                continue

            deduplication_key = cleaned.casefold()

            if deduplication_key in seen:
                continue

            seen.add(deduplication_key)
            normalized.append(cleaned)

        return normalized

    model_config = ConfigDict(extra="forbid")


class JobRequirementsCreate(JobRequirementsData):
    provider: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    extraction_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
    extra="forbid",
    protected_namespaces=(),
    )

    @field_validator("provider", "model_name")
    @classmethod
    def normalize_required_strings(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("field must not be empty or whitespace")

        return value


class JobRequirementsRead(JobRequirementsData):
    id: UUID
    job_id: UUID
    provider: str
    model_name: str
    extraction_metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
    from_attributes=True,
    extra="forbid",
    protected_namespaces=(),
    )