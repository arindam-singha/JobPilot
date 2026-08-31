from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class JobCreate(BaseModel):
    title: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    location: str | None = None
    job_url: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)

    @field_validator(
        "title",
        "company",
        "job_url",
        "description",
        "source",
    )
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Field must not be empty or whitespace")

        return value

    @field_validator("location")
    @classmethod
    def normalize_location(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None


class JobIngestRequest(BaseModel):
    job_url: str = Field(..., min_length=1)
    job_description: str = Field(..., min_length=20)
    title: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    location: str | None = None

    @field_validator("job_url", "title", "company", "job_description")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Field must not be empty or whitespace")

        return value

    @field_validator("location")
    @classmethod
    def normalize_location(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None


class JobRead(BaseModel):
    id: UUID
    title: str
    company: str
    location: str | None = None
    job_url: str
    description: str
    source: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}