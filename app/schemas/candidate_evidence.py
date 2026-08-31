from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CandidateEvidenceCreate(BaseModel):
    evidence_type: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source_type: str = Field(..., min_length=1)
    source_id: UUID | None = None
    metadata_json: str | None = None

    @field_validator(
        "evidence_type",
        "title",
        "content",
        "source_type",
    )
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("field must not be empty or whitespace")

        return value

    @field_validator("metadata_json")
    @classmethod
    def normalize_metadata(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None


class CandidateEvidenceRead(BaseModel):
    id: UUID
    profile_id: UUID
    evidence_type: str
    title: str
    content: str
    source_type: str
    source_id: UUID | None = None
    metadata_json: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CandidateEvidenceUpdate(BaseModel):
    evidence_type: str | None = None
    title: str | None = None
    content: str | None = None
    source_type: str | None = None
    source_id: UUID | None = None
    metadata_json: str | None = None

    @field_validator(
        "evidence_type",
        "title",
        "content",
        "source_type",
    )
    @classmethod
    def validate_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError("field must not be empty or whitespace")

        return value

    @field_validator("metadata_json")
    @classmethod
    def normalize_metadata(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None