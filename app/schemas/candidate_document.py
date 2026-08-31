from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CandidateDocumentCreate(BaseModel):
    filename: str = Field(..., min_length=1)
    content_type: str = Field(..., min_length=1)
    storage_path: str = Field(..., min_length=1)
    file_size: int = Field(..., ge=0)
    extracted_text: Optional[str] = None
    extraction_status: str = "uploaded"
    extraction_error: Optional[str] = None

    @field_validator("filename", "content_type", "storage_path")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field must not be empty or whitespace")
        return value

    @field_validator("extracted_text", "extraction_error")
    @classmethod
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class CandidateDocumentRead(BaseModel):
    id: UUID
    profile_id: UUID
    filename: str
    content_type: str
    storage_path: str
    file_size: int
    extracted_text: Optional[str] = None
    extraction_status: str
    extraction_error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CandidateDocumentUpdate(BaseModel):
    extracted_text: Optional[str] = None
    extraction_status: Optional[str] = None
    extraction_error: Optional[str] = None
    storage_path: Optional[str] = None

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("storage_path must not be empty or whitespace")
        return value

    @field_validator("extracted_text", "extraction_error")
    @classmethod
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("extraction_status")
    @classmethod
    def validate_extraction_status(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("extraction_status must not be empty or whitespace")
        return value
