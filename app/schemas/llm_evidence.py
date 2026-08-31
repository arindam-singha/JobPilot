from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LlmEvidenceItem(BaseModel):
    """One structured evidence item extracted from a document chunk."""

    evidence_type: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "evidence_type",
        "title",
        "content",
    )
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("field must not be empty or whitespace")

        return value

    model_config = ConfigDict(extra="forbid")


class LlmChunkExtractionResult(BaseModel):
    """Structured extraction result for one document chunk."""

    evidence: list[LlmEvidenceItem] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")