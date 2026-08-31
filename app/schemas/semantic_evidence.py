from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SemanticEvidenceMatch(BaseModel):
    evidence_id: UUID
    profile_id: UUID
    evidence_type: str
    title: str
    content: str
    source_type: str
    source_id: UUID | None = None
    similarity: float = Field(..., ge=0.0, le=1.0)
    embedding_provider: str
    embedding_model: str

    model_config = ConfigDict(extra="forbid")


class SemanticEvidenceSearchResult(BaseModel):
    profile_id: UUID
    query: str
    provider: str
    model: str
    matches: list[SemanticEvidenceMatch] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("query must not be empty or whitespace")

        return cleaned

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )