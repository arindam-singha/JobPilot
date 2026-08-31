from __future__ import annotations

import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.tailored_resume import (
    ResumeGroundingBundle,
    ResumeHeader,
    SelectedResumeEvidence,
)


class GroundedCoverLetterParagraph(BaseModel):
    """A cover-letter paragraph backed by stored candidate evidence."""

    text: str = Field(..., min_length=1)
    evidence_ids: list[UUID] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned:
            raise ValueError("text must not be empty or whitespace")
        return cleaned

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, value: list[UUID]) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_ids must not contain duplicates")
        return value

    model_config = ConfigDict(extra="forbid")


class CoverLetterGenerationContext(BaseModel):
    """Trusted job, profile, and evidence supplied to the LLM."""

    job_title: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    location: str | None = None
    job_description: str = Field(..., min_length=1)
    header: ResumeHeader
    grounding: ResumeGroundingBundle

    model_config = ConfigDict(extra="forbid")


class TailoredCoverLetterContent(BaseModel):
    """LLM output before immutable job/profile facts are attached."""

    subject: str = Field(..., min_length=1)
    salutation: str = Field(..., min_length=1)
    opening: GroundedCoverLetterParagraph
    body_paragraphs: list[GroundedCoverLetterParagraph] = Field(min_length=1, max_length=3)
    closing: GroundedCoverLetterParagraph
    sign_off: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class TailoredCoverLetterDraft(BaseModel):
    """Structured, grounded cover letter ready for rendering."""

    job_id: UUID
    profile_id: UUID
    header: ResumeHeader
    job_title: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    subject: str = Field(..., min_length=1)
    salutation: str = Field(..., min_length=1)
    opening: GroundedCoverLetterParagraph
    body_paragraphs: list[GroundedCoverLetterParagraph] = Field(min_length=1, max_length=3)
    closing: GroundedCoverLetterParagraph
    sign_off: str = Field(..., min_length=1)
    evidence_catalog: list[SelectedResumeEvidence] = Field(min_length=1)
    generator_provider: str = Field(..., min_length=1)
    generator_model: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_provenance(self) -> TailoredCoverLetterDraft:
        allowed = {item.evidence_id for item in self.evidence_catalog}
        paragraphs = [self.opening, *self.body_paragraphs, self.closing]
        unknown = {
            evidence_id
            for paragraph in paragraphs
            for evidence_id in paragraph.evidence_ids
            if evidence_id not in allowed
        }
        if unknown:
            raise ValueError("cover-letter paragraphs reference unknown evidence")
        return self

    model_config = ConfigDict(extra="forbid")
