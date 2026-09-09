from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SkillGapEvidence(BaseModel):
    evidence_id: UUID
    evidence_type: str
    title: str
    content: str

    model_config = ConfigDict(extra="forbid")


class SkillGapGenerationContext(BaseModel):
    job_id: UUID
    profile_id: UUID
    job_title: str
    company: str
    job_description: str
    overall_match_score: float = Field(..., ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    preliminary_missing_skills: list[str] = Field(default_factory=list)
    candidate_evidence: list[SkillGapEvidence] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ResolvedSkillEquivalence(BaseModel):
    job_requirement: str = Field(..., min_length=1)
    candidate_term: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class MissingSkillPreparation(BaseModel):
    skill: str = Field(..., min_length=1)
    priority: Literal["high", "medium", "low"]
    why_it_matters: str = Field(..., min_length=1)
    candidate_overlap: str | None = None
    preparation_topics: list[str] = Field(min_length=1, max_length=6)
    practical_exercise: str = Field(..., min_length=1)
    interview_questions: list[str] = Field(min_length=1, max_length=6)

    @field_validator("preparation_topics", "interview_questions")
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty item is required")
        return list(dict.fromkeys(cleaned))

    model_config = ConfigDict(extra="forbid")


class SkillGapReportContent(BaseModel):
    executive_summary: str = Field(..., min_length=1)
    resolved_equivalences: list[ResolvedSkillEquivalence] = Field(default_factory=list)
    missing_skills: list[MissingSkillPreparation] = Field(default_factory=list)
    preparation_strategy: list[str] = Field(min_length=1, max_length=8)

    model_config = ConfigDict(extra="forbid")


class SkillGapReport(BaseModel):
    job_id: UUID
    profile_id: UUID
    job_title: str
    company: str
    overall_match_score: float = Field(..., ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    executive_summary: str
    resolved_equivalences: list[ResolvedSkillEquivalence] = Field(default_factory=list)
    missing_skills: list[MissingSkillPreparation] = Field(default_factory=list)
    preparation_strategy: list[str] = Field(default_factory=list)
    generator_provider: str
    generator_model: str

    model_config = ConfigDict(extra="forbid")
