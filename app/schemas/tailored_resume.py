from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ResumeRequirementTrace(BaseModel):
    """Why a stored evidence item was selected for a job requirement."""

    category: str = Field(..., min_length=1)
    requirement: str = Field(..., min_length=1)
    deterministic_score: float = Field(..., ge=0.0, le=1.0)
    semantic_score: float = Field(..., ge=0.0, le=1.0)
    hybrid_score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)

    model_config = ConfigDict(extra="forbid")


class SelectedResumeEvidence(BaseModel):
    """Authoritative evidence content and its complete selection provenance."""

    evidence_id: UUID
    profile_id: UUID
    evidence_type: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source_type: str = Field(..., min_length=1)
    source_id: UUID | None = None
    metadata_json: str | None = None
    selection_score: float = Field(..., ge=0.0, le=1.0)
    requirement_traces: list[ResumeRequirementTrace] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class ResumeGroundingBundle(BaseModel):
    """Immutable, job-specific input boundary for tailored resume generation."""

    job_id: UUID
    profile_id: UUID
    hybrid_overall_score: float = Field(..., ge=0.0, le=100.0)
    matched_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    selected_evidence: list[SelectedResumeEvidence] = Field(min_length=1)
    embedding_provider: str = Field(..., min_length=1)
    embedding_model: str = Field(..., min_length=1)
    selection_method: str = "matched_requirement_hybrid_score"
    minimum_evidence_score: float = Field(..., ge=0.0, le=1.0)
    max_evidence_per_requirement: int = Field(..., ge=1)
    max_total_evidence: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_selected_evidence(self) -> ResumeGroundingBundle:
        evidence_ids = [item.evidence_id for item in self.selected_evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("selected_evidence must not contain duplicate evidence IDs")

        if any(item.profile_id != self.profile_id for item in self.selected_evidence):
            raise ValueError("all selected evidence must belong to the requested profile")

        return self

    model_config = ConfigDict(extra="forbid")


class ResumeHeader(BaseModel):
    """ATS-safe candidate contact header; these fields are profile facts."""

    full_name: str = Field(..., min_length=1)
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    github_url: str | None = None
    portfolio_url: str | None = None

    model_config = ConfigDict(extra="forbid")


class GroundedResumeStatement(BaseModel):
    """A generated resume claim backed by one or more stored evidence items."""

    text: str = Field(..., min_length=1)
    evidence_ids: list[UUID] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("text must not be empty or whitespace")
        return cleaned

    @field_validator("evidence_ids")
    @classmethod
    def validate_unique_evidence_ids(cls, value: list[UUID]) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_ids must not contain duplicates")
        return value

    model_config = ConfigDict(extra="forbid")


class TailoredResumeSection(BaseModel):
    """An ATS-friendly named section containing only grounded statements."""

    heading: str = Field(..., min_length=1)
    statements: list[GroundedResumeStatement] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class ResumeGenerationContext(BaseModel):
    """Trusted facts supplied to a resume-generation provider."""

    job_title: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    location: str | None = None
    job_description: str = Field(..., min_length=1)
    header: ResumeHeader
    grounding: ResumeGroundingBundle

    model_config = ConfigDict(extra="forbid")


class TailoredResumeContent(BaseModel):
    """LLM-produced content before immutable facts and provenance are attached."""

    target_title: str = Field(..., min_length=1)
    professional_summary: list[GroundedResumeStatement] = Field(min_length=1)
    sections: list[TailoredResumeSection] = Field(default_factory=list)
    skills: list[GroundedResumeStatement] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

class ResumeSkillGroup(BaseModel):
    """Verified skills grouped for ATS-friendly rendering."""

    category: str = Field(..., min_length=1)
    skills: list[str] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class ResumeExperienceEntry(BaseModel):
    """Professional experience copied from the confirmed profile."""

    role: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    is_current: bool = False
    bullets: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ResumeProjectEntry(BaseModel):
    """Project copied from the confirmed profile."""

    name: str = Field(..., min_length=1)
    technologies: str | None = None
    description: str | None = None
    achievements: str | None = None
    project_url: str | None = None

    model_config = ConfigDict(extra="forbid")


class ResumeEducationEntry(BaseModel):
    """Education copied from the confirmed profile."""

    institution: str = Field(..., min_length=1)
    degree: str = Field(..., min_length=1)
    field_of_study: str | None = None
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


class ResumePublicationEntry(BaseModel):
    """Publication copied from the confirmed profile."""

    title: str = Field(..., min_length=1)
    venue: str | None = None
    publication_date: str | None = None
    url: str | None = None
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


class ResumeCertificationEntry(BaseModel):
    """Certification copied from the confirmed profile."""

    name: str = Field(..., min_length=1)
    issuing_organization: str | None = None
    issue_date: str | None = None
    expiry_date: str | None = None
    credential_id: str | None = None
    credential_url: str | None = None

    model_config = ConfigDict(extra="forbid")


class ResumeAchievementEntry(BaseModel):
    """Achievement copied from the confirmed profile."""

    title: str = Field(..., min_length=1)
    description: str | None = None
    date: str | None = None

    model_config = ConfigDict(extra="forbid")

class TailoredResumeDraft(BaseModel):
    """Structured resume output produced before Markdown or PDF rendering."""

    job_id: UUID
    profile_id: UUID
    header: ResumeHeader
    target_title: str = Field(..., min_length=1)
    verified_summary: str | None = None
    professional_summary: list[GroundedResumeStatement] = Field(min_length=1)
    sections: list[TailoredResumeSection] = Field(default_factory=list)
    skills: list[GroundedResumeStatement] = Field(default_factory=list)
    skill_groups: list[ResumeSkillGroup] = Field(default_factory=list)
    experiences: list[ResumeExperienceEntry] = Field(default_factory=list)
    projects: list[ResumeProjectEntry] = Field(default_factory=list)
    education: list[ResumeEducationEntry] = Field(default_factory=list)
    publications: list[ResumePublicationEntry] = Field(default_factory=list)
    certifications: list[ResumeCertificationEntry] = Field(default_factory=list)
    achievements: list[ResumeAchievementEntry] = Field(default_factory=list)
    evidence_catalog: list[SelectedResumeEvidence] = Field(min_length=1)
    generator_provider: str = Field(..., min_length=1)
    generator_model: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_statement_provenance(self) -> TailoredResumeDraft:
        catalog_ids = {item.evidence_id for item in self.evidence_catalog}
        statements = [
            *self.professional_summary,
            *self.skills,
            *(statement for section in self.sections for statement in section.statements),
        ]
        unknown_ids = {
            evidence_id
            for statement in statements
            for evidence_id in statement.evidence_ids
            if evidence_id not in catalog_ids
        }
        if unknown_ids:
            raise ValueError("resume statements reference evidence absent from evidence_catalog")
        return self

    model_config = ConfigDict(extra="forbid")
