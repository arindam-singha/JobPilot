"""Review payloads reuse the existing validated profile schemas."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, create_model

from app.schemas.candidate_profile import (
    CandidateAchievementCreate,
    CandidateCertificationCreate,
    CandidateEducationCreate,
    CandidateExperienceCreate,
    CandidateProfileCreate,
    CandidateProjectCreate,
    CandidatePublicationCreate,
    CandidateSkillCreate,
)


def review_row(name, base):
    return create_model(name, __base__=base, id=(UUID | None, None), source_quote=(str, ""))


ExperienceRow = review_row("ExperienceRow", CandidateExperienceCreate)
SkillRow = review_row("SkillRow", CandidateSkillCreate)
EducationRow = review_row("EducationRow", CandidateEducationCreate)
ProjectRow = review_row("ProjectRow", CandidateProjectCreate)
PublicationRow = review_row("PublicationRow", CandidatePublicationCreate)
CertificationRow = review_row("CertificationRow", CandidateCertificationCreate)
AchievementRow = review_row("AchievementRow", CandidateAchievementCreate)


class ProfileImportData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: CandidateProfileCreate
    experiences: list[ExperienceRow] = Field(default_factory=list)
    skills: list[SkillRow] = Field(default_factory=list)
    education: list[EducationRow] = Field(default_factory=list)
    projects: list[ProjectRow] = Field(default_factory=list)
    publications: list[PublicationRow] = Field(default_factory=list)
    certifications: list[CertificationRow] = Field(default_factory=list)
    achievements: list[AchievementRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ProfileImportReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_id: UUID
    document_id: UUID
    base_revision: str
    data: ProfileImportData
    source_text: str = ""


class ProfileImportConfirm(ProfileImportReview):
    confirmed: bool = False
