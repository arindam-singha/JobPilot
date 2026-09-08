from __future__ import annotations

from datetime import date as date_type
from datetime import datetime as datetime_type
from uuid import UUID

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator, model_validator


def _strip_optional_string(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.strip()
    return value or None


def _validate_non_empty_string(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty or whitespace")
    return value


class CandidateProfileCreate(BaseModel):
    full_name: str = Field(..., min_length=1)
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: AnyUrl | None = None
    github_url: AnyUrl | None = None
    portfolio_url: AnyUrl | None = None
    professional_summary: str | None = None
    target_roles: str | None = None
    total_experience_years: float | None = None

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str) -> str:
        return _validate_non_empty_string(value, "full_name")

    @field_validator(
        "email",
        "phone",
        "location",
        "professional_summary",
        "target_roles",
    )
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        return _strip_optional_string(value)

    @field_validator("total_experience_years")
    @classmethod
    def validate_total_experience_years(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("total_experience_years must be greater than or equal to 0")
        return value


class CandidateProfileUpdate(BaseModel):
    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: AnyUrl | None = None
    github_url: AnyUrl | None = None
    portfolio_url: AnyUrl | None = None
    professional_summary: str | None = None
    target_roles: str | None = None
    total_experience_years: float | None = None

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_non_empty_string(value, "full_name")

    @field_validator(
        "email",
        "phone",
        "location",
        "professional_summary",
        "target_roles",
    )
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        return _strip_optional_string(value)

    @field_validator("total_experience_years")
    @classmethod
    def validate_total_experience_years(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("total_experience_years must be greater than or equal to 0")
        return value


class CandidateExperienceCreate(BaseModel):
    company: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    location: str | None = None
    start_date: date_type | None = None
    end_date: date_type | None = None
    is_current: bool = False
    description: str | None = None
    achievements: str | None = None

    @field_validator("company")
    @classmethod
    def validate_company(cls, value: str) -> str:
        return _validate_non_empty_string(value, "company")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        return _validate_non_empty_string(value, "role")

    @field_validator("location", "description", "achievements")
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        return _strip_optional_string(value)

    @model_validator(mode="after")
    def validate_date_ranges(self) -> CandidateExperienceCreate:
        if self.start_date is not None and self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        if self.is_current and self.end_date is not None:
            raise ValueError("end_date must be None when is_current is true")
        return self


class CandidateExperienceRead(BaseModel):
    id: UUID
    profile_id: UUID
    company: str
    role: str
    location: str | None = None
    start_date: date_type | None = None
    end_date: date_type | None = None
    is_current: bool = False
    description: str | None = None
    achievements: str | None = None

    model_config = ConfigDict(from_attributes=True)


class CandidateSkillCreate(BaseModel):
    name: str = Field(..., min_length=1)
    category: str | None = None
    proficiency: str | None = None
    years_of_experience: float | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_non_empty_string(value, "name")

    @field_validator("category", "proficiency")
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        return _strip_optional_string(value)

    @field_validator("years_of_experience")
    @classmethod
    def validate_years_of_experience(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("years_of_experience must be greater than or equal to 0")
        return value


class CandidateSkillRead(BaseModel):
    id: UUID
    profile_id: UUID
    name: str
    category: str | None = None
    proficiency: str | None = None
    years_of_experience: float | None = None

    model_config = ConfigDict(from_attributes=True)


class CandidateEducationCreate(BaseModel):
    institution: str = Field(..., min_length=1)
    degree: str = Field(..., min_length=1)
    field_of_study: str | None = None
    location: str | None = None
    start_date: date_type | None = None
    end_date: date_type | None = None
    description: str | None = None

    @field_validator("institution")
    @classmethod
    def validate_institution(cls, value: str) -> str:
        return _validate_non_empty_string(value, "institution")

    @field_validator("degree")
    @classmethod
    def validate_degree(cls, value: str) -> str:
        return _validate_non_empty_string(value, "degree")

    @field_validator("field_of_study", "location", "description")
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        return _strip_optional_string(value)

    @model_validator(mode="after")
    def validate_date_ranges(self) -> CandidateEducationCreate:
        if self.start_date is not None and self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        return self


class CandidateEducationRead(BaseModel):
    id: UUID
    profile_id: UUID
    institution: str
    degree: str
    field_of_study: str | None = None
    location: str | None = None
    start_date: date_type | None = None
    end_date: date_type | None = None
    description: str | None = None

    model_config = ConfigDict(from_attributes=True)


class CandidateProjectCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str | None = None
    technologies: str | None = None
    achievements: str | None = None
    project_url: AnyUrl | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_non_empty_string(value, "name")

    @field_validator("description", "technologies", "achievements")
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        return _strip_optional_string(value)


class CandidateProjectRead(BaseModel):
    id: UUID
    profile_id: UUID
    name: str
    description: str | None = None
    technologies: str | None = None
    achievements: str | None = None
    project_url: AnyUrl | None = None

    model_config = ConfigDict(from_attributes=True)


class CandidatePublicationCreate(BaseModel):
    title: str = Field(..., min_length=1)
    venue: str | None = None
    publication_date: date_type | None = None
    url: AnyUrl | None = None
    description: str | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _validate_non_empty_string(value, "title")

    @field_validator("venue", "description")
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        return _strip_optional_string(value)


class CandidatePublicationRead(BaseModel):
    id: UUID
    profile_id: UUID
    title: str
    venue: str | None = None
    publication_date: date_type | None = None
    url: AnyUrl | None = None
    description: str | None = None

    model_config = ConfigDict(from_attributes=True)


class CandidateCertificationCreate(BaseModel):
    name: str = Field(..., min_length=1)
    issuing_organization: str | None = None
    issue_date: date_type | None = None
    expiry_date: date_type | None = None
    credential_id: str | None = None
    credential_url: AnyUrl | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_non_empty_string(value, "name")

    @field_validator("issuing_organization", "credential_id")
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        return _strip_optional_string(value)

    @model_validator(mode="after")
    def validate_date_ranges(self) -> CandidateCertificationCreate:
        if self.issue_date is not None and self.expiry_date is not None and self.expiry_date < self.issue_date:
            raise ValueError("expiry_date must not be earlier than issue_date")
        return self


class CandidateCertificationRead(BaseModel):
    id: UUID
    profile_id: UUID
    name: str
    issuing_organization: str | None = None
    issue_date: date_type | None = None
    expiry_date: date_type | None = None
    credential_id: str | None = None
    credential_url: AnyUrl | None = None

    model_config = ConfigDict(from_attributes=True)


class CandidateAchievementCreate(BaseModel):
    title: str = Field(..., min_length=1)
    description: str | None = None
    date: date_type | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _validate_non_empty_string(value, "title")

    @field_validator("description")
    @classmethod
    def normalize_optional_string(cls, value: str | None) -> str | None:
        return _strip_optional_string(value)


class CandidateAchievementRead(BaseModel):
    id: UUID
    profile_id: UUID
    title: str
    description: str | None = None
    date: date_type | None = None

    model_config = ConfigDict(from_attributes=True)


class CandidateProfileRead(BaseModel):
    id: UUID
    full_name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: AnyUrl | None = None
    github_url: AnyUrl | None = None
    portfolio_url: AnyUrl | None = None
    professional_summary: str | None = None
    target_roles: str | None = None
    total_experience_years: float | None = None
    created_at: datetime_type
    updated_at: datetime_type
    experiences: list[CandidateExperienceRead] = Field(default_factory=list)
    skills: list[CandidateSkillRead] = Field(default_factory=list)
    education: list[CandidateEducationRead] = Field(
        default_factory=list,
        validation_alias="educations",
        serialization_alias="education",
    )
    projects: list[CandidateProjectRead] = Field(default_factory=list)
    publications: list[CandidatePublicationRead] = Field(default_factory=list)
    certifications: list[CandidateCertificationRead] = Field(default_factory=list)
    achievements: list[CandidateAchievementRead] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
