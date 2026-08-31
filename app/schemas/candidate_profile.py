from __future__ import annotations

from datetime import date as date_type, datetime as datetime_type
from typing import Optional
from uuid import UUID

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator, model_validator


def _strip_optional_string(value: Optional[str]) -> Optional[str]:
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
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[AnyUrl] = None
    github_url: Optional[AnyUrl] = None
    portfolio_url: Optional[AnyUrl] = None
    professional_summary: Optional[str] = None
    target_roles: Optional[str] = None
    total_experience_years: Optional[float] = None

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
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        return _strip_optional_string(value)

    @field_validator("total_experience_years")
    @classmethod
    def validate_total_experience_years(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("total_experience_years must be greater than or equal to 0")
        return value


class CandidateProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[AnyUrl] = None
    github_url: Optional[AnyUrl] = None
    portfolio_url: Optional[AnyUrl] = None
    professional_summary: Optional[str] = None
    target_roles: Optional[str] = None
    total_experience_years: Optional[float] = None

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: Optional[str]) -> Optional[str]:
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
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        return _strip_optional_string(value)

    @field_validator("total_experience_years")
    @classmethod
    def validate_total_experience_years(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("total_experience_years must be greater than or equal to 0")
        return value


class CandidateExperienceCreate(BaseModel):
    company: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    location: Optional[str] = None
    start_date: Optional[date_type] = None
    end_date: Optional[date_type] = None
    is_current: bool = False
    description: Optional[str] = None
    achievements: Optional[str] = None

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
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        return _strip_optional_string(value)

    @model_validator(mode="after")
    def validate_date_ranges(self) -> "CandidateExperienceCreate":
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
    location: Optional[str] = None
    start_date: Optional[date_type] = None
    end_date: Optional[date_type] = None
    is_current: bool = False
    description: Optional[str] = None
    achievements: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class CandidateSkillCreate(BaseModel):
    name: str = Field(..., min_length=1)
    category: Optional[str] = None
    proficiency: Optional[str] = None
    years_of_experience: Optional[float] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_non_empty_string(value, "name")

    @field_validator("category", "proficiency")
    @classmethod
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        return _strip_optional_string(value)

    @field_validator("years_of_experience")
    @classmethod
    def validate_years_of_experience(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("years_of_experience must be greater than or equal to 0")
        return value


class CandidateSkillRead(BaseModel):
    id: UUID
    profile_id: UUID
    name: str
    category: Optional[str] = None
    proficiency: Optional[str] = None
    years_of_experience: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class CandidateEducationCreate(BaseModel):
    institution: str = Field(..., min_length=1)
    degree: str = Field(..., min_length=1)
    field_of_study: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[date_type] = None
    end_date: Optional[date_type] = None
    description: Optional[str] = None

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
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        return _strip_optional_string(value)

    @model_validator(mode="after")
    def validate_date_ranges(self) -> "CandidateEducationCreate":
        if self.start_date is not None and self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        return self


class CandidateEducationRead(BaseModel):
    id: UUID
    profile_id: UUID
    institution: str
    degree: str
    field_of_study: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[date_type] = None
    end_date: Optional[date_type] = None
    description: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class CandidateProjectCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    technologies: Optional[str] = None
    achievements: Optional[str] = None
    project_url: Optional[AnyUrl] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_non_empty_string(value, "name")

    @field_validator("description", "technologies", "achievements")
    @classmethod
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        return _strip_optional_string(value)


class CandidateProjectRead(BaseModel):
    id: UUID
    profile_id: UUID
    name: str
    description: Optional[str] = None
    technologies: Optional[str] = None
    achievements: Optional[str] = None
    project_url: Optional[AnyUrl] = None

    model_config = ConfigDict(from_attributes=True)


class CandidatePublicationCreate(BaseModel):
    title: str = Field(..., min_length=1)
    venue: Optional[str] = None
    publication_date: Optional[date_type] = None
    url: Optional[AnyUrl] = None
    description: Optional[str] = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _validate_non_empty_string(value, "title")

    @field_validator("venue", "description")
    @classmethod
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        return _strip_optional_string(value)


class CandidatePublicationRead(BaseModel):
    id: UUID
    profile_id: UUID
    title: str
    venue: Optional[str] = None
    publication_date: Optional[date_type] = None
    url: Optional[AnyUrl] = None
    description: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class CandidateCertificationCreate(BaseModel):
    name: str = Field(..., min_length=1)
    issuing_organization: Optional[str] = None
    issue_date: Optional[date_type] = None
    expiry_date: Optional[date_type] = None
    credential_id: Optional[str] = None
    credential_url: Optional[AnyUrl] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_non_empty_string(value, "name")

    @field_validator("issuing_organization", "credential_id")
    @classmethod
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        return _strip_optional_string(value)

    @model_validator(mode="after")
    def validate_date_ranges(self) -> "CandidateCertificationCreate":
        if self.issue_date is not None and self.expiry_date is not None and self.expiry_date < self.issue_date:
            raise ValueError("expiry_date must not be earlier than issue_date")
        return self


class CandidateCertificationRead(BaseModel):
    id: UUID
    profile_id: UUID
    name: str
    issuing_organization: Optional[str] = None
    issue_date: Optional[date_type] = None
    expiry_date: Optional[date_type] = None
    credential_id: Optional[str] = None
    credential_url: Optional[AnyUrl] = None

    model_config = ConfigDict(from_attributes=True)


class CandidateAchievementCreate(BaseModel):
    title: str = Field(..., min_length=1)
    description: Optional[str] = None
    date: Optional[date_type] = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _validate_non_empty_string(value, "title")

    @field_validator("description")
    @classmethod
    def normalize_optional_string(cls, value: Optional[str]) -> Optional[str]:
        return _strip_optional_string(value)


class CandidateAchievementRead(BaseModel):
    id: UUID
    profile_id: UUID
    title: str
    description: Optional[str] = None
    date: Optional[date_type] = None

    model_config = ConfigDict(from_attributes=True)


class CandidateProfileRead(BaseModel):
    id: UUID
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[AnyUrl] = None
    github_url: Optional[AnyUrl] = None
    portfolio_url: Optional[AnyUrl] = None
    professional_summary: Optional[str] = None
    target_roles: Optional[str] = None
    total_experience_years: Optional[float] = None
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
