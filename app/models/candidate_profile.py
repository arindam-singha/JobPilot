from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import Date, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.candidate_evidence import CandidateEvidence

from app.db.base import Base

if TYPE_CHECKING:
    from collections.abc import Iterable

    from app.models.candidate_document import CandidateDocument


class CandidateProfile(Base):
    """Persisted candidate profile."""

    __tablename__ = "candidate_profiles"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )
    full_name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str | None] = mapped_column(String, nullable=True)
    phone: Mapped[str | None] = mapped_column(String, nullable=True)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    linkedin_url: Mapped[str | None] = mapped_column(String, nullable=True)
    github_url: Mapped[str | None] = mapped_column(String, nullable=True)
    portfolio_url: Mapped[str | None] = mapped_column(String, nullable=True)
    professional_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_roles: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_experience_years: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    experiences: Mapped[list[CandidateExperience]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        order_by="CandidateExperience.id",
    )
    skills: Mapped[list[CandidateSkill]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        order_by="CandidateSkill.id",
    )
    educations: Mapped[list[CandidateEducation]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        order_by="CandidateEducation.id",
    )
    projects: Mapped[list[CandidateProject]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        order_by="CandidateProject.id",
    )
    publications: Mapped[list[CandidatePublication]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        order_by="CandidatePublication.id",
    )
    certifications: Mapped[list[CandidateCertification]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        order_by="CandidateCertification.id",
    )
    achievements: Mapped[list[CandidateAchievement]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        order_by="CandidateAchievement.id",
    )
    documents: Mapped[list["CandidateDocument"]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        order_by="CandidateDocument.id",
    )
    evidence: Mapped[list["CandidateEvidence"]] = relationship(
    back_populates="profile",
    cascade="all, delete-orphan",
    order_by="CandidateEvidence.id",
    )


class CandidateExperience(Base):
    """Professional experience for a candidate."""

    __tablename__ = "candidate_experiences"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )
    profile_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    company: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_current: Mapped[bool] = mapped_column(default=False, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    achievements: Mapped[str | None] = mapped_column(Text, nullable=True)

    profile: Mapped[CandidateProfile] = relationship(back_populates="experiences")


class CandidateSkill(Base):
    """Skill entry for a candidate."""

    __tablename__ = "candidate_skills"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )
    profile_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    proficiency: Mapped[str | None] = mapped_column(String, nullable=True)
    years_of_experience: Mapped[float | None] = mapped_column(nullable=True)

    profile: Mapped[CandidateProfile] = relationship(back_populates="skills")


class CandidateEducation(Base):
    """Educational background for a candidate."""

    __tablename__ = "candidate_education"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )
    profile_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    institution: Mapped[str] = mapped_column(String, nullable=False)
    degree: Mapped[str] = mapped_column(String, nullable=False)
    field_of_study: Mapped[str | None] = mapped_column(String, nullable=True)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    profile: Mapped[CandidateProfile] = relationship(back_populates="educations")


class CandidateProject(Base):
    """Project for a candidate."""

    __tablename__ = "candidate_projects"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )
    profile_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    technologies: Mapped[str | None] = mapped_column(Text, nullable=True)
    achievements: Mapped[str | None] = mapped_column(Text, nullable=True)
    project_url: Mapped[str | None] = mapped_column(String, nullable=True)

    profile: Mapped[CandidateProfile] = relationship(back_populates="projects")


class CandidatePublication(Base):
    """Publication for a candidate."""

    __tablename__ = "candidate_publications"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )
    profile_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    venue: Mapped[str | None] = mapped_column(String, nullable=True)
    publication_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    profile: Mapped[CandidateProfile] = relationship(back_populates="publications")


class CandidateCertification(Base):
    """Certification for a candidate."""

    __tablename__ = "candidate_certifications"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )
    profile_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    issuing_organization: Mapped[str | None] = mapped_column(String, nullable=True)
    issue_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    expiry_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    credential_id: Mapped[str | None] = mapped_column(String, nullable=True)
    credential_url: Mapped[str | None] = mapped_column(String, nullable=True)

    profile: Mapped[CandidateProfile] = relationship(back_populates="certifications")


class CandidateAchievement(Base):
    """Achievement for a candidate."""

    __tablename__ = "candidate_achievements"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )
    profile_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    date: Mapped[date | None] = mapped_column(Date, nullable=True)

    profile: Mapped[CandidateProfile] = relationship(back_populates="achievements")
