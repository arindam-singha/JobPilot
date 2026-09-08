from datetime import UTC, date, datetime

import pytest
from app.models.candidate_profile import (
    CandidateAchievement,
    CandidateCertification,
    CandidateEducation,
    CandidateExperience,
    CandidateProfile,
    CandidateProject,
    CandidatePublication,
    CandidateSkill,
)
from app.schemas.candidate_profile import (
    CandidateAchievementCreate,
    CandidateCertificationCreate,
    CandidateEducationCreate,
    CandidateExperienceCreate,
    CandidateProfileCreate,
    CandidateProfileRead,
    CandidateProfileUpdate,
    CandidateProjectCreate,
    CandidatePublicationCreate,
    CandidateSkillCreate,
)
from pydantic import ValidationError


def test_valid_candidate_profile_create() -> None:
    profile = CandidateProfileCreate(
        full_name="Jane Doe",
        email="jane@example.com",
        location="Seattle, WA",
        linkedin_url="https://www.linkedin.com/in/janedoe",
        github_url="https://github.com/janedoe",
        portfolio_url="https://janedoe.dev",
        total_experience_years=7.5,
    )

    assert profile.full_name == "Jane Doe"
    assert profile.total_experience_years == 7.5


def test_candidate_profile_full_name_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        CandidateProfileCreate(full_name="   ")


def test_candidate_profile_negative_experience_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateProfileCreate(full_name="Jane Doe", total_experience_years=-1)


def test_valid_candidate_experience_create() -> None:
    experience = CandidateExperienceCreate(
        company="Acme",
        role="Senior Engineer",
        location="Remote",
        start_date=date(2020, 1, 1),
        end_date=date(2023, 1, 1),
        description="Led platform work.",
    )

    assert experience.company == "Acme"
    assert experience.is_current is False


def test_invalid_experience_date_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateExperienceCreate(
            company="Acme",
            role="Senior Engineer",
            start_date=date(2023, 1, 1),
            end_date=date(2022, 1, 1),
        )


def test_current_experience_with_end_date_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateExperienceCreate(
            company="Acme",
            role="Senior Engineer",
            is_current=True,
            end_date=date(2024, 1, 1),
        )


def test_valid_candidate_skill_create() -> None:
    skill = CandidateSkillCreate(
        name="Python",
        category="Languages",
        proficiency="Expert",
        years_of_experience=6.5,
    )

    assert skill.name == "Python"


def test_negative_skill_years_of_experience_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateSkillCreate(name="Python", years_of_experience=-1)


def test_valid_candidate_education_create() -> None:
    education = CandidateEducationCreate(
        institution="University of Washington",
        degree="B.S.",
        field_of_study="Computer Science",
        start_date=date(2015, 9, 1),
        end_date=date(2019, 6, 1),
    )

    assert education.degree == "B.S."


def test_invalid_education_date_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateEducationCreate(
            institution="University of Washington",
            degree="B.S.",
            start_date=date(2020, 1, 1),
            end_date=date(2019, 1, 1),
        )


def test_valid_candidate_project_create() -> None:
    project = CandidateProjectCreate(
        name="JobPilot",
        description="Built an applicant assistant.",
        technologies="Python, FastAPI, SQLAlchemy",
        project_url="https://example.com/project",
    )

    assert project.name == "JobPilot"


def test_invalid_project_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateProjectCreate(name="Project", project_url="not-a-valid-url")


def test_valid_candidate_publication_create() -> None:
    publication = CandidatePublicationCreate(
        title="AI for Hiring",
        venue="ACM",
        publication_date=date(2024, 1, 1),
        url="https://example.com/paper",
    )

    assert publication.title == "AI for Hiring"


def test_valid_candidate_certification_create() -> None:
    certification = CandidateCertificationCreate(
        name="AWS Certified Cloud Practitioner",
        issuing_organization="Amazon Web Services",
        issue_date=date(2024, 1, 1),
        expiry_date=date(2027, 1, 1),
        credential_url="https://example.com/cert",
    )

    assert certification.name == "AWS Certified Cloud Practitioner"


def test_invalid_certification_date_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateCertificationCreate(
            name="Certification",
            issue_date=date(2025, 1, 1),
            expiry_date=date(2024, 1, 1),
        )


def test_valid_candidate_achievement_create() -> None:
    achievement = CandidateAchievementCreate(
        title="Top Performer",
        description="Awarded for team impact.",
        date=date(2024, 5, 1),
    )

    assert achievement.title == "Top Performer"


def test_candidate_profile_update_supports_partial_updates() -> None:
    update = CandidateProfileUpdate(full_name="Updated Name")
    assert update.full_name == "Updated Name"

    with pytest.raises(ValidationError):
        CandidateProfileUpdate(full_name="   ")


def test_candidate_profile_read_can_be_constructed_from_orm_objects() -> None:
    profile = CandidateProfile(
        id=None,
        full_name="Jane Doe",
        email="jane@example.com",
        location="Seattle, WA",
        professional_summary="Experienced platform engineer.",
    )
    profile.id = "123e4567-e89b-12d3-a456-426614174000"
    profile.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    profile.updated_at = datetime(2024, 1, 2, tzinfo=UTC)

    read = CandidateProfileRead.model_validate(profile)

    assert read.full_name == "Jane Doe"
    assert read.email == "jane@example.com"


def test_nested_candidate_profile_read_contains_all_collections() -> None:
    profile = CandidateProfile(
        full_name="Jane Doe",
        email="jane@example.com",
        professional_summary="Experienced engineer.",
    )
    profile.id = "123e4567-e89b-12d3-a456-426614174000"
    profile.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    profile.updated_at = datetime(2024, 1, 2, tzinfo=UTC)
    profile.experiences = [
        CandidateExperience(
            id="123e4567-e89b-12d3-a456-426614174001",
            profile_id=profile.id,
            company="Acme",
            role="Engineer",
            is_current=True,
        )
    ]
    profile.skills = [
        CandidateSkill(
            id="123e4567-e89b-12d3-a456-426614174002",
            profile_id=profile.id,
            name="Python",
        )
    ]
    profile.educations = [
        CandidateEducation(
            id="123e4567-e89b-12d3-a456-426614174003",
            profile_id=profile.id,
            institution="UW",
            degree="BS",
        )
    ]
    profile.projects = [
        CandidateProject(
            id="123e4567-e89b-12d3-a456-426614174004",
            profile_id=profile.id,
            name="JobPilot",
        )
    ]
    profile.publications = [
        CandidatePublication(
            id="123e4567-e89b-12d3-a456-426614174005",
            profile_id=profile.id,
            title="Paper",
        )
    ]
    profile.certifications = [
        CandidateCertification(
            id="123e4567-e89b-12d3-a456-426614174006",
            profile_id=profile.id,
            name="Cert",
        )
    ]
    profile.achievements = [
        CandidateAchievement(
            id="123e4567-e89b-12d3-a456-426614174007",
            profile_id=profile.id,
            title="Award",
        )
    ]

    read = CandidateProfileRead.model_validate(profile)

    assert len(read.experiences) == 1
    assert len(read.skills) == 1
    assert len(read.education) == 1
    assert len(read.projects) == 1
    assert len(read.publications) == 1
    assert len(read.certifications) == 1
    assert len(read.achievements) == 1
