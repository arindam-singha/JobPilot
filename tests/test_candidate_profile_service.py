from datetime import date
from uuid import uuid4

import pytest
from app.models.candidate_profile import (
    CandidateExperience,
)
from app.schemas.candidate_profile import (
    CandidateAchievementCreate,
    CandidateCertificationCreate,
    CandidateEducationCreate,
    CandidateExperienceCreate,
    CandidateProfileCreate,
    CandidateProfileUpdate,
    CandidateProjectCreate,
    CandidatePublicationCreate,
    CandidateSkillCreate,
)
from app.services.candidate_profile_service import (
    CandidateAchievementNotFoundError,
    CandidateCertificationNotFoundError,
    CandidateEducationNotFoundError,
    CandidateExperienceNotFoundError,
    CandidateProfileNotFoundError,
    CandidateProfileService,
    CandidateProjectNotFoundError,
    CandidatePublicationNotFoundError,
    CandidateSkillNotFoundError,
)


@pytest.mark.asyncio
async def test_create_profile_success(database_session) -> None:
    service = CandidateProfileService(database_session)
    payload = CandidateProfileCreate(
        full_name="Jane Doe",
        email="jane@example.com",
        location="Seattle, WA",
        professional_summary="Experienced engineer.",
    )

    profile = await service.create_profile(payload)

    assert profile.id is not None
    assert profile.full_name == "Jane Doe"
    assert profile.email == "jane@example.com"


@pytest.mark.asyncio
async def test_get_profile_returns_loaded_profile(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(
        CandidateProfileCreate(
            full_name="John Smith",
            email="john@example.com",
        )
    )

    other = await service.get_profile(profile.id)

    assert other.id == profile.id
    assert other.experiences == []
    assert other.skills == []
    assert other.educations == []
    assert other.projects == []
    assert other.publications == []
    assert other.certifications == []
    assert other.achievements == []


@pytest.mark.asyncio
async def test_get_profile_eager_loads_collections(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Ada Lovelace"))

    await service.add_experience(profile.id, CandidateExperienceCreate(company="Acme", role="Engineer"))
    await service.add_skill(profile.id, CandidateSkillCreate(name="Python"))
    await service.add_education(
        profile.id,
        CandidateEducationCreate(institution="Oxford", degree="MSc", start_date=date(2010, 1, 1)),
    )
    await service.add_project(profile.id, CandidateProjectCreate(name="Orbital"))
    await service.add_publication(
        profile.id,
        CandidatePublicationCreate(title="On Computing", publication_date=date(2020, 1, 1)),
    )
    await service.add_certification(
        profile.id,
        CandidateCertificationCreate(name="Certified Engineer", issue_date=date(2021, 1, 1)),
    )
    await service.add_achievement(
        profile.id,
        CandidateAchievementCreate(title="Top Performer", date=date(2024, 1, 1)),
    )

    loaded = await service.get_profile(profile.id)

    assert len(loaded.experiences) == 1
    assert len(loaded.skills) == 1
    assert len(loaded.educations) == 1
    assert len(loaded.projects) == 1
    assert len(loaded.publications) == 1
    assert len(loaded.certifications) == 1
    assert len(loaded.achievements) == 1


@pytest.mark.asyncio
async def test_get_profile_raises_for_unknown_uuid(database_session) -> None:
    service = CandidateProfileService(database_session)

    with pytest.raises(CandidateProfileNotFoundError):
        await service.get_profile(uuid4())


@pytest.mark.asyncio
async def test_update_profile_changes_only_supplied_fields(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(
        CandidateProfileCreate(full_name="Jane Doe", email="jane@example.com", location="Seattle")
    )

    updated = await service.update_profile(
        profile.id,
        CandidateProfileUpdate(full_name="Jane Updated", total_experience_years=8.5),
    )

    assert updated.full_name == "Jane Updated"
    assert updated.email == "jane@example.com"
    assert updated.location == "Seattle"
    assert updated.total_experience_years == 8.5


@pytest.mark.asyncio
async def test_add_experience_creates_record(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))

    experience = await service.add_experience(
        profile.id,
        CandidateExperienceCreate(
            company="Acme",
            role="Engineer",
            start_date=date(2022, 1, 1),
            end_date=date(2024, 1, 1),
        ),
    )

    assert experience.profile_id == profile.id
    assert experience.company == "Acme"


@pytest.mark.asyncio
async def test_update_experience_updates_correct_record(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    experience = await service.add_experience(
        profile.id,
        CandidateExperienceCreate(company="OldCo", role="Engineer"),
    )

    updated = await service.update_experience(
        profile.id,
        experience.id,
        CandidateExperienceCreate(company="NewCo", role="Senior Engineer"),
    )

    assert updated.company == "NewCo"
    assert updated.role == "Senior Engineer"


@pytest.mark.asyncio
async def test_delete_experience_removes_record(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    experience = await service.add_experience(
        profile.id,
        CandidateExperienceCreate(company="Acme", role="Engineer"),
    )

    await service.delete_experience(profile.id, experience.id)

    with pytest.raises(CandidateExperienceNotFoundError):
        await service.update_experience(
            profile.id,
            experience.id,
            CandidateExperienceCreate(company="Acme", role="Manager"),
        )


@pytest.mark.asyncio
async def test_cross_profile_experience_access_is_rejected(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile_a = await service.create_profile(CandidateProfileCreate(full_name="A"))
    profile_b = await service.create_profile(CandidateProfileCreate(full_name="B"))
    exp = await service.add_experience(
        profile_b.id,
        CandidateExperienceCreate(company="Other", role="Engineer"),
    )

    with pytest.raises(CandidateExperienceNotFoundError):
        await service.update_experience(profile_a.id, exp.id, CandidateExperienceCreate(company="X", role="Y"))

    with pytest.raises(CandidateExperienceNotFoundError):
        await service.delete_experience(profile_a.id, exp.id)


@pytest.mark.asyncio
async def test_add_skill_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))

    skill = await service.add_skill(profile.id, CandidateSkillCreate(name="Python", years_of_experience=4.5))

    assert skill.profile_id == profile.id
    assert skill.name == "Python"


@pytest.mark.asyncio
async def test_update_skill_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    skill = await service.add_skill(profile.id, CandidateSkillCreate(name="Python"))

    updated = await service.update_skill(
        profile.id,
        skill.id,
        CandidateSkillCreate(name="Python", proficiency="Expert", years_of_experience=6.0),
    )

    assert updated.proficiency == "Expert"
    assert updated.years_of_experience == 6.0


@pytest.mark.asyncio
async def test_delete_skill_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    skill = await service.add_skill(profile.id, CandidateSkillCreate(name="Python"))

    await service.delete_skill(profile.id, skill.id)

    with pytest.raises(CandidateSkillNotFoundError):
        await service.update_skill(profile.id, skill.id, CandidateSkillCreate(name="Java"))


@pytest.mark.asyncio
async def test_add_education_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))

    education = await service.add_education(
        profile.id,
        CandidateEducationCreate(
            institution="University of Washington",
            degree="B.S.",
            field_of_study="Computer Science",
        ),
    )

    assert education.profile_id == profile.id
    assert education.institution == "University of Washington"


@pytest.mark.asyncio
async def test_update_education_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    education = await service.add_education(
        profile.id,
        CandidateEducationCreate(institution="Old School", degree="BA"),
    )

    updated = await service.update_education(
        profile.id,
        education.id,
        CandidateEducationCreate(institution="New School", degree="BS"),
    )

    assert updated.institution == "New School"
    assert updated.degree == "BS"


@pytest.mark.asyncio
async def test_delete_education_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    education = await service.add_education(
        profile.id,
        CandidateEducationCreate(institution="University", degree="B.S."),
    )

    await service.delete_education(profile.id, education.id)

    with pytest.raises(CandidateEducationNotFoundError):
        await service.update_education(profile.id, education.id, CandidateEducationCreate(institution="Other", degree="MS"))


@pytest.mark.asyncio
async def test_add_project_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))

    project = await service.add_project(profile.id, CandidateProjectCreate(name="JobPilot"))

    assert project.profile_id == profile.id
    assert project.name == "JobPilot"


@pytest.mark.asyncio
async def test_update_project_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    project = await service.add_project(profile.id, CandidateProjectCreate(name="Old Project"))

    updated = await service.update_project(
        profile.id,
        project.id,
        CandidateProjectCreate(name="New Project", description="Updated"),
    )

    assert updated.name == "New Project"
    assert updated.description == "Updated"


@pytest.mark.asyncio
async def test_delete_project_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    project = await service.add_project(profile.id, CandidateProjectCreate(name="Project"))

    await service.delete_project(profile.id, project.id)

    with pytest.raises(CandidateProjectNotFoundError):
        await service.update_project(profile.id, project.id, CandidateProjectCreate(name="Other"))


@pytest.mark.asyncio
async def test_add_publication_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))

    publication = await service.add_publication(
        profile.id,
        CandidatePublicationCreate(title="A Paper", venue="ACM"),
    )

    assert publication.profile_id == profile.id
    assert publication.title == "A Paper"


@pytest.mark.asyncio
async def test_update_publication_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    publication = await service.add_publication(
        profile.id,
        CandidatePublicationCreate(title="Old Paper"),
    )

    updated = await service.update_publication(
        profile.id,
        publication.id,
        CandidatePublicationCreate(title="New Paper", venue="IEEE"),
    )

    assert updated.title == "New Paper"
    assert updated.venue == "IEEE"


@pytest.mark.asyncio
async def test_delete_publication_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    publication = await service.add_publication(profile.id, CandidatePublicationCreate(title="Paper"))

    await service.delete_publication(profile.id, publication.id)

    with pytest.raises(CandidatePublicationNotFoundError):
        await service.update_publication(profile.id, publication.id, CandidatePublicationCreate(title="Other"))


@pytest.mark.asyncio
async def test_add_certification_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))

    certification = await service.add_certification(
        profile.id,
        CandidateCertificationCreate(name="AWS", issue_date=date(2024, 1, 1)),
    )

    assert certification.profile_id == profile.id
    assert certification.name == "AWS"


@pytest.mark.asyncio
async def test_update_certification_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    cert = await service.add_certification(profile.id, CandidateCertificationCreate(name="Old Cert"))

    updated = await service.update_certification(
        profile.id,
        cert.id,
        CandidateCertificationCreate(name="New Cert", issuing_organization="AWS"),
    )

    assert updated.name == "New Cert"
    assert updated.issuing_organization == "AWS"


@pytest.mark.asyncio
async def test_delete_certification_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    cert = await service.add_certification(profile.id, CandidateCertificationCreate(name="Cert"))

    await service.delete_certification(profile.id, cert.id)

    with pytest.raises(CandidateCertificationNotFoundError):
        await service.update_certification(profile.id, cert.id, CandidateCertificationCreate(name="Other"))


@pytest.mark.asyncio
async def test_add_achievement_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))

    achievement = await service.add_achievement(
        profile.id,
        CandidateAchievementCreate(title="Top Performer", date=date(2024, 1, 1)),
    )

    assert achievement.profile_id == profile.id
    assert achievement.title == "Top Performer"


@pytest.mark.asyncio
async def test_update_achievement_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    achievement = await service.add_achievement(profile.id, CandidateAchievementCreate(title="Old Award"))

    updated = await service.update_achievement(
        profile.id,
        achievement.id,
        CandidateAchievementCreate(title="New Award", description="Updated"),
    )

    assert updated.title == "New Award"
    assert updated.description == "Updated"


@pytest.mark.asyncio
async def test_delete_achievement_works(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    achievement = await service.add_achievement(profile.id, CandidateAchievementCreate(title="Award"))

    await service.delete_achievement(profile.id, achievement.id)

    with pytest.raises(CandidateAchievementNotFoundError):
        await service.update_achievement(profile.id, achievement.id, CandidateAchievementCreate(title="Other"))


@pytest.mark.asyncio
async def test_profile_delete_cascades_to_children(database_session) -> None:
    service = CandidateProfileService(database_session)
    profile = await service.create_profile(CandidateProfileCreate(full_name="Cascade User"))
    await service.add_experience(profile.id, CandidateExperienceCreate(company="Acme", role="Engineer"))
    await service.add_skill(profile.id, CandidateSkillCreate(name="Python"))

    await database_session.delete(profile)
    await database_session.commit()

    assert await database_session.get(CandidateExperience, uuid4()) is None
