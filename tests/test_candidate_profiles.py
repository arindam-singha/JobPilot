import pytest
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models import CandidateProfile, CandidateExperience, CandidateSkill


@pytest.mark.asyncio
async def test_candidate_profile_persists_with_related_records(database_session) -> None:
    profile = CandidateProfile(
        full_name="Jane Doe",
        email="jane@example.com",
        location="Seattle, WA",
        professional_summary="Experienced platform engineer.",
    )
    experience = CandidateExperience(
        company="Acme Corp",
        role="Senior Engineer",
        location="Seattle, WA",
        is_current=True,
        profile=profile,
    )
    skill = CandidateSkill(
        name="Python",
        category="Languages",
        proficiency="Expert",
        years_of_experience=6.5,
        profile=profile,
    )

    database_session.add_all([profile, experience, skill])
    await database_session.commit()
    await database_session.refresh(profile)

    related = await database_session.execute(
        select(CandidateProfile)
        .where(CandidateProfile.email == "jane@example.com")
        .options(selectinload(CandidateProfile.experiences), selectinload(CandidateProfile.skills))
    )
    saved_profile = related.scalar_one()

    assert saved_profile.id is not None
    assert len(saved_profile.experiences) == 1
    assert saved_profile.experiences[0].company == "Acme Corp"
    assert len(saved_profile.skills) == 1
    assert saved_profile.skills[0].name == "Python"
    assert saved_profile.full_name == "Jane Doe"
