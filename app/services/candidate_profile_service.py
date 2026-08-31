from __future__ import annotations

import profile
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

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
    CandidateProfileUpdate,
    CandidateProjectCreate,
    CandidatePublicationCreate,
    CandidateSkillCreate,
)


class CandidateProfileError(Exception):
    """Base exception for candidate profile service errors."""


class CandidateProfileNotFoundError(CandidateProfileError):
    """Raised when a candidate profile cannot be found."""


class CandidateExperienceNotFoundError(CandidateProfileError):
    """Raised when a candidate experience cannot be found."""


class CandidateSkillNotFoundError(CandidateProfileError):
    """Raised when a candidate skill cannot be found."""


class CandidateEducationNotFoundError(CandidateProfileError):
    """Raised when a candidate education record cannot be found."""


class CandidateProjectNotFoundError(CandidateProfileError):
    """Raised when a candidate project cannot be found."""


class CandidatePublicationNotFoundError(CandidateProfileError):
    """Raised when a candidate publication cannot be found."""


class CandidateCertificationNotFoundError(CandidateProfileError):
    """Raised when a candidate certification cannot be found."""


class CandidateAchievementNotFoundError(CandidateProfileError):
    """Raised when a candidate achievement cannot be found."""


class CandidateProfileService:
    """Service layer for candidate profile CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_profiles(self) -> list[CandidateProfile]:
        query = (
            select(CandidateProfile)
            .options(
                selectinload(CandidateProfile.experiences),
                selectinload(CandidateProfile.skills),
                selectinload(CandidateProfile.educations),
                selectinload(CandidateProfile.projects),
                selectinload(CandidateProfile.publications),
                selectinload(CandidateProfile.certifications),
                selectinload(CandidateProfile.achievements),
            )
            .order_by(CandidateProfile.created_at.desc())
        )
        result = await self.session.execute(query)
        return list(result.scalars().unique().all())

    # async def create_profile(self, data: CandidateProfileCreate) -> CandidateProfile:
    #     profile = CandidateProfile(**data.model_dump())
    #     self.session.add(profile)
    #     await self.session.commit()
    #     await self.session.refresh(profile)
    #     return await self.get_profile(profile.id)

    async def create_profile(self, data: CandidateProfileCreate) -> CandidateProfile:
        payload = data.model_dump()

        for field_name in ("linkedin_url", "github_url", "portfolio_url"):
            if payload[field_name] is not None:
                payload[field_name] = str(payload[field_name])

        profile = CandidateProfile(**payload)
        self.session.add(profile)
        await self.session.commit()
        await self.session.refresh(profile)
        return await self.get_profile(profile.id)
    

    async def get_profile(self, profile_id: UUID) -> CandidateProfile:
        query = (
            select(CandidateProfile)
            .where(CandidateProfile.id == profile_id)
            .options(
                selectinload(CandidateProfile.experiences),
                selectinload(CandidateProfile.skills),
                selectinload(CandidateProfile.educations),
                selectinload(CandidateProfile.projects),
                selectinload(CandidateProfile.publications),
                selectinload(CandidateProfile.certifications),
                selectinload(CandidateProfile.achievements),
            )
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(query)
        profile = result.scalar_one_or_none()

        if profile is None:
            raise CandidateProfileNotFoundError(f"Candidate profile {profile_id} not found")

        return profile

    async def update_profile(
        self,
        profile_id: UUID,
        data: CandidateProfileUpdate,
    ) -> CandidateProfile:
        profile = await self.get_profile(profile_id)
        payload = data.model_dump(exclude_unset=True)

        for field_name in ("linkedin_url", "github_url", "portfolio_url"):
            if field_name in payload and payload[field_name] is not None:
                payload[field_name] = str(payload[field_name])

        for field_name, value in payload.items():
            setattr(profile, field_name, value)


        # payload = data.model_dump(exclude_unset=True)

        # for field_name, value in payload.items():
        #     setattr(profile, field_name, value)

        await self.session.commit()
        return await self.get_profile(profile_id)

    async def add_experience(
        self,
        profile_id: UUID,
        data: CandidateExperienceCreate,
    ) -> CandidateExperience:
        await self.get_profile(profile_id)
        experience = CandidateExperience(profile_id=profile_id, **data.model_dump())
        self.session.add(experience)
        await self.session.commit()
        await self.session.refresh(experience)
        return experience

    async def update_experience(
        self,
        profile_id: UUID,
        experience_id: UUID,
        data: CandidateExperienceCreate,
    ) -> CandidateExperience:
        experience = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=experience_id,
            model=CandidateExperience,
            error_cls=CandidateExperienceNotFoundError,
        )

        for field_name, value in data.model_dump().items():
            setattr(experience, field_name, value)

        await self.session.commit()
        await self.session.refresh(experience)
        return experience

    async def delete_experience(self, profile_id: UUID, experience_id: UUID) -> None:
        experience = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=experience_id,
            model=CandidateExperience,
            error_cls=CandidateExperienceNotFoundError,
        )
        await self.session.delete(experience)
        await self.session.commit()

    async def add_skill(self, profile_id: UUID, data: CandidateSkillCreate) -> CandidateSkill:
        await self.get_profile(profile_id)
        skill = CandidateSkill(profile_id=profile_id, **data.model_dump())
        self.session.add(skill)
        await self.session.commit()
        await self.session.refresh(skill)
        return skill

    async def update_skill(
        self,
        profile_id: UUID,
        skill_id: UUID,
        data: CandidateSkillCreate,
    ) -> CandidateSkill:
        skill = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=skill_id,
            model=CandidateSkill,
            error_cls=CandidateSkillNotFoundError,
        )

        for field_name, value in data.model_dump().items():
            setattr(skill, field_name, value)

        await self.session.commit()
        await self.session.refresh(skill)
        return skill

    async def delete_skill(self, profile_id: UUID, skill_id: UUID) -> None:
        skill = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=skill_id,
            model=CandidateSkill,
            error_cls=CandidateSkillNotFoundError,
        )
        await self.session.delete(skill)
        await self.session.commit()

    async def add_education(
        self,
        profile_id: UUID,
        data: CandidateEducationCreate,
    ) -> CandidateEducation:
        await self.get_profile(profile_id)
        education = CandidateEducation(profile_id=profile_id, **data.model_dump())
        self.session.add(education)
        await self.session.commit()
        await self.session.refresh(education)
        return education

    async def update_education(
        self,
        profile_id: UUID,
        education_id: UUID,
        data: CandidateEducationCreate,
    ) -> CandidateEducation:
        education = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=education_id,
            model=CandidateEducation,
            error_cls=CandidateEducationNotFoundError,
        )

        for field_name, value in data.model_dump().items():
            setattr(education, field_name, value)

        await self.session.commit()
        await self.session.refresh(education)
        return education

    async def delete_education(self, profile_id: UUID, education_id: UUID) -> None:
        education = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=education_id,
            model=CandidateEducation,
            error_cls=CandidateEducationNotFoundError,
        )
        await self.session.delete(education)
        await self.session.commit()

    async def add_project(
        self,
        profile_id: UUID,
        data: CandidateProjectCreate,
    ) -> CandidateProject:
        await self.get_profile(profile_id)
        project = CandidateProject(profile_id=profile_id, **data.model_dump())
        self.session.add(project)
        await self.session.commit()
        await self.session.refresh(project)
        return project

    async def update_project(
        self,
        profile_id: UUID,
        project_id: UUID,
        data: CandidateProjectCreate,
    ) -> CandidateProject:
        project = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=project_id,
            model=CandidateProject,
            error_cls=CandidateProjectNotFoundError,
        )

        for field_name, value in data.model_dump().items():
            setattr(project, field_name, value)

        await self.session.commit()
        await self.session.refresh(project)
        return project

    async def delete_project(self, profile_id: UUID, project_id: UUID) -> None:
        project = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=project_id,
            model=CandidateProject,
            error_cls=CandidateProjectNotFoundError,
        )
        await self.session.delete(project)
        await self.session.commit()

    async def add_publication(
        self,
        profile_id: UUID,
        data: CandidatePublicationCreate,
    ) -> CandidatePublication:
        await self.get_profile(profile_id)
        publication = CandidatePublication(profile_id=profile_id, **data.model_dump())
        self.session.add(publication)
        await self.session.commit()
        await self.session.refresh(publication)
        return publication

    async def update_publication(
        self,
        profile_id: UUID,
        publication_id: UUID,
        data: CandidatePublicationCreate,
    ) -> CandidatePublication:
        publication = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=publication_id,
            model=CandidatePublication,
            error_cls=CandidatePublicationNotFoundError,
        )

        for field_name, value in data.model_dump().items():
            setattr(publication, field_name, value)

        await self.session.commit()
        await self.session.refresh(publication)
        return publication

    async def delete_publication(self, profile_id: UUID, publication_id: UUID) -> None:
        publication = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=publication_id,
            model=CandidatePublication,
            error_cls=CandidatePublicationNotFoundError,
        )
        await self.session.delete(publication)
        await self.session.commit()

    async def add_certification(
        self,
        profile_id: UUID,
        data: CandidateCertificationCreate,
    ) -> CandidateCertification:
        await self.get_profile(profile_id)
        certification = CandidateCertification(profile_id=profile_id, **data.model_dump())
        self.session.add(certification)
        await self.session.commit()
        await self.session.refresh(certification)
        return certification

    async def update_certification(
        self,
        profile_id: UUID,
        certification_id: UUID,
        data: CandidateCertificationCreate,
    ) -> CandidateCertification:
        certification = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=certification_id,
            model=CandidateCertification,
            error_cls=CandidateCertificationNotFoundError,
        )

        for field_name, value in data.model_dump().items():
            setattr(certification, field_name, value)

        await self.session.commit()
        await self.session.refresh(certification)
        return certification

    async def delete_certification(self, profile_id: UUID, certification_id: UUID) -> None:
        certification = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=certification_id,
            model=CandidateCertification,
            error_cls=CandidateCertificationNotFoundError,
        )
        await self.session.delete(certification)
        await self.session.commit()

    async def add_achievement(
        self,
        profile_id: UUID,
        data: CandidateAchievementCreate,
    ) -> CandidateAchievement:
        await self.get_profile(profile_id)
        achievement = CandidateAchievement(profile_id=profile_id, **data.model_dump())
        self.session.add(achievement)
        await self.session.commit()
        await self.session.refresh(achievement)
        return achievement

    async def update_achievement(
        self,
        profile_id: UUID,
        achievement_id: UUID,
        data: CandidateAchievementCreate,
    ) -> CandidateAchievement:
        achievement = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=achievement_id,
            model=CandidateAchievement,
            error_cls=CandidateAchievementNotFoundError,
        )

        for field_name, value in data.model_dump().items():
            setattr(achievement, field_name, value)

        await self.session.commit()
        await self.session.refresh(achievement)
        return achievement

    async def delete_achievement(self, profile_id: UUID, achievement_id: UUID) -> None:
        achievement = await self._get_child_for_profile(
            profile_id=profile_id,
            resource_id=achievement_id,
            model=CandidateAchievement,
            error_cls=CandidateAchievementNotFoundError,
        )
        await self.session.delete(achievement)
        await self.session.commit()

    async def _get_child_for_profile(
        self,
        *,
        profile_id: UUID,
        resource_id: UUID,
        model: type[CandidateExperience | CandidateSkill | CandidateEducation | CandidateProject | CandidatePublication | CandidateCertification | CandidateAchievement],
        error_cls: type[CandidateProfileError],
    ):
        profile = await self.session.get(CandidateProfile, profile_id)
        if profile is None:
            raise CandidateProfileNotFoundError(f"Candidate profile {profile_id} not found")

        query = select(model).where(model.id == resource_id, model.profile_id == profile_id)
        result = await self.session.execute(query)
        resource = result.scalar_one_or_none()

        if resource is None:
            raise error_cls(f"{model.__name__} {resource_id} not found for candidate profile {profile_id}")

        return resource
