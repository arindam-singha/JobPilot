from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.candidate_profile import (
    CandidateAchievementCreate,
    CandidateAchievementRead,
    CandidateCertificationCreate,
    CandidateCertificationRead,
    CandidateEducationCreate,
    CandidateEducationRead,
    CandidateExperienceCreate,
    CandidateExperienceRead,
    CandidateProfileCreate,
    CandidateProfileRead,
    CandidateProfileUpdate,
    CandidateProjectCreate,
    CandidateProjectRead,
    CandidatePublicationCreate,
    CandidatePublicationRead,
    CandidateSkillCreate,
    CandidateSkillRead,
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

router = APIRouter(prefix="/api/v1/candidate-profile", tags=["candidate-profile"])


def _raise_profile_not_found() -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate profile not found")


def _raise_child_not_found(resource_name: str) -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{resource_name} not found")


@router.post(
    "",
    response_model=CandidateProfileRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_profile(
    payload: CandidateProfileCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateProfileRead:
    service = CandidateProfileService(db)
    profile = await service.create_profile(payload)
    return CandidateProfileRead.model_validate(profile)


@router.get(
    "",
    response_model=list[CandidateProfileRead],
)
async def list_profiles(
    db: AsyncSession = Depends(get_db),
) -> list[CandidateProfileRead]:
    profiles = await CandidateProfileService(db).list_profiles()
    return [CandidateProfileRead.model_validate(profile) for profile in profiles]


@router.get(
    "/{profile_id}",
    response_model=CandidateProfileRead,
)
async def get_profile(
    profile_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> CandidateProfileRead:
    service = CandidateProfileService(db)
    try:
        profile = await service.get_profile(profile_id)
    except CandidateProfileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate profile not found") from exc

    return CandidateProfileRead.model_validate(profile)


@router.patch(
    "/{profile_id}",
    response_model=CandidateProfileRead,
)
async def update_profile(
    profile_id: UUID,
    payload: CandidateProfileUpdate,
    db: AsyncSession = Depends(get_db),
) -> CandidateProfileRead:
    service = CandidateProfileService(db)
    try:
        profile = await service.update_profile(profile_id, payload)
    except CandidateProfileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate profile not found") from exc

    return CandidateProfileRead.model_validate(profile)


@router.post(
    "/{profile_id}/experiences",
    response_model=CandidateExperienceRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_experience(
    profile_id: UUID,
    payload: CandidateExperienceCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateExperienceRead:
    service = CandidateProfileService(db)
    try:
        experience = await service.add_experience(profile_id, payload)
    except CandidateProfileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate profile not found") from exc

    return CandidateExperienceRead.model_validate(experience)


@router.patch(
    "/{profile_id}/experiences/{experience_id}",
    response_model=CandidateExperienceRead,
)
async def update_experience(
    profile_id: UUID,
    experience_id: UUID,
    payload: CandidateExperienceCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateExperienceRead:
    service = CandidateProfileService(db)
    try:
        experience = await service.update_experience(profile_id, experience_id, payload)
    except (CandidateProfileNotFoundError, CandidateExperienceNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate experience not found") from exc

    return CandidateExperienceRead.model_validate(experience)


# @router.delete(
#     "/{profile_id}/experiences/{experience_id}",
#     status_code=status.HTTP_204_NO_CONTENT,
#     response_class=None,
# )

@router.delete(
    "/{profile_id}/experiences/{experience_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_experience(
    profile_id: UUID,
    experience_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    service = CandidateProfileService(db)
    try:
        await service.delete_experience(profile_id, experience_id)
    except (CandidateProfileNotFoundError, CandidateExperienceNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate experience not found") from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{profile_id}/skills",
    response_model=CandidateSkillRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_skill(
    profile_id: UUID,
    payload: CandidateSkillCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateSkillRead:
    service = CandidateProfileService(db)
    try:
        skill = await service.add_skill(profile_id, payload)
    except CandidateProfileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate profile not found") from exc

    return CandidateSkillRead.model_validate(skill)


@router.patch(
    "/{profile_id}/skills/{skill_id}",
    response_model=CandidateSkillRead,
)
async def update_skill(
    profile_id: UUID,
    skill_id: UUID,
    payload: CandidateSkillCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateSkillRead:
    service = CandidateProfileService(db)
    try:
        skill = await service.update_skill(profile_id, skill_id, payload)
    except (CandidateProfileNotFoundError, CandidateSkillNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate skill not found") from exc

    return CandidateSkillRead.model_validate(skill)


@router.delete(
    "/{profile_id}/skills/{skill_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_skill(
    profile_id: UUID,
    skill_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    service = CandidateProfileService(db)
    try:
        await service.delete_skill(profile_id, skill_id)
    except (CandidateProfileNotFoundError, CandidateSkillNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate skill not found") from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{profile_id}/education",
    response_model=CandidateEducationRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_education(
    profile_id: UUID,
    payload: CandidateEducationCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateEducationRead:
    service = CandidateProfileService(db)
    try:
        education = await service.add_education(profile_id, payload)
    except CandidateProfileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate profile not found") from exc

    return CandidateEducationRead.model_validate(education)


@router.patch(
    "/{profile_id}/education/{education_id}",
    response_model=CandidateEducationRead,
)
async def update_education(
    profile_id: UUID,
    education_id: UUID,
    payload: CandidateEducationCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateEducationRead:
    service = CandidateProfileService(db)
    try:
        education = await service.update_education(profile_id, education_id, payload)
    except (CandidateProfileNotFoundError, CandidateEducationNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate education not found") from exc

    return CandidateEducationRead.model_validate(education)


@router.delete(
    "/{profile_id}/education/{education_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_education(
    profile_id: UUID,
    education_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    service = CandidateProfileService(db)
    try:
        await service.delete_education(profile_id, education_id)
    except (CandidateProfileNotFoundError, CandidateEducationNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate education not found") from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{profile_id}/projects",
    response_model=CandidateProjectRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_project(
    profile_id: UUID,
    payload: CandidateProjectCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateProjectRead:
    service = CandidateProfileService(db)
    try:
        project = await service.add_project(profile_id, payload)
    except CandidateProfileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate profile not found") from exc

    return CandidateProjectRead.model_validate(project)


@router.patch(
    "/{profile_id}/projects/{project_id}",
    response_model=CandidateProjectRead,
)
async def update_project(
    profile_id: UUID,
    project_id: UUID,
    payload: CandidateProjectCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateProjectRead:
    service = CandidateProfileService(db)
    try:
        project = await service.update_project(profile_id, project_id, payload)
    except (CandidateProfileNotFoundError, CandidateProjectNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate project not found") from exc

    return CandidateProjectRead.model_validate(project)


@router.delete(
    "/{profile_id}/projects/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_project(
    profile_id: UUID,
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    service = CandidateProfileService(db)
    try:
        await service.delete_project(profile_id, project_id)
    except (CandidateProfileNotFoundError, CandidateProjectNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate project not found") from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{profile_id}/publications",
    response_model=CandidatePublicationRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_publication(
    profile_id: UUID,
    payload: CandidatePublicationCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidatePublicationRead:
    service = CandidateProfileService(db)
    try:
        publication = await service.add_publication(profile_id, payload)
    except CandidateProfileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate profile not found") from exc

    return CandidatePublicationRead.model_validate(publication)


@router.patch(
    "/{profile_id}/publications/{publication_id}",
    response_model=CandidatePublicationRead,
)
async def update_publication(
    profile_id: UUID,
    publication_id: UUID,
    payload: CandidatePublicationCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidatePublicationRead:
    service = CandidateProfileService(db)
    try:
        publication = await service.update_publication(profile_id, publication_id, payload)
    except (CandidateProfileNotFoundError, CandidatePublicationNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate publication not found") from exc

    return CandidatePublicationRead.model_validate(publication)


@router.delete(
    "/{profile_id}/publications/{publication_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_publication(
    profile_id: UUID,
    publication_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    service = CandidateProfileService(db)
    try:
        await service.delete_publication(profile_id, publication_id)
    except (CandidateProfileNotFoundError, CandidatePublicationNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate publication not found") from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{profile_id}/certifications",
    response_model=CandidateCertificationRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_certification(
    profile_id: UUID,
    payload: CandidateCertificationCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateCertificationRead:
    service = CandidateProfileService(db)
    try:
        certification = await service.add_certification(profile_id, payload)
    except CandidateProfileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate profile not found") from exc

    return CandidateCertificationRead.model_validate(certification)


@router.patch(
    "/{profile_id}/certifications/{certification_id}",
    response_model=CandidateCertificationRead,
)
async def update_certification(
    profile_id: UUID,
    certification_id: UUID,
    payload: CandidateCertificationCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateCertificationRead:
    service = CandidateProfileService(db)
    try:
        certification = await service.update_certification(profile_id, certification_id, payload)
    except (CandidateProfileNotFoundError, CandidateCertificationNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate certification not found") from exc

    return CandidateCertificationRead.model_validate(certification)


@router.delete(
    "/{profile_id}/certifications/{certification_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_certification(
    profile_id: UUID,
    certification_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    service = CandidateProfileService(db)
    try:
        await service.delete_certification(profile_id, certification_id)
    except (CandidateProfileNotFoundError, CandidateCertificationNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate certification not found") from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{profile_id}/achievements",
    response_model=CandidateAchievementRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_achievement(
    profile_id: UUID,
    payload: CandidateAchievementCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateAchievementRead:
    service = CandidateProfileService(db)
    try:
        achievement = await service.add_achievement(profile_id, payload)
    except CandidateProfileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate profile not found") from exc

    return CandidateAchievementRead.model_validate(achievement)


@router.patch(
    "/{profile_id}/achievements/{achievement_id}",
    response_model=CandidateAchievementRead,
)
async def update_achievement(
    profile_id: UUID,
    achievement_id: UUID,
    payload: CandidateAchievementCreate,
    db: AsyncSession = Depends(get_db),
) -> CandidateAchievementRead:
    service = CandidateProfileService(db)
    try:
        achievement = await service.update_achievement(profile_id, achievement_id, payload)
    except (CandidateProfileNotFoundError, CandidateAchievementNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate achievement not found") from exc

    return CandidateAchievementRead.model_validate(achievement)


@router.delete(
    "/{profile_id}/achievements/{achievement_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_achievement(
    profile_id: UUID,
    achievement_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    service = CandidateProfileService(db)
    try:
        await service.delete_achievement(profile_id, achievement_id)
    except (CandidateProfileNotFoundError, CandidateAchievementNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate achievement not found") from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
