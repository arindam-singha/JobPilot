from datetime import UTC, datetime
from uuid import uuid4

import pytest
from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import CandidateProfile
from app.schemas.candidate_evidence import (
    CandidateEvidenceCreate,
    CandidateEvidenceRead,
    CandidateEvidenceUpdate,
)
from app.schemas.candidate_profile import CandidateProfileCreate
from app.services.candidate_evidence_service import (
    CandidateEvidenceNotFoundError,
    CandidateEvidenceService,
)
from app.services.candidate_profile_service import (
    CandidateProfileNotFoundError,
    CandidateProfileService,
)
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import selectinload


def evidence_payload(**overrides):
    payload = {
        "evidence_type": "experience",
        "title": "Python delivery",
        "content": "Built an async service.",
        "source_type": "profile",
        "metadata_json": "  {\"confidence\": 0.9}  ",
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("field", ["evidence_type", "title", "content", "source_type"])
def test_evidence_create_rejects_empty_required_strings(field) -> None:
    payload = evidence_payload(**{field: ""})
    with pytest.raises(ValidationError):
        CandidateEvidenceCreate.model_validate(payload)


@pytest.mark.parametrize("field", ["evidence_type", "title", "content", "source_type"])
def test_evidence_create_rejects_whitespace_required_strings(field) -> None:
    payload = evidence_payload(**{field: "  \t"})
    with pytest.raises(ValidationError):
        CandidateEvidenceCreate.model_validate(payload)


def test_evidence_schema_normalizes_metadata_and_required_strings() -> None:
    payload = CandidateEvidenceCreate.model_validate(evidence_payload(title="  Title  "))
    assert payload.title == "Title"
    assert payload.metadata_json == '{"confidence": 0.9}'
    assert CandidateEvidenceCreate.model_validate(evidence_payload(metadata_json="  ")).metadata_json is None


def test_evidence_read_represents_orm_object() -> None:
    evidence = CandidateEvidence(
        id=uuid4(),
        profile_id=uuid4(),
        evidence_type="skill",
        title="Python",
        content="Advanced Python",
        source_type="profile",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    read = CandidateEvidenceRead.model_validate(evidence)
    assert read.id == evidence.id
    assert read.profile_id == evidence.profile_id
    assert read.title == "Python"


def test_evidence_update_accepts_partial_data() -> None:
    update = CandidateEvidenceUpdate(title="Updated")
    assert update.model_dump(exclude_unset=True) == {"title": "Updated"}


async def create_profile(session, name="Jane Doe"):
    return await CandidateProfileService(session).create_profile(
        CandidateProfileCreate(full_name=name)
    )


@pytest.mark.asyncio
async def test_evidence_service_crud_and_partial_update(database_session, async_session_maker) -> None:
    profile = await create_profile(database_session)
    service = CandidateEvidenceService(database_session)
    created = await service.create_evidence(
        profile.id, CandidateEvidenceCreate.model_validate(evidence_payload())
    )

    assert created.profile_id == profile.id
    assert (await service.list_evidence(profile.id)) == [created]
    retrieved = await service.get_evidence(profile.id, created.id)
    updated = await service.update_evidence(
        profile.id, created.id, CandidateEvidenceUpdate(title="Updated title")
    )
    assert retrieved.id == created.id
    assert updated.title == "Updated title"
    assert updated.content == created.content

    async with async_session_maker() as new_session:
        persisted = await CandidateEvidenceService(new_session).get_evidence(profile.id, created.id)
        assert persisted.id == created.id

    await service.delete_evidence(profile.id, created.id)
    with pytest.raises(CandidateEvidenceNotFoundError):
        await service.get_evidence(profile.id, created.id)


@pytest.mark.asyncio
async def test_evidence_service_rejects_unknown_and_cross_profile_access(database_session) -> None:
    profile_a = await create_profile(database_session, "A")
    profile_b = await create_profile(database_session, "B")
    service = CandidateEvidenceService(database_session)
    with pytest.raises(CandidateProfileNotFoundError):
        await service.create_evidence(uuid4(), CandidateEvidenceCreate.model_validate(evidence_payload()))

    evidence = await service.create_evidence(
        profile_b.id, CandidateEvidenceCreate.model_validate(evidence_payload())
    )
    with pytest.raises(CandidateEvidenceNotFoundError):
        await service.get_evidence(profile_a.id, evidence.id)
    with pytest.raises(CandidateEvidenceNotFoundError):
        await service.update_evidence(profile_a.id, evidence.id, CandidateEvidenceUpdate(title="No"))
    with pytest.raises(CandidateEvidenceNotFoundError):
        await service.delete_evidence(profile_a.id, evidence.id)


@pytest.mark.asyncio
async def test_evidence_relationship_and_profile_cascade(database_session) -> None:
    profile = await create_profile(database_session)
    evidence = await CandidateEvidenceService(database_session).create_evidence(
        profile.id, CandidateEvidenceCreate.model_validate(evidence_payload())
    )
    result = await database_session.execute(
        select(CandidateProfile)
        .options(selectinload(CandidateProfile.evidence))
        .where(CandidateProfile.id == profile.id)
    )
    loaded_profile = result.scalar_one()
    assert loaded_profile.evidence[0].id == evidence.id

    await database_session.delete(loaded_profile)
    await database_session.commit()
    assert await database_session.get(CandidateEvidence, evidence.id) is None


@pytest.mark.asyncio
async def test_evidence_api_crud_and_profile_scoping(async_client) -> None:
    profile_response = await async_client.post(
        "/api/v1/candidate-profile", json={"full_name": "Ada Lovelace"}
    )
    profile_id = profile_response.json()["id"]
    other_profile_id = (
        await async_client.post("/api/v1/candidate-profile", json={"full_name": "Other"})
    ).json()["id"]
    path = f"/api/v1/candidate-profile/{profile_id}/evidence"

    created_response = await async_client.post(path, json=evidence_payload())
    assert created_response.status_code == 201
    evidence_id = created_response.json()["id"]
    assert (await async_client.get(path)).json()[0]["id"] == evidence_id
    assert (await async_client.get(f"{path}/{evidence_id}")).status_code == 200

    patched = await async_client.patch(f"{path}/{evidence_id}", json={"title": "Patched"})
    assert patched.status_code == 200
    assert patched.json()["title"] == "Patched"
    assert patched.json()["content"] == "Built an async service."

    assert (await async_client.get(f"/api/v1/candidate-profile/{other_profile_id}/evidence/{evidence_id}")).status_code == 404
    assert (await async_client.delete(f"{path}/{evidence_id}")).status_code == 204
    assert (await async_client.get(f"{path}/{evidence_id}")).status_code == 404


@pytest.mark.asyncio
async def test_evidence_api_rejects_unknown_profile(async_client) -> None:
    response = await async_client.post(
        f"/api/v1/candidate-profile/{uuid4()}/evidence", json=evidence_payload()
    )
    assert response.status_code == 404
    assert (await async_client.get(f"/api/v1/candidate-profile/{uuid4()}/evidence")).status_code == 404
