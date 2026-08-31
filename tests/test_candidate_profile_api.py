from uuid import uuid4

import pytest


async def _create_profile(async_client, payload=None):
    data = payload or {
        "full_name": "Ada Lovelace",
        "email": "ada@example.com",
        "location": "London",
        "professional_summary": "First computer programmer.",
        "total_experience_years": 12.5,
    }
    response = await async_client.post("/api/v1/candidate-profile", json=data)
    assert response.status_code == 201
    return response


@pytest.mark.asyncio
async def test_create_profile_returns_201_and_profile_shape(async_client) -> None:
    response = await _create_profile(async_client)
    body = response.json()

    assert body["full_name"] == "Ada Lovelace"
    assert body["email"] == "ada@example.com"
    assert body["location"] == "London"
    assert body["professional_summary"] == "First computer programmer."
    assert body["total_experience_years"] == 12.5
    assert body["experiences"] == []
    assert body["skills"] == []
    assert body["education"] == []
    assert body["projects"] == []
    assert body["publications"] == []
    assert body["certifications"] == []
    assert body["achievements"] == []
    assert "id" in body
    assert "created_at" in body
    assert "updated_at" in body


@pytest.mark.asyncio
async def test_get_profile_includes_all_nested_collections(async_client) -> None:
    create_response = await _create_profile(async_client)
    profile_id = create_response.json()["id"]

    await async_client.post(f"/api/v1/candidate-profile/{profile_id}/experiences", json={"company": "Acme", "role": "Engineer"})
    await async_client.post(f"/api/v1/candidate-profile/{profile_id}/skills", json={"name": "Python", "years_of_experience": 8})
    await async_client.post(f"/api/v1/candidate-profile/{profile_id}/education", json={"institution": "Oxford", "degree": "MSc", "field_of_study": "CS"})
    await async_client.post(f"/api/v1/candidate-profile/{profile_id}/projects", json={"name": "JobPilot"})
    await async_client.post(f"/api/v1/candidate-profile/{profile_id}/publications", json={"title": "On Computing"})
    await async_client.post(f"/api/v1/candidate-profile/{profile_id}/certifications", json={"name": "AWS"})
    await async_client.post(f"/api/v1/candidate-profile/{profile_id}/achievements", json={"title": "Top Performer"})

    response = await async_client.get(f"/api/v1/candidate-profile/{profile_id}")

    assert response.status_code == 200
    body = response.json()
    assert len(body["experiences"]) == 1
    assert len(body["skills"]) == 1
    assert len(body["education"]) == 1
    assert len(body["projects"]) == 1
    assert len(body["publications"]) == 1
    assert len(body["certifications"]) == 1
    assert len(body["achievements"]) == 1


@pytest.mark.asyncio
async def test_get_unknown_profile_returns_404(async_client) -> None:
    response = await async_client.get(f"/api/v1/candidate-profile/{uuid4()}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_patch_profile_updates_only_supplied_fields(async_client) -> None:
    create_response = await _create_profile(async_client)
    profile_id = create_response.json()["id"]

    response = await async_client.patch(
        f"/api/v1/candidate-profile/{profile_id}",
        json={"full_name": "Ada Updated", "total_experience_years": 15},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["full_name"] == "Ada Updated"
    assert body["email"] == "ada@example.com"
    assert body["total_experience_years"] == 15


@pytest.mark.asyncio
async def test_patch_unknown_profile_returns_404(async_client) -> None:
    response = await async_client.patch(
        f"/api/v1/candidate-profile/{uuid4()}",
        json={"full_name": "No One"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_and_patch_delete_experience(async_client) -> None:
    profile_response = await _create_profile(async_client)
    profile_id = profile_response.json()["id"]

    create_response = await async_client.post(
        f"/api/v1/candidate-profile/{profile_id}/experiences",
        json={"company": "Contoso", "role": "Engineer"},
    )
    assert create_response.status_code == 201
    experience_id = create_response.json()["id"]

    patch_response = await async_client.patch(
        f"/api/v1/candidate-profile/{profile_id}/experiences/{experience_id}",
        json={"company": "NewCo", "role": "Senior Engineer"},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["company"] == "NewCo"

    delete_response = await async_client.delete(f"/api/v1/candidate-profile/{profile_id}/experiences/{experience_id}")
    assert delete_response.status_code == 204


@pytest.mark.asyncio
async def test_unknown_experience_and_cross_profile_experience_access_return_404(async_client) -> None:
    profile_a = (await _create_profile(async_client, {"full_name": "A"})).json()["id"]
    profile_b = (await _create_profile(async_client, {"full_name": "B"})).json()["id"]
    created = await async_client.post(
        f"/api/v1/candidate-profile/{profile_b}/experiences",
        json={"company": "Other", "role": "Engineer"},
    )
    experience_id = created.json()["id"]

    missing = await async_client.patch(f"/api/v1/candidate-profile/{profile_a}/experiences/{uuid4()}", json={"company": "X", "role": "Y"})
    assert missing.status_code == 404

    cross = await async_client.patch(
        f"/api/v1/candidate-profile/{profile_a}/experiences/{experience_id}",
        json={"company": "X", "role": "Y"},
    )
    assert cross.status_code == 404


@pytest.mark.asyncio
async def test_skill_crud_async_client(async_client) -> None:
    profile_id = (await _create_profile(async_client)).json()["id"]

    create_response = await async_client.post(
        f"/api/v1/candidate-profile/{profile_id}/skills",
        json={"name": "Python", "years_of_experience": 5.5},
    )
    assert create_response.status_code == 201
    skill_id = create_response.json()["id"]

    patch_response = await async_client.patch(
        f"/api/v1/candidate-profile/{profile_id}/skills/{skill_id}",
        json={"name": "Python", "proficiency": "Expert"},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["proficiency"] == "Expert"

    delete_response = await async_client.delete(f"/api/v1/candidate-profile/{profile_id}/skills/{skill_id}")
    assert delete_response.status_code == 204


@pytest.mark.asyncio
async def test_education_crud_async_client(async_client) -> None:
    profile_id = (await _create_profile(async_client)).json()["id"]

    create_response = await async_client.post(
        f"/api/v1/candidate-profile/{profile_id}/education",
        json={"institution": "University of Washington", "degree": "BS", "field_of_study": "CS"},
    )
    assert create_response.status_code == 201
    education_id = create_response.json()["id"]

    patch_response = await async_client.patch(
        f"/api/v1/candidate-profile/{profile_id}/education/{education_id}",
        json={"institution": "Stanford", "degree": "MS"},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["institution"] == "Stanford"

    delete_response = await async_client.delete(f"/api/v1/candidate-profile/{profile_id}/education/{education_id}")
    assert delete_response.status_code == 204


@pytest.mark.asyncio
async def test_project_crud_async_client(async_client) -> None:
    profile_id = (await _create_profile(async_client)).json()["id"]

    create_response = await async_client.post(
        f"/api/v1/candidate-profile/{profile_id}/projects",
        json={"name": "JobPilot", "description": "Makes hiring easier."},
    )
    assert create_response.status_code == 201
    project_id = create_response.json()["id"]

    patch_response = await async_client.patch(
        f"/api/v1/candidate-profile/{profile_id}/projects/{project_id}",
        json={"name": "JobPilot v2", "description": "Updated"},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["name"] == "JobPilot v2"

    delete_response = await async_client.delete(f"/api/v1/candidate-profile/{profile_id}/projects/{project_id}")
    assert delete_response.status_code == 204


@pytest.mark.asyncio
async def test_publication_crud_async_client(async_client) -> None:
    profile_id = (await _create_profile(async_client)).json()["id"]

    create_response = await async_client.post(
        f"/api/v1/candidate-profile/{profile_id}/publications",
        json={"title": "A Paper", "venue": "IEEE"},
    )
    assert create_response.status_code == 201
    publication_id = create_response.json()["id"]

    patch_response = await async_client.patch(
        f"/api/v1/candidate-profile/{profile_id}/publications/{publication_id}",
        json={"title": "A Better Paper", "venue": "ACM"},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["title"] == "A Better Paper"

    delete_response = await async_client.delete(f"/api/v1/candidate-profile/{profile_id}/publications/{publication_id}")
    assert delete_response.status_code == 204


@pytest.mark.asyncio
async def test_certification_crud_async_client(async_client) -> None:
    profile_id = (await _create_profile(async_client)).json()["id"]

    create_response = await async_client.post(
        f"/api/v1/candidate-profile/{profile_id}/certifications",
        json={"name": "AWS", "issuing_organization": "Amazon"},
    )
    assert create_response.status_code == 201
    certification_id = create_response.json()["id"]

    patch_response = await async_client.patch(
        f"/api/v1/candidate-profile/{profile_id}/certifications/{certification_id}",
        json={"name": "AWS SAA", "issuing_organization": "Amazon Web Services"},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["name"] == "AWS SAA"

    delete_response = await async_client.delete(f"/api/v1/candidate-profile/{profile_id}/certifications/{certification_id}")
    assert delete_response.status_code == 204


@pytest.mark.asyncio
async def test_achievement_crud_async_client(async_client) -> None:
    profile_id = (await _create_profile(async_client)).json()["id"]

    create_response = await async_client.post(
        f"/api/v1/candidate-profile/{profile_id}/achievements",
        json={"title": "Top Performer", "description": "Delivered ahead of schedule."},
    )
    assert create_response.status_code == 201
    achievement_id = create_response.json()["id"]

    patch_response = await async_client.patch(
        f"/api/v1/candidate-profile/{profile_id}/achievements/{achievement_id}",
        json={"title": "Top Deliverer", "description": "Improved performance."},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["title"] == "Top Deliverer"

    delete_response = await async_client.delete(f"/api/v1/candidate-profile/{profile_id}/achievements/{achievement_id}")
    assert delete_response.status_code == 204


@pytest.mark.asyncio
async def test_invalid_profile_data_returns_422(async_client) -> None:
    response = await async_client.post(
        "/api/v1/candidate-profile",
        json={"full_name": "   ", "email": "bad"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_uuid_path_parameter_returns_422(async_client) -> None:
    response = await async_client.get("/api/v1/candidate-profile/not-a-uuid")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_nested_data_returns_422(async_client) -> None:
    profile_id = (await _create_profile(async_client)).json()["id"]
    response = await async_client.post(
        f"/api/v1/candidate-profile/{profile_id}/experiences",
        json={"company": "", "role": "Engineer"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_and_job_routes_still_work(async_client) -> None:
    health_response = await async_client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    job_response = await async_client.post(
        "/api/v1/jobs",
        json={
            "title": "API Engineer",
            "company": "Example",
            "location": "Remote",
            "job_url": "https://example.com/jobs/abc",
            "description": "Build API systems.",
            "source": "manual",
        },
    )
    assert job_response.status_code == 201
