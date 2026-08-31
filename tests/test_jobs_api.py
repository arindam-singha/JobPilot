from uuid import uuid4

async def test_ingest_linkedin_job(async_client) -> None:
    payload = {
        "job_url": "https://www.linkedin.com/jobs/view/123456789",
        "job_description": (
            "We are looking for a Senior Robotics Engineer "
            "with experience in Python, ROS2, and deep learning."
        ),
        "title": "Senior Robotics Engineer",
        "company": "Example Robotics",
        "location": "Abu Dhabi, UAE",
    }

    response = await async_client.post(
        "/api/v1/jobs/ingest",
        json=payload,
    )

    assert response.status_code == 201

    body = response.json()

    assert body["title"] == payload["title"]
    assert body["company"] == payload["company"]
    assert body["location"] == payload["location"]
    assert body["job_url"] == payload["job_url"]
    assert body["description"] == payload["job_description"]
    assert body["source"] == "linkedin"

async def test_ingest_indeed_job(async_client) -> None:
    payload = {
        "job_url": "https://www.indeed.com/viewjob?jk=123456",
        "job_description": (
            "Senior machine learning engineer responsible "
            "for production AI systems."
        ),
        "title": "Senior Machine Learning Engineer",
        "company": "Example AI",
        "location": "Remote",
    }

    response = await async_client.post(
        "/api/v1/jobs/ingest",
        json=payload,
    )

    assert response.status_code == 201
    assert response.json()["source"] == "indeed"

async def test_ingest_unknown_source(async_client) -> None:
    payload = {
        "job_url": "https://careers.example.com/jobs/123",
        "job_description": (
            "Build machine learning and robotics applications "
            "for autonomous systems."
        ),
        "title": "Robotics AI Engineer",
        "company": "Example Robotics",
        "location": "Bangalore, India",
    }

    response = await async_client.post(
        "/api/v1/jobs/ingest",
        json=payload,
    )

    assert response.status_code == 201
    assert response.json()["source"] == "other"

async def test_ingest_invalid_url(async_client) -> None:
    payload = {
        "job_url": "not-a-url",
        "job_description": (
            "This is a sufficiently long job description "
            "for validation purposes."
        ),
        "title": "Software Engineer",
        "company": "Example",
    }

    response = await async_client.post(
        "/api/v1/jobs/ingest",
        json=payload,
    )

    assert response.status_code == 422

async def test_ingest_duplicate_job_returns_existing_job(
    async_client,
) -> None:
    payload = {
        "job_url": "https://www.linkedin.com/jobs/view/987654321",
        "job_description": (
            "Senior robotics engineer working on autonomous "
            "systems and machine learning."
        ),
        "title": "Senior Robotics Engineer",
        "company": "Example Robotics",
        "location": "Pune, India",
    }

    first_response = await async_client.post(
        "/api/v1/jobs/ingest",
        json=payload,
    )

    second_response = await async_client.post(
        "/api/v1/jobs/ingest",
        json=payload,
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 200

    first_body = first_response.json()
    second_body = second_response.json()

    assert first_body["id"] == second_body["id"]
    
async def test_create_and_fetch_job(async_client) -> None:
    payload = {
        "title": "Senior Python Engineer",
        "company": "Example Labs",
        "location": "Remote",
        "job_url": "https://example.com/jobs/123",
        "description": "Build reliable APis and ingestion tools.",
        "source": "manual",
    }

    create_response = await async_client.post("/api/v1/jobs", json=payload)
    assert create_response.status_code == 201
    body = create_response.json()

    assert body["title"] == payload["title"]
    assert body["company"] == payload["company"]
    assert body["job_url"] == payload["job_url"]

    job_id = body["id"]

    list_response = await async_client.get("/api/v1/jobs")
    assert list_response.status_code == 200
    assert isinstance(list_response.json(), list)

    single_response = await async_client.get(f"/api/v1/jobs/{job_id}")
    assert single_response.status_code == 200
    assert single_response.json()["id"] == job_id


async def test_get_missing_job_returns_404(async_client) -> None:
    missing_id = str(uuid4())
    response = await async_client.get(f"/api/v1/jobs/{missing_id}")
    assert response.status_code == 404
