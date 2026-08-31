async def test_job_ingestion_end_to_end(async_client, database_session) -> None:
    payload = {
        "job_url": "https://www.linkedin.com/jobs/view/e2e-123456",
        "job_description": (
            "We are looking for a Senior Robotics Engineer with "
            "experience in Python, ROS2, computer vision, and "
            "deep learning for autonomous systems."
        ),
        "title": "Senior Robotics Engineer",
        "company": "E2E Robotics",
        "location": "Abu Dhabi, UAE",
    }

    # 1. Ingest a new job through the HTTP API.
    first_response = await async_client.post(
        "/api/v1/jobs/ingest",
        json=payload,
    )

    assert first_response.status_code == 201

    first_body = first_response.json()

    assert first_body["title"] == payload["title"]
    assert first_body["company"] == payload["company"]
    assert first_body["location"] == payload["location"]
    assert first_body["job_url"] == payload["job_url"]
    assert first_body["description"] == payload["job_description"]
    assert first_body["source"] == "linkedin"

    job_id = first_body["id"]

    # 2. Verify that the job was actually persisted in PostgreSQL.
    from uuid import UUID

    from sqlalchemy import select

    from app.models.job import Job

    result = await database_session.execute(
        select(Job).where(Job.id == UUID(job_id))
    )

    stored_job = result.scalar_one_or_none()

    assert stored_job is not None
    assert str(stored_job.id) == job_id
    assert stored_job.title == payload["title"]
    assert stored_job.company == payload["company"]
    assert stored_job.job_url == payload["job_url"]
    assert stored_job.source == "linkedin"

    # 3. Submit the same job again.
    second_response = await async_client.post(
        "/api/v1/jobs/ingest",
        json=payload,
    )

    # Duplicate ingestion should return the existing job.
    assert second_response.status_code == 200

    second_body = second_response.json()

    # 4. Verify that the existing database record was returned.
    assert second_body["id"] == job_id
    assert second_body["job_url"] == payload["job_url"]