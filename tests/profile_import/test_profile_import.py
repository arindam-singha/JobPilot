"""Isolated import tests; run with --confcutdir=tests/profile_import.

No production database or live Ollama instance is used.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from app.cv.text_processing import detect_sections, normalize_text
from app.schemas.profile_import import ProfileImportConfirm, ProfileImportData
from app.services import profile_import_service as service
from app.services.profile_import_service import ProfileImportError as HTTPException
from pydantic import ValidationError


def data(**kwargs):
    return ProfileImportData(
        profile={"full_name": "Candidate", "email": "saved@example.com"}, **kwargs
    )


def test_real_resume_headings():
    sections = detect_sections(
        normalize_text(
            "PROFESSIONAL EXPERIENCE\nRole\nSELECTED GENERATIVE AI & MACHINE LEARNING PROJECTS\n"
            "Project\nEDUCATION\nPhD\nSELECTED PUBLICATIONS\nPaper\nAWARDS & RECOGNITION\n"
            "Scholarship\nADDITIONAL INFORMATION\nLanguages"
        )
    )
    assert [s.name for s in sections] == [
        "experience",
        "projects",
        "education",
        "publications",
        "achievements",
        "additional_information",
    ]


def test_merge_preserves_manual_fields_and_deduplicates():
    old = data(skills=[{"id": uuid4(), "name": "Python", "proficiency": "user value"}])
    incoming = data(
        skills=[
            {"name": "Python", "source_quote": "Python"},
            {"name": "ROS2", "source_quote": "ROS2"},
            {"name": "ROS2", "source_quote": "ROS2"},
        ]
    )
    incoming.profile.email = "resume@example.com"
    merged = service.merge_review(incoming, old, "Python ROS2")
    assert merged.profile.email == "saved@example.com"
    assert [r.name for r in merged.skills] == ["Python", "ROS2"]
    assert merged.skills[0].proficiency == "user value"
    assert any("email" in w for w in merged.warnings)


def test_missing_source_is_flagged_not_silently_trusted():
    merged = service.merge_review(
        data(skills=[{"name": "Invented", "source_quote": "no"}]), data(), "Python"
    )
    assert not merged.skills
    assert any("source excerpt" in w for w in merged.warnings)


def test_dates_are_validated():
    with pytest.raises(ValidationError):
        data(
            experiences=[
                {
                    "company": "A",
                    "role": "B",
                    "start_date": "2024-01-01",
                    "end_date": "2023-01-01",
                }
            ]
        )
    with pytest.raises(ValidationError):
        data(
            experiences=[
                {
                    "company": "A",
                    "role": "B",
                    "is_current": True,
                    "end_date": "2024-01-01",
                }
            ]
        )


def test_partial_dates_preserved_as_text():
    item = data(
        education=[
            {
                "institution": "University",
                "degree": "PhD",
                "description": "2016 - 2021",
                "source_quote": "PhD 2016 - 2021",
            }
        ]
    )
    merged = service.merge_review(item, data(), "PhD 2016 - 2021")
    assert merged.education[0].start_date is None
    assert merged.education[0].description == "2016 - 2021"
    assert any("incomplete dates" in w for w in merged.warnings)


def test_evidence_keeps_employer_dates_and_entity_identity():
    entity_id, profile_id, document_id = uuid4(), uuid4(), uuid4()
    reviewed = data(
        experiences=[
            {
                "id": entity_id,
                "company": "Employer",
                "role": "Lead",
                "start_date": "2024-07-01",
                "is_current": True,
                "achievements": "Built RAG",
                "source_quote": "Lead Employer Built RAG",
            }
        ]
    )
    row = service.evidence_rows(profile_id, document_id, reviewed)[0]
    assert row.source_id == entity_id
    assert "Employer" in row.content and "2024-07-01" in row.content
    assert "Built RAG" in row.content
    assert row.embedding is None
    assert json.loads(row.metadata_json)["document_id"] == str(document_id)


def test_revision_changes_on_edits_not_order():
    a = data(skills=[{"id": uuid4(), "name": "Python"}, {"id": uuid4(), "name": "ROS2"}])
    b = a.model_copy(deep=True)
    b.skills.reverse()
    assert service.revision(a) == service.revision(b)
    b.skills[0].name = "Changed"
    assert service.revision(a) != service.revision(b)


@pytest.mark.asyncio
async def test_no_confirmation_no_writes():
    session = AsyncMock()
    with pytest.raises(HTTPException) as error:
        await service.confirm_review(
            session,
            ProfileImportConfirm(
                profile_id=uuid4(), document_id=uuid4(), base_revision="x", data=data()
            ),
        )
    assert error.value.status_code == 422
    session.execute.assert_not_called()


# @pytest.mark.asyncio
# async def test_ollama_structured_request(monkeypatch):
#     httpx = pytest.importorskip("httpx")
#     monkeypatch.setenv("OLLAMA_PROFILE_MODEL", "test-model")

#     def respond(request):
#         payload = json.loads(request.content)
#         assert payload["format"]["properties"]["experiences"]
#         assert payload["options"]["temperature"] == 0
#         return httpx.Response(200, json={"message": {"content": data().model_dump_json()}})

#     async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
#         result = await service.extract_profile("Candidate", client)
#     assert result.profile.full_name == "Candidate"

@pytest.mark.asyncio
async def test_ollama_structured_request(monkeypatch):
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("OLLAMA_PROFILE_MODEL", "test-model")

    def respond(request):
        payload = json.loads(request.content)

        assert payload["format"]["properties"]["items"]
        assert payload["options"]["temperature"] == 0

        generated = {
            "items": [
                {
                    "company": "Example Company",
                    "role": "Engineer",
                    "description": "Candidate",
                }
            ]
        }

        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(generated),
                }
            },
        )

    resume_text = """
PROFESSIONAL EXPERIENCE
Candidate
""".strip()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        result = await service.extract_profile(
            resume_text,
            data().profile,
            client=client,
        )

    assert result.profile.full_name == "Candidate"
    assert len(result.experiences) == 1
    assert result.experiences[0].company == "Example Company"
    assert result.experiences[0].role == "Engineer"


# @pytest.mark.asyncio
# @pytest.mark.parametrize(
#     "body",
#     [
#         {"message": {"content": "not-json"}},
#         {"done_reason": "length", "message": {"content": "{}"}},
#         {"error": "model unavailable"},
#     ],
# )
# async def test_bad_llm_output_does_not_become_ready(monkeypatch, body):
#     httpx = pytest.importorskip("httpx")
#     monkeypatch.setenv("OLLAMA_PROFILE_MODEL", "test-model")
#     async with httpx.AsyncClient(
#         transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
#     ) as client:
#         with pytest.raises(HTTPException) as error:
#             await service.extract_profile("Candidate", client)
#     assert error.value.status_code == 502

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"message": {"content": "not-json"}},
        {"done_reason": "length", "message": {"content": "{}"}},
        {"error": "model unavailable"},
    ],
)
async def test_bad_llm_output_does_not_become_ready(
    monkeypatch,
    body,
):
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("OLLAMA_PROFILE_MODEL", "test-model")

    resume_text = """
PROFESSIONAL EXPERIENCE
Candidate
""".strip()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=body)
        )
    ) as client:
        with pytest.raises(HTTPException) as error:
            await service.extract_profile(
                resume_text,
                data().profile,
                client=client,
            )

    assert error.value.status_code == 502


@pytest.mark.asyncio
async def test_stale_review_does_not_save(monkeypatch):
    current = data()
    monkeypatch.setattr(service, "snapshot", lambda profile: current)
    monkeypatch.setattr(
        service,
        "CandidateDocumentService",
        lambda session: SimpleNamespace(get_document=AsyncMock(return_value=object())),
    )
    monkeypatch.setattr(
        service,
        "CandidateProfileService",
        lambda session: SimpleNamespace(get_profile=AsyncMock(return_value=object())),
    )
    session = AsyncMock()
    with pytest.raises(HTTPException) as error:
        await service.confirm_review(
            session,
            ProfileImportConfirm(
                profile_id=uuid4(),
                document_id=uuid4(),
                base_revision="stale",
                data=current,
                confirmed=True,
            ),
        )
    assert error.value.status_code == 409
    session.commit.assert_not_called()
