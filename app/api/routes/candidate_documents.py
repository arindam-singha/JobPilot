from __future__ import annotations

import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.embeddings.embedding_provider_factory import create_embedding_provider
from app.llm.evidence_provider_factory import create_evidence_provider
from app.models.candidate_evidence import CandidateEvidence
from app.schemas.application_package import CandidateProfilePreparationRead
from app.schemas.candidate_document import CandidateDocumentRead
from app.schemas.profile_import import ProfileImportConfirm, ProfileImportReview
from app.services.candidate_document_service import (
    CandidateDocumentNotFoundError,
    CandidateDocumentService,
)
from app.services.candidate_evidence_embedding_service import (
    CandidateEvidenceEmbeddingService,
)
from app.services.candidate_profile_service import (
    CandidateProfileNotFoundError,
    CandidateProfileService,
)
from app.services.document_ingestion_service import (
    DocumentIngestionService,
    DocumentIngestionValidationError,
)
from app.services.document_storage_service import DocumentStorageService
from app.services.llm_evidence_extraction_service import LlmEvidenceExtractionService
from app.services.profile_import_service import (
    REVIEWED_SOURCE,
    ProfileImportError,
    confirm_review,
    create_review,
    load_saved_review,
    revision,
    snapshot,
)


async def _import_result(operation):
    try:
        return await operation
    except ProfileImportError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc
    except (CandidateDocumentNotFoundError, CandidateProfileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc


router = APIRouter(
    prefix="/api/v1/candidate-profile",
    tags=["candidate-documents"],
)


@router.post("/{profile_id}/documents/{document_id}/review", response_model=ProfileImportReview)
async def review_document(
    profile_id: UUID, document_id: UUID, db: Annotated[AsyncSession, Depends(get_db)]
):
    return await _import_result(create_review(db, profile_id, document_id))


@router.get("/{profile_id}/documents/{document_id}/review", response_model=ProfileImportReview)
async def read_saved_review(
    profile_id: UUID, document_id: UUID, db: Annotated[AsyncSession, Depends(get_db)]
):
    return await _import_result(load_saved_review(db, profile_id, document_id))


@router.post("/{profile_id}/documents/{document_id}/confirm-review")
async def save_document_review(
    profile_id: UUID,
    document_id: UUID,
    payload: ProfileImportConfirm,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    if payload.profile_id != profile_id or payload.document_id != document_id:
        raise HTTPException(422, "Review IDs do not match the selected profile/document.")
    return await _import_result(confirm_review(db, payload))


@router.post("/{profile_id}/prepare-reviewed")
async def prepare_reviewed_profile(profile_id: UUID, db: Annotated[AsyncSession, Depends(get_db)]):
    records = list(
        (
            await db.scalars(
                select(CandidateEvidence).where(
                    CandidateEvidence.profile_id == profile_id,
                    CandidateEvidence.source_type == REVIEWED_SOURCE,
                )
            )
        ).all()
    )
    if not records:
        raise HTTPException(409, "Review and confirm the candidate profile first.")
    current_revision = revision(snapshot(await CandidateProfileService(db).get_profile(profile_id)))
    if any(
        json.loads(row.metadata_json or "{}").get("profile_revision") != current_revision
        for row in records
    ):
        raise HTTPException(409, "Profile changed after confirmation. Review and confirm it again.")
    embedded = await CandidateEvidenceEmbeddingService(
        session=db, provider=create_embedding_provider()
    ).embed_profile_evidence(profile_id, force=False)
    return {
        "ready": bool(embedded) and all(r.embedding is not None for r in embedded),
        "embedded_records": sum(r.embedding is not None for r in embedded),
    }


@router.post(
    "/{profile_id}/documents/upload",
    response_model=CandidateDocumentRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_candidate_document(
    profile_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    file: Annotated[UploadFile, File(...)],
) -> CandidateDocumentRead:
    content_type = file.content_type or ""
    try:
        document = await DocumentIngestionService(
            session=db,
            document_service=CandidateDocumentService(db),
            storage_service=DocumentStorageService(),
        ).ingest_document(
            profile_id=profile_id,
            file_content=await file.read(),
            filename=file.filename or "resume",
            content_type=content_type,
        )
    except DocumentIngestionValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return CandidateDocumentRead.model_validate(document)


@router.post(
    "/{profile_id}/documents/{document_id}/prepare",
    response_model=CandidateProfilePreparationRead,
)
async def prepare_candidate_document(
    profile_id: UUID,
    document_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CandidateProfilePreparationRead:
    document = await CandidateDocumentService(db).get_document(profile_id, document_id)
    if document.extraction_status != "extracted":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=document.extraction_error or "Resume text was not extracted",
        )
    evidence = await LlmEvidenceExtractionService(
        db,
        create_evidence_provider(),
    ).extract_document_evidence(profile_id, document_id)
    embedded = await CandidateEvidenceEmbeddingService(
        session=db,
        provider=create_embedding_provider(),
    ).embed_profile_evidence(profile_id, force=False)
    embedded_count = sum(item.embedding is not None for item in embedded)
    return CandidateProfilePreparationRead(
        profile_id=profile_id,
        document_id=document_id,
        evidence_records=len(evidence),
        embedded_records=embedded_count,
        ready=bool(embedded) and embedded_count == len(embedded),
    )
