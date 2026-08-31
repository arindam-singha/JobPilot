from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.embeddings.embedding_provider_factory import create_embedding_provider
from app.llm.evidence_provider_factory import create_evidence_provider
from app.schemas.application_package import CandidateProfilePreparationRead
from app.schemas.candidate_document import CandidateDocumentRead
from app.services.candidate_document_service import CandidateDocumentService
from app.services.candidate_evidence_embedding_service import (
    CandidateEvidenceEmbeddingService,
)
from app.services.document_ingestion_service import (
    DocumentIngestionService,
    DocumentIngestionValidationError,
)
from app.services.document_storage_service import DocumentStorageService
from app.services.llm_evidence_extraction_service import LlmEvidenceExtractionService

router = APIRouter(
    prefix="/api/v1/candidate-profile",
    tags=["candidate-documents"],
)


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
