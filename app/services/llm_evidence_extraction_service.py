from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.cv.models import TextChunk
from app.llm.evidence_provider import EvidenceExtractionProvider
from app.models.candidate_evidence import CandidateEvidence
from app.schemas.llm_evidence import LlmEvidenceItem
from app.services.evidence_extraction_service import (
    EvidenceExtractionService,
)


LLM_SOURCE_TYPE = "candidate_document_llm"


class LlmEvidenceExtractionError(Exception):
    """Base exception for LLM evidence extraction failures."""


class LlmEvidenceProviderFailureError(LlmEvidenceExtractionError):
    """Raised when the configured provider fails."""


class LlmEvidenceExtractionService:
    """Extract and persist structured candidate evidence using an LLM provider."""

    def __init__(
        self,
        session: AsyncSession,
        provider: EvidenceExtractionProvider,
        *,
        max_chunk_characters: int = 2000,
        chunk_overlap_characters: int = 200,
    ) -> None:
        self.session = session
        self.provider = provider
        self.chunk_service = EvidenceExtractionService(
            session,
            max_chunk_characters=max_chunk_characters,
            chunk_overlap_characters=chunk_overlap_characters,
        )

    async def extract_document_evidence(
        self,
        profile_id: UUID,
        document_id: UUID,
    ) -> list[CandidateEvidence]:
        """Extract and persist LLM-generated evidence for one document.

        Existing LLM-generated evidence for the same document is replaced.
        Deterministic and manually created evidence records are preserved.
        """

        chunks = await self.chunk_service.build_document_chunks(
            profile_id,
            document_id,
        )

        generated_records: list[CandidateEvidence] = []

        for chunk in chunks:
            try:
                result = await self.provider.extract_evidence(chunk)
            except Exception as exc:
                raise LlmEvidenceProviderFailureError(
                    f"Evidence provider {self.provider.provider_name!r} "
                    f"failed for chunk {chunk.index}"
                ) from exc

            for item_index, item in enumerate(result.evidence):
                generated_records.append(
                    self._build_evidence_record(
                        profile_id=profile_id,
                        document_id=document_id,
                        chunk=chunk,
                        item=item,
                        item_index=item_index,
                    )
                )

        try:
            await self.session.execute(
                delete(CandidateEvidence).where(
                    CandidateEvidence.profile_id == profile_id,
                    CandidateEvidence.source_type == LLM_SOURCE_TYPE,
                    CandidateEvidence.source_id == document_id,
                )
            )

            self.session.add_all(generated_records)
            await self.session.commit()

            for record in generated_records:
                await self.session.refresh(record)

        except Exception:
            await self.session.rollback()
            raise

        return generated_records

    def _build_evidence_record(
        self,
        *,
        profile_id: UUID,
        document_id: UUID,
        chunk: TextChunk,
        item: LlmEvidenceItem,
        item_index: int,
    ) -> CandidateEvidence:
        metadata = self._build_metadata(
            chunk=chunk,
            item=item,
            item_index=item_index,
        )

        return CandidateEvidence(
            profile_id=profile_id,
            evidence_type=item.evidence_type,
            title=item.title,
            content=item.content,
            source_type=LLM_SOURCE_TYPE,
            source_id=document_id,
            metadata_json=json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def _build_metadata(
        self,
        *,
        chunk: TextChunk,
        item: LlmEvidenceItem,
        item_index: int,
    ) -> dict[str, Any]:
        return {
            "chunk_index": chunk.index,
            "item_index": item_index,
            "section": chunk.section,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "confidence": item.confidence,
            "provider": self.provider.provider_name,
            "model": self.provider.model_name,
            "generation_method": "llm_structured_extraction",
            "provider_metadata": item.metadata,
        }