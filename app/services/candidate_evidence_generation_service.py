from __future__ import annotations

import json
from uuid import UUID

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.cv.models import TextChunk
from app.models.candidate_evidence import CandidateEvidence
from app.services.evidence_extraction_service import EvidenceExtractionService

SECTION_EVIDENCE_TYPE_MAP: dict[str | None, str] = {
    None: "general",
    "summary": "summary",
    "professional_summary": "summary",
    "profile": "summary",
    "objective": "summary",
    "skills": "skill",
    "technical_skills": "skill",
    "core_competencies": "skill",
    "experience": "experience",
    "education": "education",
    "projects": "project",
    "publications": "publication",
    "certifications": "certification",
    "achievements": "achievement",
    "awards": "achievement",
}


SECTION_TITLE_MAP: dict[str | None, str] = {
    None: "General candidate information",
    "summary": "Candidate summary",
    "professional_summary": "Professional summary",
    "profile": "Candidate profile",
    "objective": "Career objective",
    "skills": "Skills",
    "technical_skills": "Technical skills",
    "core_competencies": "Core competencies",
    "experience": "Professional experience",
    "education": "Education",
    "projects": "Projects",
    "publications": "Publications",
    "certifications": "Certifications",
    "achievements": "Achievements",
    "awards": "Awards",
}


class CandidateEvidenceGenerationError(Exception):
    """Base exception for candidate evidence generation errors."""


class CandidateEvidenceGenerationService:
    """Generate deterministic evidence records from candidate documents."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        max_chunk_characters: int = 2000,
        chunk_overlap_characters: int = 200,
    ) -> None:
        self.session = session
        self.extraction_service = EvidenceExtractionService(
            session,
            max_chunk_characters=max_chunk_characters,
            chunk_overlap_characters=chunk_overlap_characters,
        )

    async def generate_document_evidence(
        self,
        profile_id: UUID,
        document_id: UUID,
    ) -> list[CandidateEvidence]:
        """Generate and persist evidence derived from one document.

        Previously generated evidence for the same profile and document is
        replaced, making repeated generation idempotent.
        """

        chunks = await self.extraction_service.build_document_chunks(
            profile_id,
            document_id,
        )

        evidence_records = [
            self._build_evidence(
                profile_id=profile_id,
                document_id=document_id,
                chunk=chunk,
            )
            for chunk in chunks
        ]

        try:
            await self.session.execute(
                delete(CandidateEvidence).where(
                    CandidateEvidence.profile_id == profile_id,
                    CandidateEvidence.source_type == "candidate_document",
                    CandidateEvidence.source_id == document_id,
                )
            )

            self.session.add_all(evidence_records)
            await self.session.commit()

            for evidence in evidence_records:
                await self.session.refresh(evidence)

        except Exception:
            await self.session.rollback()
            raise

        return evidence_records

    @staticmethod
    def _build_evidence(
        *,
        profile_id: UUID,
        document_id: UUID,
        chunk: TextChunk,
    ) -> CandidateEvidence:
        evidence_type = SECTION_EVIDENCE_TYPE_MAP.get(
            chunk.section,
            "general",
        )

        base_title = SECTION_TITLE_MAP.get(
            chunk.section,
            "Candidate evidence",
        )

        metadata = {
            "chunk_index": chunk.index,
            "section": chunk.section,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "generation_method": "deterministic_section_chunking",
        }

        return CandidateEvidence(
            profile_id=profile_id,
            evidence_type=evidence_type,
            title=f"{base_title} — chunk {chunk.index + 1}",
            content=chunk.text,
            source_type="candidate_document",
            source_id=document_id,
            metadata_json=json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )