from __future__ import annotations

from typing import Protocol

from app.cv.models import TextChunk
from app.schemas.llm_evidence import LlmChunkExtractionResult


class EvidenceExtractionProviderError(Exception):
    """Base exception for LLM evidence provider failures."""


class EvidenceExtractionProvider(Protocol):
    """Interface implemented by LLM-backed evidence extraction providers."""

    @property
    def provider_name(self) -> str:
        """Stable provider identifier stored in evidence provenance."""

    @property
    def model_name(self) -> str:
        """Model identifier stored in evidence provenance."""

    async def extract_evidence(
        self,
        chunk: TextChunk,
    ) -> LlmChunkExtractionResult:
        """Extract structured candidate evidence from one text chunk."""