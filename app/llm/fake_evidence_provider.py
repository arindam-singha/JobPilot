from __future__ import annotations

from collections.abc import Callable

from app.cv.models import TextChunk
from app.schemas.llm_evidence import (
    LlmChunkExtractionResult,
    LlmEvidenceItem,
)


class FakeEvidenceExtractionProvider:
    """Deterministic provider used for tests and local development."""

    def __init__(
        self,
        responder: (
            Callable[[TextChunk], LlmChunkExtractionResult] | None
        ) = None,
    ) -> None:
        self._responder = responder

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-evidence-model"

    async def extract_evidence(
        self,
        chunk: TextChunk,
    ) -> LlmChunkExtractionResult:
        if self._responder is not None:
            return self._responder(chunk)

        evidence_type = self._map_section_to_type(chunk.section)

        return LlmChunkExtractionResult(
            evidence=[
                LlmEvidenceItem(
                    evidence_type=evidence_type,
                    title=self._build_title(chunk),
                    content=chunk.text,
                    confidence=1.0,
                    metadata={
                        "fake_provider": True,
                    },
                )
            ]
        )

    @staticmethod
    def _map_section_to_type(section: str | None) -> str:
        mapping: dict[str | None, str] = {
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

        return mapping.get(section, "general")

    @staticmethod
    def _build_title(chunk: TextChunk) -> str:
        if chunk.section is None:
            return f"General evidence from chunk {chunk.index + 1}"

        readable_section = chunk.section.replace("_", " ").title()

        return f"{readable_section} evidence from chunk {chunk.index + 1}"