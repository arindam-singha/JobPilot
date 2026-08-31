from __future__ import annotations

from app.llm.cover_letter_provider import CoverLetterGenerationProviderError
from app.schemas.cover_letter import (
    CoverLetterGenerationContext,
    GroundedCoverLetterParagraph,
    TailoredCoverLetterContent,
)


class FakeCoverLetterGenerationProvider:
    """Deterministic provider used by tests and local smoke checks."""

    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-cover-letter-v1"

    async def generate_cover_letter(
        self,
        context: CoverLetterGenerationContext,
    ) -> TailoredCoverLetterContent:
        if self.should_fail:
            raise CoverLetterGenerationProviderError("Fake provider failure")

        evidence = context.grounding.selected_evidence[0]
        paragraph = GroundedCoverLetterParagraph(
            text=evidence.content,
            evidence_ids=[evidence.evidence_id],
        )
        return TailoredCoverLetterContent(
            subject=f"Application for {context.job_title}",
            salutation="Dear Hiring Manager,",
            opening=paragraph,
            body_paragraphs=[paragraph],
            closing=paragraph,
            sign_off=f"Sincerely,\n{context.header.full_name}",
        )
