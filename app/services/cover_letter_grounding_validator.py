from __future__ import annotations

import re

from app.schemas.cover_letter import TailoredCoverLetterDraft


class CoverLetterGroundingValidationError(Exception):
    """Raised when generated text contains unsupported factual values."""


class CoverLetterGroundingValidator:
    """Validate provenance and reject numeric claims absent from cited evidence."""

    _NUMBER_PATTERN = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?%?(?![\w])")

    def validate(self, draft: TailoredCoverLetterDraft) -> None:
        evidence_by_id = {item.evidence_id: item.content for item in draft.evidence_catalog}
        paragraphs = [draft.opening, *draft.body_paragraphs, draft.closing]
        for paragraph in paragraphs:
            cited_text = " ".join(evidence_by_id[item] for item in paragraph.evidence_ids)
            supported_numbers = set(self._NUMBER_PATTERN.findall(cited_text))
            generated_numbers = set(self._NUMBER_PATTERN.findall(paragraph.text))
            unsupported = generated_numbers - supported_numbers
            if unsupported:
                values = ", ".join(sorted(unsupported))
                raise CoverLetterGroundingValidationError(f"Unsupported numeric claim(s): {values}")
