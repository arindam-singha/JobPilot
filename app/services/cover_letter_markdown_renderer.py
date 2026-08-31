from __future__ import annotations

import re

from app.schemas.cover_letter import TailoredCoverLetterDraft


class CoverLetterMarkdownRenderingError(Exception):
    pass


class CoverLetterMarkdownRenderer:
    _WHITESPACE = re.compile(r"\s+")

    def render(self, draft: TailoredCoverLetterDraft) -> str:
        contact = [
            draft.header.email,
            draft.header.phone,
            draft.header.location,
            draft.header.linkedin_url,
        ]
        lines = [f"# {self._clean(draft.header.full_name)}"]
        cleaned_contact = [self._clean(item) for item in contact if item]
        if cleaned_contact:
            lines.extend(["", " | ".join(cleaned_contact)])
        lines.extend(
            [
                "",
                f"**Subject: {self._clean(draft.subject)}**",
                "",
                self._clean(draft.salutation),
                "",
                self._clean(draft.opening.text),
            ]
        )
        for paragraph in draft.body_paragraphs:
            lines.extend(["", self._clean(paragraph.text)])
        lines.extend(
            [
                "",
                self._clean(draft.closing.text),
                "",
                draft.sign_off.strip(),
            ]
        )
        result = "\n".join(lines).strip()
        if not result:
            raise CoverLetterMarkdownRenderingError("Rendered cover letter is empty")
        return f"{result}\n"

    def _clean(self, value: str) -> str:
        return self._WHITESPACE.sub(" ", value).strip()
