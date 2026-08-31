from __future__ import annotations

import html
import re
from io import BytesIO

from reportlab.lib.pagesizes import A4  # type: ignore[import-untyped]
from reportlab.lib.styles import (  # type: ignore[import-untyped]
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import inch  # type: ignore[import-untyped]
from reportlab.platypus import (  # type: ignore[import-untyped]
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from app.schemas.cover_letter import TailoredCoverLetterDraft


class CoverLetterPdfRenderingError(Exception):
    pass


class CoverLetterPdfRenderer:
    _WHITESPACE = re.compile(r"\s+")

    def __init__(self) -> None:
        styles = getSampleStyleSheet()
        self.name_style = ParagraphStyle(
            "CoverLetterName", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=16
        )
        self.meta_style = ParagraphStyle(
            "CoverLetterMeta", parent=styles["Normal"], fontSize=9, leading=12
        )
        self.body_style = ParagraphStyle(
            "CoverLetterBody", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=10
        )

    def render(self, draft: TailoredCoverLetterDraft) -> bytes:
        output = BytesIO()
        document = SimpleDocTemplate(
            output,
            pagesize=A4,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.65 * inch,
            bottomMargin=0.65 * inch,
            title=f"Cover Letter - {draft.job_title} at {draft.company}",
            author=draft.header.full_name,
        )
        contact = " | ".join(
            self._safe(item)
            for item in [draft.header.email, draft.header.phone, draft.header.location]
            if item
        )
        story: list[object] = [Paragraph(self._safe(draft.header.full_name), self.name_style)]
        if contact:
            story.extend([Paragraph(contact, self.meta_style), Spacer(1, 12)])
        story.extend(
            [
                Paragraph(f"<b>Subject: {self._safe(draft.subject)}</b>", self.body_style),
                Paragraph(self._safe(draft.salutation), self.body_style),
                Paragraph(self._safe(draft.opening.text), self.body_style),
            ]
        )
        story.extend(
            Paragraph(self._safe(item.text), self.body_style) for item in draft.body_paragraphs
        )
        story.extend(
            [
                Paragraph(self._safe(draft.closing.text), self.body_style),
                Paragraph(self._safe(draft.sign_off).replace("\n", "<br/>"), self.body_style),
            ]
        )
        try:
            document.build(story)
        except Exception as exc:
            raise CoverLetterPdfRenderingError("Failed to build cover-letter PDF") from exc
        result = output.getvalue()
        if not result.startswith(b"%PDF-"):
            raise CoverLetterPdfRenderingError("Renderer did not produce a valid PDF")
        return result

    def _safe(self, value: str) -> str:
        return html.escape(self._WHITESPACE.sub(" ", value).strip())
