from __future__ import annotations

import html
import re
from io import BytesIO

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from app.schemas.tailored_resume import (
    GroundedResumeStatement,
    ResumeHeader,
    TailoredResumeDraft,
)


class TailoredResumePdfRenderingError(Exception):
    """Raised when a tailored resume cannot be rendered."""


class TailoredResumePdfRenderer:
    """Render a structured resume as a single-column ATS PDF."""

    _WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(self) -> None:
        base_styles = getSampleStyleSheet()

        self._name_style = ParagraphStyle(
            "ResumeName",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=4,
        )

        self._title_style = ParagraphStyle(
            "ResumeTargetTitle",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=4,
        )

        self._contact_style = ParagraphStyle(
            "ResumeContact",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            spaceAfter=8,
        )

        self._heading_style = ParagraphStyle(
            "ResumeSectionHeading",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        )

        self._bullet_style = ParagraphStyle(
            "ResumeBullet",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12,
            leftIndent=12,
            firstLineIndent=-8,
            spaceAfter=3,
        )

    def render(
        self,
        draft: TailoredResumeDraft,
    ) -> bytes:
        output = BytesIO()

        document = SimpleDocTemplate(
            output,
            pagesize=A4,
            rightMargin=0.65 * inch,
            leftMargin=0.65 * inch,
            topMargin=0.55 * inch,
            bottomMargin=0.55 * inch,
            title=(f"{draft.header.full_name} - " f"{draft.target_title}"),
            author=draft.header.full_name,
            subject="Tailored Resume",
        )

        story: list[object] = []

        self._append_header(
            story,
            draft.header,
            draft.target_title,
        )

        self._append_section(
            story,
            heading="Professional Summary",
            statements=draft.professional_summary,
        )

        if draft.skills:
            self._append_section(
                story,
                heading="Skills",
                statements=draft.skills,
            )

        for section in draft.sections:
            self._append_section(
                story,
                heading=section.heading,
                statements=section.statements,
            )

        if not story:
            raise TailoredResumePdfRenderingError("Rendered resume is empty")

        try:
            document.build(story)
        except Exception as exc:
            raise TailoredResumePdfRenderingError("Failed to build tailored resume PDF") from exc

        pdf_bytes = output.getvalue()

        if not pdf_bytes.startswith(b"%PDF-"):
            raise TailoredResumePdfRenderingError("Renderer did not produce a valid PDF")

        return pdf_bytes

    def _append_header(
        self,
        story: list[object],
        header: ResumeHeader,
        target_title: str,
    ) -> None:
        full_name = self._safe_text(header.full_name)

        title = self._safe_text(target_title)

        if not full_name:
            raise TailoredResumePdfRenderingError("Resume header name is empty")

        if not title:
            raise TailoredResumePdfRenderingError("Resume target title is empty")

        story.append(
            Paragraph(
                full_name,
                self._name_style,
            )
        )

        story.append(
            Paragraph(
                title,
                self._title_style,
            )
        )

        contact_items = self._contact_items(header)

        if contact_items:
            story.append(
                Paragraph(
                    " | ".join(contact_items),
                    self._contact_style,
                )
            )

        story.append(Spacer(1, 2))

    def _append_section(
        self,
        story: list[object],
        *,
        heading: str,
        statements: list[GroundedResumeStatement],
    ) -> None:
        if not statements:
            return

        cleaned_heading = self._safe_text(heading)

        if not cleaned_heading:
            raise TailoredResumePdfRenderingError("Resume section heading is empty")

        story.append(
            Paragraph(
                cleaned_heading.upper(),
                self._heading_style,
            )
        )

        for statement in statements:
            cleaned_statement = self._safe_text(statement.text)

            if not cleaned_statement:
                raise TailoredResumePdfRenderingError("Resume statement is empty")

            story.append(
                Paragraph(
                    f"- {cleaned_statement}",
                    self._bullet_style,
                )
            )

    def _contact_items(
        self,
        header: ResumeHeader,
    ) -> list[str]:
        raw_items = [
            header.email,
            header.phone,
            header.location,
            self._labelled_value(
                "LinkedIn",
                header.linkedin_url,
            ),
            self._labelled_value(
                "GitHub",
                header.github_url,
            ),
            self._labelled_value(
                "Portfolio",
                header.portfolio_url,
            ),
        ]

        return [
            self._safe_text(item)
            for item in raw_items
            if item is not None and self._normalize_text(item)
        ]

    @staticmethod
    def _labelled_value(
        label: str,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        if not cleaned:
            return None

        return f"{label}: {cleaned}"

    def _safe_text(
        self,
        value: str,
    ) -> str:
        return html.escape(self._normalize_text(value))

    def _normalize_text(
        self,
        value: str,
    ) -> str:
        without_controls = "".join(
            character for character in value if character in "\t\n\r" or ord(character) >= 32
        )

        return self._WHITESPACE_PATTERN.sub(
            " ",
            without_controls,
        ).strip()
