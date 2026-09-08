from __future__ import annotations

import html
import re
from datetime import datetime
from io import BytesIO

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

from app.schemas.tailored_resume import TailoredResumeDraft


class TailoredResumePdfRenderingError(Exception):
    """Raised when a tailored resume cannot be rendered as PDF."""


class TailoredResumePdfRenderer:
    """Render trusted structured fields as a single-column ATS PDF."""

    _SPACE = re.compile(r"\s+")
    _INK = HexColor("#152331")
    _MUTED = HexColor("#536575")
    _ACCENT = HexColor("#176b8a")
    _ACCENT_DARK = HexColor("#103f59")

    def __init__(self) -> None:
        styles = getSampleStyleSheet()
        self.name = ParagraphStyle(
            "ResumeName",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=25,
            textColor=self._ACCENT_DARK,
            alignment=TA_CENTER,
            spaceAfter=3,
        )
        self.role = ParagraphStyle(
            "ResumeRole",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=self._ACCENT,
            alignment=TA_CENTER,
            spaceAfter=3,
        )
        self.contact = ParagraphStyle(
            "ResumeContact",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=self._MUTED,
            alignment=TA_CENTER,
            spaceAfter=7,
        )
        self.heading = ParagraphStyle(
            "ResumeHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13,
            textColor=self._ACCENT_DARK,
            spaceBefore=8,
            spaceAfter=2,
            keepWithNext=True,
        )
        self.entry = ParagraphStyle(
            "ResumeEntry",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.6,
            leading=12,
            textColor=self._INK,
            spaceBefore=4,
            spaceAfter=2,
            keepWithNext=True,
        )
        self.body = ParagraphStyle(
            "ResumeBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=11.5,
            textColor=self._INK,
            spaceAfter=3,
        )
        self.bullet = ParagraphStyle(
            "ResumeBullet",
            parent=self.body,
            leftIndent=10,
            firstLineIndent=-7,
            spaceAfter=2,
        )

    def render(self, draft: TailoredResumeDraft) -> bytes:
        output = BytesIO()
        document = SimpleDocTemplate(
            output,
            pagesize=A4,
            leftMargin=13 * mm,
            rightMargin=13 * mm,
            topMargin=11 * mm,
            bottomMargin=11 * mm,
            title=f"{draft.header.full_name} - {draft.target_title}",
            author=draft.header.full_name,
            subject="Tailored Resume",
        )
        story: list[object] = []
        self._header(story, draft)

        if draft.verified_summary:
            self._section(story, "Professional Summary")
            self._paragraph(story, draft.verified_summary)

        if draft.skill_groups:
            self._section(story, "Core Skills")
            for group in draft.skill_groups:
                skills = ", ".join(item.strip() for item in group.skills if item.strip())
                if skills:
                    self._paragraph(
                        story,
                        f"<b>{self._safe(group.category)}:</b> "
                        f"{self._safe(skills)}",
                    )

        if draft.experiences:
            self._section(story, "Professional Experience")
            for item in draft.experiences:
                organization = self._join(item.company, item.location)
                dates = self._date_range(item.start_date, item.end_date, item.is_current)
                self._entry_title(story, item.role, organization, dates)
                for bullet in item.bullets:
                    self._bullet(story, bullet)

        if draft.projects:
            self._section(story, "Selected Projects")
            for item in draft.projects:
                detail = item.technologies or ""
                self._entry_title(story, item.name, detail, "")
                self._optional_bullet(story, item.description)
                self._optional_bullet(story, item.achievements)

        if draft.education:
            self._section(story, "Education")
            for item in draft.education:
                degree = self._join(item.degree, item.field_of_study)
                institution = self._join(item.institution, item.location)
                dates = self._date_range(item.start_date, item.end_date, False)
                self._entry_title(story, degree, institution, dates)
                if item.description:
                    self._paragraph(story, item.description)

        if draft.publications:
            self._section(story, "Selected Publications")
            for item in draft.publications:
                parts = [item.title, item.venue, self._format_date(item.publication_date)]
                text = " | ".join(part for part in parts if part)
                if item.description:
                    text = f"{text} - {item.description}"
                self._bullet(story, text)

        if draft.certifications:
            self._section(story, "Certifications")
            for item in draft.certifications:
                text = item.name
                if item.issuing_organization:
                    text = f"{text} - {item.issuing_organization}"
                if item.issue_date:
                    text = f"{text} ({self._format_date(item.issue_date)})"
                self._bullet(story, text)

        if draft.achievements:
            self._section(story, "Awards and Recognition")
            for item in draft.achievements:
                text = item.title
                if item.date:
                    text = f"{text} ({self._format_date(item.date)})"
                if item.description:
                    text = f"{text} - {item.description}"
                self._bullet(story, text)

        try:
            document.build(story)
        except Exception as exc:
            raise TailoredResumePdfRenderingError("Failed to build resume PDF") from exc

        result = output.getvalue()
        if not result.startswith(b"%PDF-"):
            raise TailoredResumePdfRenderingError("Renderer did not produce a PDF")
        return result

    def _header(self, story: list[object], draft: TailoredResumeDraft) -> None:
        name = self._safe(draft.header.full_name)
        title = self._safe(draft.target_title)
        if not name or not title:
            raise TailoredResumePdfRenderingError("Resume header is incomplete")
        story.append(Paragraph(name, self.name))
        story.append(Paragraph(title, self.role))
        contacts = [
            draft.header.location,
            draft.header.phone,
            draft.header.email,
            self._label("LinkedIn", draft.header.linkedin_url),
            self._label("GitHub", draft.header.github_url),
            self._label("Portfolio", draft.header.portfolio_url),
        ]
        values = [self._safe(item) for item in contacts if item and item.strip()]
        if values:
            story.append(Paragraph(" | ".join(values), self.contact))

    def _section(self, story: list[object], heading: str) -> None:
        story.append(Spacer(1, 2))
        story.append(Paragraph(self._safe(heading.upper()), self.heading))
        story.append(HRFlowable(width="100%", thickness=1.2, color=self._ACCENT, spaceAfter=3))

    def _entry_title(self, story: list[object], title: str, organization: str, dates: str) -> None:
        parts = [f"<b>{self._safe(title)}</b>"]
        if organization:
            parts.append(self._safe(organization))
        if dates:
            parts.append(self._safe(dates))
        story.append(Paragraph(" | ".join(parts), self.entry))

    def _paragraph(self, story: list[object], value: str) -> None:
        if value.strip():
            if value.lstrip().startswith("<b>"):
                story.append(Paragraph(value, self.body))
            else:
                story.append(Paragraph(self._safe(value), self.body))

    def _bullet(self, story: list[object], value: str) -> None:
        if value.strip():
            story.append(Paragraph(f"- {self._safe(value)}", self.bullet))

    def _optional_bullet(self, story: list[object], value: str | None) -> None:
        if value:
            self._bullet(story, value)

    @classmethod
    def _date_range(cls, start: str | None, end: str | None, current: bool) -> str:
        start_text = cls._format_date(start)
        end_text = "Present" if current else cls._format_date(end)
        if start_text and end_text:
            return f"{start_text} - {end_text}"
        return start_text or end_text

    @staticmethod
    def _format_date(value: str | None) -> str:
        if not value:
            return ""
        try:
            return datetime.strptime(value, "%Y-%m-%d").strftime("%b %Y")
        except ValueError:
            return value.strip()

    @staticmethod
    def _join(first: str, second: str | None) -> str:
        return ", ".join(item.strip() for item in (first, second) if item and item.strip())

    @staticmethod
    def _label(label: str, value: str | None) -> str | None:
        return f"{label}: {value.strip()}" if value and value.strip() else None

    @classmethod
    def _safe(cls, value: object) -> str:
        normalized = cls._SPACE.sub(" ", str(value)).strip()
        return html.escape(normalized, quote=True)
