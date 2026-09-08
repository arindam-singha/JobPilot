from __future__ import annotations

import re
from datetime import datetime

from app.schemas.tailored_resume import ResumeHeader, TailoredResumeDraft


class TailoredResumeMarkdownRenderingError(Exception):
    """Raised when a resume cannot be rendered."""


class TailoredResumeMarkdownRenderer:
    """Render deterministic resume fields as ATS-friendly Markdown."""

    _WHITESPACE_PATTERN = re.compile(r"\s+")
    _SPECIAL = frozenset("\\`*_{}[]<>#|")

    def render(self, draft: TailoredResumeDraft) -> str:
        lines: list[str] = []
        self._header(lines, draft.header, draft.target_title)

        if draft.verified_summary:
            self._heading(lines, "Professional Summary")
            lines.append(self._escape(draft.verified_summary))
            lines.append("")

        if draft.skill_groups:
            self._heading(lines, "Core Skills")
            for group in draft.skill_groups:
                skills = ", ".join(self._escape(item) for item in group.skills if item.strip())
                if skills:
                    lines.append(f"**{self._escape(group.category)}:** {skills}")
            lines.append("")

        if draft.experiences:
            self._heading(lines, "Professional Experience")
            for item in draft.experiences:
                organization = self._join(item.company, item.location)
                lines.append(f"### {self._escape(item.role)} | {self._escape(organization)}")
                date_range = self._date_range(item.start_date, item.end_date, item.is_current)
                if date_range:
                    lines.append(f"**{date_range}**")
                self._bullets(lines, item.bullets)
                lines.append("")

        if draft.projects:
            self._heading(lines, "Selected Projects")
            for item in draft.projects:
                title = item.name
                if item.technologies:
                    title = f"{title} | {item.technologies}"
                lines.append(f"### {self._escape(title)}")
                self._optional_bullet(lines, item.description)
                self._optional_bullet(lines, item.achievements)
                lines.append("")

        if draft.education:
            self._heading(lines, "Education")
            for item in draft.education:
                degree = self._join(item.degree, item.field_of_study)
                institution = self._join(item.institution, item.location)
                lines.append(f"### {self._escape(degree)} | {self._escape(institution)}")
                date_range = self._date_range(item.start_date, item.end_date, False)
                if date_range:
                    lines.append(f"**{date_range}**")
                if item.description:
                    lines.append(self._escape(item.description))
                lines.append("")

        self._publication_section(lines, draft)
        self._certification_section(lines, draft)
        self._achievement_section(lines, draft)

        rendered = "\n".join(lines).strip()
        if not rendered:
            raise TailoredResumeMarkdownRenderingError("Rendered resume is empty")
        return f"{rendered}\n"

    def _header(self, lines: list[str], header: ResumeHeader, target_title: str) -> None:
        name = self._escape(header.full_name)
        title = self._escape(target_title)
        if not name or not title:
            raise TailoredResumeMarkdownRenderingError("Resume header is incomplete")
        lines.extend([f"# {name}", "", title])
        contacts = [
            header.location,
            header.phone,
            header.email,
            self._label("LinkedIn", header.linkedin_url),
            self._label("GitHub", header.github_url),
            self._label("Portfolio", header.portfolio_url),
        ]
        values = [self._escape(item) for item in contacts if item and item.strip()]
        if values:
            lines.extend(["", " | ".join(values)])
        lines.append("")

    def _publication_section(self, lines: list[str], draft: TailoredResumeDraft) -> None:
        if not draft.publications:
            return
        self._heading(lines, "Selected Publications")
        for item in draft.publications:
            parts = [item.title, item.venue, self._format_date(item.publication_date)]
            text = " | ".join(self._escape(part) for part in parts if part)
            if item.description:
                text = f"{text} - {self._escape(item.description)}"
            lines.append(f"- {text}")
        lines.append("")

    def _certification_section(self, lines: list[str], draft: TailoredResumeDraft) -> None:
        if not draft.certifications:
            return
        self._heading(lines, "Certifications")
        for item in draft.certifications:
            text = item.name
            if item.issuing_organization:
                text = f"{text} - {item.issuing_organization}"
            if item.issue_date:
                text = f"{text} ({self._format_date(item.issue_date)})"
            lines.append(f"- {self._escape(text)}")
        lines.append("")

    def _achievement_section(self, lines: list[str], draft: TailoredResumeDraft) -> None:
        if not draft.achievements:
            return
        self._heading(lines, "Awards and Recognition")
        for item in draft.achievements:
            text = item.title
            if item.date:
                text = f"{text} ({self._format_date(item.date)})"
            if item.description:
                text = f"{text} - {item.description}"
            lines.append(f"- {self._escape(text)}")
        lines.append("")

    @staticmethod
    def _heading(lines: list[str], value: str) -> None:
        lines.extend([f"## {value}", ""])

    def _bullets(self, lines: list[str], values: list[str]) -> None:
        for value in values:
            self._optional_bullet(lines, value)

    def _optional_bullet(self, lines: list[str], value: str | None) -> None:
        if value and value.strip():
            lines.append(f"- {self._escape(value)}")

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

    def _escape(self, value: object) -> str:
        normalized = self._WHITESPACE_PATTERN.sub(" ", str(value)).strip()
        result: list[str] = []
        for character in normalized:
            if character in self._SPECIAL:
                result.append("\\")
            result.append(character)
        return "".join(result)
