from __future__ import annotations

import re

from app.schemas.tailored_resume import (
    GroundedResumeStatement,
    ResumeHeader,
    TailoredResumeDraft,
)


class TailoredResumeMarkdownRenderingError(Exception):
    """Raised when a resume cannot be rendered."""


class TailoredResumeMarkdownRenderer:
    """Render structured resumes as ATS-friendly Markdown."""

    _WHITESPACE_PATTERN = re.compile(r"\s+")

    _MARKDOWN_SPECIAL_CHARACTERS = frozenset("\\`*_{}[]<>#|")

    def render(
        self,
        draft: TailoredResumeDraft,
    ) -> str:
        lines: list[str] = []

        self._append_header(
            lines,
            draft.header,
            draft.target_title,
        )

        self._append_statement_section(
            lines,
            heading="Professional Summary",
            statements=draft.professional_summary,
        )

        if draft.skills:
            self._append_statement_section(
                lines,
                heading="Skills",
                statements=draft.skills,
            )

        for section in draft.sections:
            self._append_statement_section(
                lines,
                heading=section.heading,
                statements=section.statements,
            )

        rendered = "\n".join(lines).strip()

        if not rendered:
            raise TailoredResumeMarkdownRenderingError("Rendered resume is empty")

        return f"{rendered}\n"

    def _append_header(
        self,
        lines: list[str],
        header: ResumeHeader,
        target_title: str,
    ) -> None:
        full_name = self._escape_text(header.full_name)

        title = self._escape_text(target_title)

        if not full_name:
            raise TailoredResumeMarkdownRenderingError("Resume header name is empty")

        if not title:
            raise TailoredResumeMarkdownRenderingError("Resume target title is empty")

        lines.append(f"# {full_name}")
        lines.append("")
        lines.append(title)

        contact_items = self._contact_items(header)

        if contact_items:
            lines.append("")
            lines.append(" | ".join(contact_items))

        lines.append("")

    def _append_statement_section(
        self,
        lines: list[str],
        *,
        heading: str,
        statements: list[GroundedResumeStatement],
    ) -> None:
        cleaned_heading = self._escape_text(heading)

        if not cleaned_heading:
            raise TailoredResumeMarkdownRenderingError("Resume section heading is empty")

        if not statements:
            return

        lines.append(f"## {cleaned_heading}")
        lines.append("")

        for statement in statements:
            text = self._escape_text(statement.text)

            if not text:
                raise (TailoredResumeMarkdownRenderingError("Resume statement is empty"))

            lines.append(f"- {text}")

        lines.append("")

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
            self._escape_text(item)
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

    def _escape_text(
        self,
        value: str,
    ) -> str:
        normalized = self._normalize_text(value)

        escaped: list[str] = []

        for character in normalized:
            if character in self._MARKDOWN_SPECIAL_CHARACTERS:
                escaped.append("\\")

            escaped.append(character)

        return "".join(escaped)

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
