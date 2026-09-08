from __future__ import annotations

import html
from datetime import datetime
from urllib.parse import urlparse

from app.schemas.tailored_resume import TailoredResumeDraft


class TailoredResumeHtmlRenderingError(Exception):
    """Raised when a tailored resume cannot be rendered as HTML."""


class TailoredResumeHtmlRenderer:
    """Render a structured resume as single-column ATS-friendly HTML."""

    def render(self, draft: TailoredResumeDraft) -> str:
        name = self._escape(draft.header.full_name)
        target_title = self._escape(draft.target_title)

        if not name:
            raise TailoredResumeHtmlRenderingError(
                "Resume candidate name is empty"
            )

        if not target_title:
            raise TailoredResumeHtmlRenderingError(
                "Resume target title is empty"
            )

        sections = [
            self._summary(draft),
            self._skills(draft),
            self._experiences(draft),
            self._projects(draft),
            self._education(draft),
            self._publications(draft),
            self._certifications(draft),
            self._achievements(draft),
        ]

        rendered_sections = "\n".join(
            section for section in sections if section
        )

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{name} - {target_title} resume">
  <title>{name} | {target_title}</title>
  <style>
    :root {{
      --ink: #152331;
      --muted: #536575;
      --accent: #176b8a;
      --accent-dark: #103f59;
      --line: #c9dce5;
      --paper: #ffffff;
      --canvas: #eaf0f3;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--canvas);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 15px;
      line-height: 1.42;
    }}

    a {{
      color: var(--accent-dark);
      text-decoration: none;
    }}

    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 5;
      padding: 12px;
      text-align: center;
      background: rgba(234, 240, 243, 0.95);
    }}

    .toolbar button {{
      border: 0;
      border-radius: 8px;
      padding: 10px 18px;
      color: white;
      background: var(--accent);
      font-weight: 700;
      cursor: pointer;
    }}

    .resume {{
      width: min(210mm, calc(100% - 28px));
      margin: 10px auto 30px;
      padding: 15mm 16mm;
      background: var(--paper);
      box-shadow: 0 12px 34px rgba(22, 50, 66, 0.15);
    }}

    header {{
      text-align: center;
    }}

    h1 {{
      margin: 0;
      color: var(--accent-dark);
      font-family: Georgia, "Times New Roman", serif;
      font-size: 35px;
      line-height: 1.05;
      letter-spacing: 0.02em;
    }}

    .role {{
      margin: 5px 0 8px;
      color: var(--accent);
      font-family: Georgia, "Times New Roman", serif;
      font-size: 18px;
      font-weight: 700;
    }}

    .contact {{
      color: var(--muted);
      font-size: 13px;
    }}

    section {{
      margin-top: 16px;
    }}

    h2 {{
      margin: 0 0 7px;
      padding-bottom: 3px;
      border-bottom: 2px solid var(--accent);
      color: var(--accent-dark);
      font-family: Georgia, "Times New Roman", serif;
      font-size: 17px;
      line-height: 1.2;
      text-transform: uppercase;
    }}

    h3 {{
      margin: 0;
      font-size: 15.5px;
    }}

    p {{
      margin: 4px 0;
    }}

    ul {{
      margin: 5px 0 0;
      padding-left: 20px;
    }}

    li {{
      margin: 3px 0;
    }}

    .skills {{
      margin: 0;
    }}

    .skill-row {{
      margin: 4px 0;
    }}

    .skill-category {{
      color: var(--accent-dark);
      font-weight: 700;
    }}

    .entry {{
      margin-bottom: 12px;
      break-inside: avoid;
    }}

    .entry-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }}

    .organization {{
      color: var(--muted);
      font-style: italic;
    }}

    time {{
      color: var(--accent-dark);
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
    }}

    .details {{
      color: #2d3d49;
    }}

    @page {{
      size: A4;
      margin: 11mm 13mm;
    }}

    @media print {{
      body {{
        background: white;
        font-size: 10.2pt;
        line-height: 1.3;
      }}

      .toolbar {{
        display: none;
      }}

      .resume {{
        width: auto;
        margin: 0;
        padding: 0;
        box-shadow: none;
      }}

      header,
      .entry {{
        break-inside: avoid;
      }}

      section {{
        margin-top: 10px;
      }}

      h1 {{
        font-size: 25pt;
      }}

      h2 {{
        margin-bottom: 5px;
        font-size: 12.5pt;
      }}

      .role {{
        font-size: 13.5pt;
      }}

      .contact {{
        font-size: 9pt;
      }}

      .entry {{
        margin-bottom: 7px;
      }}

      a {{
        color: inherit;
      }}
    }}
  </style>
</head>
<body>
  <div class="toolbar" aria-label="Resume actions">
    <button type="button" onclick="window.print()">
      Print / Save as PDF
    </button>
  </div>

  <main class="resume">
    <header>
      <h1>{name}</h1>
      <p class="role">{target_title}</p>
      <div class="contact" aria-label="Contact information">{self._contact(draft)}</div>
    </header>

    {rendered_sections}
  </main>
</body>
</html>
"""

    def _summary(self, draft: TailoredResumeDraft) -> str:
        if not draft.verified_summary:
            return ""

        summary = self._escape(draft.verified_summary)

        if not summary:
            return ""

        return self._section(
            "Professional Summary",
            f"<p>{summary}</p>",
        )

    def _skills(self, draft: TailoredResumeDraft) -> str:
        rows: list[str] = []

        for group in draft.skill_groups:
            skills = ", ".join(
                self._escape(skill)
                for skill in group.skills
                if skill.strip()
            )

            if not skills:
                continue

            rows.append(
                '<p class="skill-row">'
                f'<span class="skill-category">'
                f"{self._escape(group.category)}:</span> "
                f"{skills}</p>"
            )

        if not rows:
            return ""

        return self._section("Core Skills", "".join(rows))

    def _experiences(self, draft: TailoredResumeDraft) -> str:
        entries: list[str] = []

        for item in draft.experiences:
            date_range = self._date_range(
                item.start_date,
                item.end_date,
                item.is_current,
            )
            location = (
                f", {self._escape(item.location)}"
                if item.location
                else ""
            )
            bullets = self._bullet_list(item.bullets)

            entries.append(
                '<article class="entry">'
                '<div class="entry-head">'
                "<div>"
                f"<h3>{self._escape(item.role)}</h3> "
                '<span class="organization">'
                f"| {self._escape(item.company)}{location}"
                "</span>"
                "</div>"
                f"{self._time(date_range)}"
                "</div>"
                f"{bullets}"
                "</article>"
            )

        if not entries:
            return ""

        return self._section(
            "Professional Experience",
            "".join(entries),
        )

    def _projects(self, draft: TailoredResumeDraft) -> str:
        entries: list[str] = []

        for item in draft.projects:
            technology = (
                f" | {self._escape(item.technologies)}"
                if item.technologies
                else ""
            )
            details = self._paragraphs(
                item.description,
                item.achievements,
            )

            entries.append(
                '<article class="entry">'
                f"<h3>{self._escape(item.name)}{technology}</h3>"
                f"{details}"
                "</article>"
            )

        if not entries:
            return ""

        return self._section("Selected Projects", "".join(entries))

    def _education(self, draft: TailoredResumeDraft) -> str:
        entries: list[str] = []

        for item in draft.education:
            date_range = self._date_range(
                item.start_date,
                item.end_date,
                False,
            )
            field = (
                f", {self._escape(item.field_of_study)}"
                if item.field_of_study
                else ""
            )
            location = (
                f", {self._escape(item.location)}"
                if item.location
                else ""
            )

            entries.append(
                '<article class="entry">'
                '<div class="entry-head">'
                "<div>"
                f"<h3>{self._escape(item.degree)}{field}</h3> "
                '<span class="organization">'
                f"| {self._escape(item.institution)}{location}"
                "</span>"
                "</div>"
                f"{self._time(date_range)}"
                "</div>"
                f"{self._paragraphs(item.description)}"
                "</article>"
            )

        if not entries:
            return ""

        return self._section("Education", "".join(entries))

    def _publications(self, draft: TailoredResumeDraft) -> str:
        items: list[str] = []

        for item in draft.publications:
            parts = [self._escape(item.title)]

            if item.venue:
                parts.append(self._escape(item.venue))

            if item.publication_date:
                parts.append(self._format_date(item.publication_date))

            text = " | ".join(part for part in parts if part)

            if item.description:
                text += f" - {self._escape(item.description)}"

            items.append(f"<li>{text}</li>")

        if not items:
            return ""

        return self._section(
            "Selected Publications",
            f"<ul>{''.join(items)}</ul>",
        )

    def _certifications(self, draft: TailoredResumeDraft) -> str:
        items: list[str] = []

        for item in draft.certifications:
            text = self._escape(item.name)

            if item.issuing_organization:
                text += (
                    " - "
                    f"{self._escape(item.issuing_organization)}"
                )

            if item.issue_date:
                text += f" ({self._format_date(item.issue_date)})"

            items.append(f"<li>{text}</li>")

        if not items:
            return ""

        return self._section(
            "Certifications",
            f"<ul>{''.join(items)}</ul>",
        )

    def _achievements(self, draft: TailoredResumeDraft) -> str:
        items: list[str] = []

        for item in draft.achievements:
            text = self._escape(item.title)

            if item.date:
                text += f" ({self._format_date(item.date)})"

            if item.description:
                text += f" - {self._escape(item.description)}"

            items.append(f"<li>{text}</li>")

        if not items:
            return ""

        return self._section(
            "Awards and Recognition",
            f"<ul>{''.join(items)}</ul>",
        )

    def _contact(self, draft: TailoredResumeDraft) -> str:
        header = draft.header
        values: list[str] = []

        if header.location:
            values.append(self._escape(header.location))

        if header.phone:
            phone = self._escape(header.phone)
            values.append(f'<a href="tel:{phone}">{phone}</a>')

        if header.email:
            email = self._escape(header.email)
            values.append(
                f'<a href="mailto:{email}">{email}</a>'
            )

        links = [
            ("LinkedIn", header.linkedin_url),
            ("GitHub", header.github_url),
            ("Portfolio", header.portfolio_url),
        ]

        for label, value in links:
            safe_url = self._safe_url(value)

            if safe_url:
                values.append(
                    f'<a href="{safe_url}">{label}</a>'
                )

        return '<span aria-hidden="true"> | </span>'.join(
            f'<span class="contact-item">{value}</span>'
            for value in values
        )

    @staticmethod
    def _section(heading: str, content: str) -> str:
        return (
            "<section>"
            f"<h2>{html.escape(heading)}</h2>"
            f"{content}"
            "</section>"
        )

    def _bullet_list(self, bullets: list[str]) -> str:
        cleaned = [
            self._escape(item)
            for item in bullets
            if item.strip()
        ]

        if not cleaned:
            return ""

        return "<ul>" + "".join(
            f"<li>{item}</li>" for item in cleaned
        ) + "</ul>"

    def _paragraphs(self, *values: str | None) -> str:
        return "".join(
            f'<p class="details">{self._escape(value)}</p>'
            for value in values
            if value and value.strip()
        )

    @classmethod
    def _date_range(
        cls,
        start_date: str | None,
        end_date: str | None,
        is_current: bool,
    ) -> str:
        start = cls._format_date(start_date)
        end = "Present" if is_current else cls._format_date(end_date)

        if start and end:
            return f"{start} - {end}"

        return start or end

    @staticmethod
    def _format_date(value: str | None) -> str:
        if not value:
            return ""

        try:
            parsed = datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return html.escape(value)

        return parsed.strftime("%b %Y")

    @staticmethod
    def _time(value: str) -> str:
        if not value:
            return ""

        return f"<time>{html.escape(value)}</time>"

    @staticmethod
    def _escape(value: object) -> str:
        return html.escape(str(value).strip(), quote=True)

    @classmethod
    def _safe_url(cls, value: str | None) -> str | None:
        if not value:
            return None

        cleaned = value.strip()
        parsed = urlparse(cleaned)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None

        return cls._escape(cleaned)
