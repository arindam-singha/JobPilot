from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence

from app.cv.models import Section, TextChunk

DEFAULT_MAX_CHUNK_CHARACTERS = 2000
DEFAULT_CHUNK_OVERLAP_CHARACTERS = 200


# Only exact heading lines are recognized. This prevents a sentence such as
# "I have ten years of professional experience" from becoming a section heading.
_SECTION_HEADINGS: dict[str, str] = {
    "summary": "summary",
    "professional summary": "professional_summary",
    "profile": "profile",
    "career profile": "profile",
    "objective": "objective",
    "career objective": "objective",
    "skills": "skills",
    "technical skills": "technical_skills",
    "core skills": "skills",
    "core competencies": "core_competencies",
    "competencies": "core_competencies",
    "experience": "experience",
    "professional experience": "experience",
    "work experience": "experience",
    "employment history": "experience",
    "work history": "experience",
    "education": "education",
    "academic background": "education",
    "academic qualifications": "education",
    "projects": "projects",
    "professional projects": "projects",
    "personal projects": "projects",
    "selected generative ai & machine learning projects": "projects",
    "selected projects": "projects",
    "publications": "publications",
    "research publications": "publications",
    "selected publications": "publications",
    "research papers": "publications",
    "certifications": "certifications",
    "certificates": "certifications",
    "achievements": "achievements",
    "awards": "awards",
    "awards and achievements": "achievements",
    "awards & recognition": "achievements",
    "awards and recognition": "achievements",
    "additional information": "additional_information",
}


# def normalize_text(text: str) -> str:
#     """Normalize extracted CV text without destructively rewriting its content.

#     Processing rules:

#     - Convert CRLF and CR line endings to LF.
#     - Remove null characters.
#     - Remove trailing spaces and tabs from each line.
#     - Remove leading and trailing blank lines.
#     - Collapse three or more consecutive newlines to two newlines.
#     - Preserve capitalization and meaningful paragraph boundaries.
#     """

#     if not isinstance(text, str):
#         raise TypeError("text must be a string")

#     normalized = text.replace("\r\n", "\n").replace("\r", "\n")
#     normalized = normalized.replace("\x00", "")

#     lines = [line.rstrip(" \t") for line in normalized.split("\n")]
#     normalized = "\n".join(lines).strip()

#     # Preserve one blank line between paragraphs while removing excessive gaps.
#     normalized = re.sub(r"\n{3,}", "\n\n", normalized)

#     return normalized

def normalize_text(text: str) -> str:
    """Normalize extracted CV text without destructively rewriting its content.

    Processing rules:

    - Remove common indentation from multiline input.
    - Convert CRLF and CR line endings to LF.
    - Remove null characters.
    - Remove trailing spaces and tabs from each line.
    - Remove leading and trailing blank lines.
    - Collapse three or more consecutive newlines to two newlines.
    - Preserve capitalization and meaningful paragraph boundaries.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = textwrap.dedent(text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "")

    lines = [line.rstrip(" \t") for line in normalized.split("\n")]
    normalized = "\n".join(lines).strip()

    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized


def detect_sections(text: str) -> list[Section]:
    """Detect common CV sections in normalized text.

    The supplied text should normally be the output of ``normalize_text``.
    Heading matching is case-insensitive and tolerates a trailing colon.

    Text before the first recognized heading is returned as a section whose
    name is ``None``.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        return []

    sections: list[Section] = []

    current_name: str | None = None
    current_start = 0
    cursor = 0

    for line in text.splitlines(keepends=True):
        line_start = cursor
        line_end = cursor + len(line)
        cursor = line_end

        heading_name = _match_section_heading(line)

        if heading_name is None:
            continue

        _append_section(
            sections=sections,
            document_text=text,
            name=current_name,
            raw_start=current_start,
            raw_end=line_start,
        )

        current_name = heading_name
        current_start = line_end

    _append_section(
        sections=sections,
        document_text=text,
        name=current_name,
        raw_start=current_start,
        raw_end=len(text),
    )

    # A document containing only an unrecognized heading or ordinary text
    # should still remain available as one unclassified section.
    if not sections and text.strip():
        start_offset, end_offset = _trim_span(text, 0, len(text))
        sections.append(
            Section(
                name=None,
                text=text[start_offset:end_offset],
                start_offset=start_offset,
                end_offset=end_offset,
            )
        )

    return sections


def split_into_chunks(
    text: str,
    sections: Sequence[Section],
    *,
    max_chunk_characters: int = DEFAULT_MAX_CHUNK_CHARACTERS,
    chunk_overlap_characters: int = DEFAULT_CHUNK_OVERLAP_CHARACTERS,
) -> list[TextChunk]:
    """Split detected sections into deterministic chunks.

    Chunking preference is:

    1. Section boundaries
    2. Paragraph boundaries
    3. Line boundaries
    4. Whitespace boundaries
    5. Hard character boundary when no safer boundary exists

    Chunk offsets refer to positions in ``text``.
    """

    _validate_chunk_configuration(
        max_chunk_characters=max_chunk_characters,
        chunk_overlap_characters=chunk_overlap_characters,
    )

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    chunks: list[TextChunk] = []

    for section in sections:
        if section.start_offset < 0 or section.end_offset > len(text):
            raise ValueError("Section offsets are outside the supplied text")

        if section.end_offset <= section.start_offset:
            raise ValueError("Section end_offset must be greater than start_offset")

        section_start = section.start_offset
        section_end = section.end_offset
        chunk_start = section_start

        while chunk_start < section_end:
            proposed_end = min(
                chunk_start + max_chunk_characters,
                section_end,
            )

            chunk_end = _find_chunk_end(
                text=text,
                start=chunk_start,
                proposed_end=proposed_end,
                absolute_end=section_end,
            )

            trimmed_start, trimmed_end = _trim_span(
                text,
                chunk_start,
                chunk_end,
            )

            if trimmed_end > trimmed_start:
                chunks.append(
                    TextChunk(
                        index=len(chunks),
                        section=section.name,
                        text=text[trimmed_start:trimmed_end],
                        start_offset=trimmed_start,
                        end_offset=trimmed_end,
                    )
                )

            if chunk_end >= section_end:
                break

            next_start = _calculate_next_chunk_start(
                text=text,
                current_start=chunk_start,
                current_end=chunk_end,
                section_end=section_end,
                overlap=chunk_overlap_characters,
            )

            if next_start <= chunk_start:
                # Defensive protection against an infinite loop.
                next_start = chunk_end

            chunk_start = next_start

    return chunks


def build_chunks(
    text: str,
    *,
    max_chunk_characters: int = DEFAULT_MAX_CHUNK_CHARACTERS,
    chunk_overlap_characters: int = DEFAULT_CHUNK_OVERLAP_CHARACTERS,
) -> list[TextChunk]:
    """Normalize text, detect CV sections and create deterministic chunks."""

    normalized_text = normalize_text(text)

    if not normalized_text:
        return []

    sections = detect_sections(normalized_text)

    return split_into_chunks(
        normalized_text,
        sections,
        max_chunk_characters=max_chunk_characters,
        chunk_overlap_characters=chunk_overlap_characters,
    )


def _match_section_heading(line: str) -> str | None:
    """Return the canonical section name when a line is an exact heading."""

    candidate = line.strip()

    if not candidate:
        return None

    # Permit headings such as "TECHNICAL SKILLS:".
    candidate = candidate.removesuffix(":").strip()

    # Normalize internal heading whitespace.
    candidate = re.sub(r"\s+", " ", candidate).casefold()

    return _SECTION_HEADINGS.get(candidate)


def _append_section(
    *,
    sections: list[Section],
    document_text: str,
    name: str | None,
    raw_start: int,
    raw_end: int,
) -> None:
    """Append a non-empty section after trimming its surrounding whitespace."""

    start_offset, end_offset = _trim_span(
        document_text,
        raw_start,
        raw_end,
    )

    if end_offset <= start_offset:
        return

    sections.append(
        Section(
            name=name,
            text=document_text[start_offset:end_offset],
            start_offset=start_offset,
            end_offset=end_offset,
        )
    )


def _trim_span(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    """Trim surrounding whitespace while preserving offsets."""

    while start < end and text[start].isspace():
        start += 1

    while end > start and text[end - 1].isspace():
        end -= 1

    return start, end


def _validate_chunk_configuration(
    *,
    max_chunk_characters: int,
    chunk_overlap_characters: int,
) -> None:
    if max_chunk_characters <= 0:
        raise ValueError("max_chunk_characters must be greater than zero")

    if chunk_overlap_characters < 0:
        raise ValueError(
            "chunk_overlap_characters must be greater than or equal to zero"
        )

    if chunk_overlap_characters >= max_chunk_characters:
        raise ValueError(
            "chunk_overlap_characters must be smaller than "
            "max_chunk_characters"
        )


def _find_chunk_end(
    *,
    text: str,
    start: int,
    proposed_end: int,
    absolute_end: int,
) -> int:
    """Find a safe chunk boundary no later than ``proposed_end``."""

    if proposed_end >= absolute_end:
        return absolute_end

    # Do not choose a very early boundary that would create tiny chunks.
    minimum_preferred_end = start + max(
        1,
        (proposed_end - start) // 2,
    )

    search_window = text[minimum_preferred_end:proposed_end]

    # Prefer paragraph, line and then word boundaries.
    for separator in ("\n\n", "\n", " "):
        relative_position = search_window.rfind(separator)

        if relative_position != -1:
            boundary = (
                minimum_preferred_end
                + relative_position
                + len(separator)
            )

            if boundary > start:
                return boundary

    # A hard split is used only when no safe boundary exists, such as one
    # extremely long URL or uninterrupted string.
    return proposed_end


def _calculate_next_chunk_start(
    *,
    text: str,
    current_start: int,
    current_end: int,
    section_end: int,
    overlap: int,
) -> int:
    """Calculate the deterministic start position of the next chunk."""

    if overlap == 0:
        next_start = current_end
    else:
        next_start = max(
            current_start + 1,
            current_end - overlap,
        )

    # Avoid starting the new chunk in the middle of a word where possible.
    if (
        next_start > 0
        and next_start < section_end
        and not text[next_start - 1].isspace()
        and not text[next_start].isspace()
    ):
        while next_start < section_end and not text[next_start].isspace():
            next_start += 1

    while next_start < section_end and text[next_start].isspace():
        next_start += 1

    return next_start
