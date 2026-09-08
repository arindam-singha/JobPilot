from __future__ import annotations

import pytest
from app.cv.models import Section, TextChunk
from app.cv.text_processing import (
    build_chunks,
    detect_sections,
    normalize_text,
    split_into_chunks,
)


def test_normalize_text_converts_line_endings_and_removes_nulls() -> None:
    raw = "Summary\r\nPython\x00 developer\rExperience\r\nEngineer"

    result = normalize_text(raw)

    assert result == "Summary\nPython developer\nExperience\nEngineer"


def test_normalize_text_removes_trailing_whitespace() -> None:
    raw = "Summary   \nPython\t \nEngineer"

    result = normalize_text(raw)

    assert result == "Summary\nPython\nEngineer"


def test_normalize_text_collapses_excessive_blank_lines() -> None:
    raw = "Summary\n\n\n\nPython developer"

    result = normalize_text(raw)

    assert result == "Summary\n\nPython developer"


def test_normalize_text_preserves_capitalization() -> None:
    raw = "Python PyTorch ROS2"

    result = normalize_text(raw)

    assert result == "Python PyTorch ROS2"


def test_normalize_text_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_text(None)  # type: ignore[arg-type]


def test_detect_sections_recognizes_case_insensitive_headings() -> None:
    text = normalize_text(
        """
        PROFESSIONAL SUMMARY:
        Robotics engineer.

        Technical Skills:
        Python and PyTorch.

        work experience
        Lead Engineer.
        """
    )

    sections = detect_sections(text)

    assert [section.name for section in sections] == [
        "professional_summary",
        "technical_skills",
        "experience",
    ]

    assert sections[0].text == "Robotics engineer."
    assert sections[1].text == "Python and PyTorch."
    assert sections[2].text == "Lead Engineer."


def test_detect_sections_preserves_text_before_first_heading() -> None:
    text = normalize_text(
        """
        Ada Lovelace

        SUMMARY:
        First computer programmer.
        """
    )

    sections = detect_sections(text)

    assert len(sections) == 2
    assert sections[0].name is None
    assert sections[0].text == "Ada Lovelace"
    assert sections[1].name == "summary"


def test_detect_sections_tolerates_heading_trailing_colon() -> None:
    text = normalize_text(
        """
        EDUCATION:
        PhD in Robotics.
        """
    )

    sections = detect_sections(text)

    assert len(sections) == 1
    assert sections[0].name == "education"
    assert sections[0].text == "PhD in Robotics."


def test_detect_sections_does_not_treat_normal_sentence_as_heading() -> None:
    text = normalize_text(
        """
        I have ten years of professional experience in robotics.
        I build computer vision systems.
        """
    )

    sections = detect_sections(text)

    assert len(sections) == 1
    assert sections[0].name is None
    assert sections[0].text == (
        "I have ten years of professional experience in robotics.\n"
        "I build computer vision systems."
    )


def test_detect_sections_returns_empty_list_for_empty_text() -> None:
    assert detect_sections("") == []
    assert detect_sections("   ") == []


def test_detect_sections_preserves_source_order() -> None:
    text = normalize_text(
        """
        SKILLS:
        Python

        EXPERIENCE:
        Engineer

        EDUCATION:
        PhD
        """
    )

    sections = detect_sections(text)

    assert [section.name for section in sections] == [
        "skills",
        "experience",
        "education",
    ]


def test_section_validates_empty_text() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Section(
            name="skills",
            text="   ",
            start_offset=0,
            end_offset=3,
        )


def test_section_validates_negative_start_offset() -> None:
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        Section(
            name="skills",
            text="Python",
            start_offset=-1,
            end_offset=6,
        )


def test_section_validates_end_offset() -> None:
    with pytest.raises(ValueError, match="greater than start_offset"):
        Section(
            name="skills",
            text="Python",
            start_offset=5,
            end_offset=5,
        )


def test_text_chunk_validates_negative_index() -> None:
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        TextChunk(
            index=-1,
            section="skills",
            text="Python",
            start_offset=0,
            end_offset=6,
        )


def test_split_into_chunks_returns_one_chunk_for_small_section() -> None:
    text = "Python, PyTorch and FastAPI"
    sections = [
        Section(
            name="skills",
            text=text,
            start_offset=0,
            end_offset=len(text),
        )
    ]

    chunks = split_into_chunks(
        text,
        sections,
        max_chunk_characters=100,
        chunk_overlap_characters=10,
    )

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].section == "skills"
    assert chunks[0].text == text
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(text)


def test_split_into_chunks_splits_oversized_text() -> None:
    text = (
        "Python development and machine learning experience. "
        "Built computer vision systems for manufacturing. "
        "Deployed models using Docker and Triton Inference Server."
    )

    sections = [
        Section(
            name="experience",
            text=text,
            start_offset=0,
            end_offset=len(text),
        )
    ]

    chunks = split_into_chunks(
        text,
        sections,
        max_chunk_characters=70,
        chunk_overlap_characters=10,
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 70 for chunk in chunks)
    assert all(chunk.text for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_split_into_chunks_preserves_section_names() -> None:
    text = "Python\n\nEngineer"
    sections = [
        Section(
            name="skills",
            text="Python",
            start_offset=0,
            end_offset=6,
        ),
        Section(
            name="experience",
            text="Engineer",
            start_offset=8,
            end_offset=16,
        ),
    ]

    chunks = split_into_chunks(
        text,
        sections,
        max_chunk_characters=100,
        chunk_overlap_characters=10,
    )

    assert [chunk.section for chunk in chunks] == [
        "skills",
        "experience",
    ]


def test_split_into_chunks_rejects_non_positive_maximum() -> None:
    with pytest.raises(
        ValueError,
        match="max_chunk_characters must be greater than zero",
    ):
        split_into_chunks(
            "Python",
            [],
            max_chunk_characters=0,
            chunk_overlap_characters=0,
        )


def test_split_into_chunks_rejects_negative_overlap() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to zero",
    ):
        split_into_chunks(
            "Python",
            [],
            max_chunk_characters=100,
            chunk_overlap_characters=-1,
        )


def test_split_into_chunks_rejects_overlap_equal_to_maximum() -> None:
    with pytest.raises(
        ValueError,
        match="must be smaller",
    ):
        split_into_chunks(
            "Python",
            [],
            max_chunk_characters=100,
            chunk_overlap_characters=100,
        )


def test_split_into_chunks_rejects_overlap_greater_than_maximum() -> None:
    with pytest.raises(
        ValueError,
        match="must be smaller",
    ):
        split_into_chunks(
            "Python",
            [],
            max_chunk_characters=100,
            chunk_overlap_characters=101,
        )


def test_split_into_chunks_rejects_invalid_section_offsets() -> None:
    section = Section(
        name="skills",
        text="Python",
        start_offset=0,
        end_offset=6,
    )

    with pytest.raises(ValueError, match="outside the supplied text"):
        split_into_chunks(
            "Py",
            [section],
            max_chunk_characters=100,
            chunk_overlap_characters=10,
        )


def test_build_chunks_returns_empty_list_for_empty_input() -> None:
    assert build_chunks("") == []
    assert build_chunks("   \n\n") == []


def test_build_chunks_detects_sections_and_builds_chunks() -> None:
    raw = """
    Ada Lovelace

    PROFESSIONAL SUMMARY:
    First computer programmer.

    TECHNICAL SKILLS:
    Mathematics, algorithms and analytical engines.

    EXPERIENCE:
    Developed algorithms for the Analytical Engine.
    """

    chunks = build_chunks(
        raw,
        max_chunk_characters=100,
        chunk_overlap_characters=10,
    )

    assert [chunk.section for chunk in chunks] == [
        None,
        "professional_summary",
        "technical_skills",
        "experience",
    ]

    assert chunks[0].text == "Ada Lovelace"
    assert chunks[1].text == "First computer programmer."


def test_build_chunks_is_deterministic() -> None:
    raw = """
    SKILLS:
    Python, PyTorch, FastAPI, Docker and PostgreSQL.

    EXPERIENCE:
    Built machine learning platforms for manufacturing applications.
    """

    first_result = build_chunks(
        raw,
        max_chunk_characters=50,
        chunk_overlap_characters=10,
    )

    second_result = build_chunks(
        raw,
        max_chunk_characters=50,
        chunk_overlap_characters=10,
    )

    assert first_result == second_result


def test_chunk_offsets_map_back_to_normalized_text() -> None:
    raw = """
    SUMMARY:
    Robotics and AI engineer.

    SKILLS:
    Python and PyTorch.
    """

    normalized = normalize_text(raw)

    chunks = build_chunks(
        raw,
        max_chunk_characters=100,
        chunk_overlap_characters=10,
    )

    for chunk in chunks:
        assert normalized[chunk.start_offset:chunk.end_offset] == chunk.text


def test_chunk_indexes_are_contiguous_across_sections() -> None:
    raw = """
    SUMMARY:
    Robotics engineer.

    SKILLS:
    Python.

    EDUCATION:
    PhD in Robotics.
    """

    chunks = build_chunks(
        raw,
        max_chunk_characters=100,
        chunk_overlap_characters=10,
    )

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))