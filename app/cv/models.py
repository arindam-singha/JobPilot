from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Section:
    """A detected section inside normalized candidate-document text.

    Offsets refer to character positions in the normalized document text.
    The heading line itself is not included in the section text.
    """

    name: str | None
    text: str
    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Section text must not be empty")

        if self.start_offset < 0:
            raise ValueError("Section start_offset must be greater than or equal to zero")

        if self.end_offset <= self.start_offset:
            raise ValueError("Section end_offset must be greater than start_offset")


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A deterministic chunk derived from normalized candidate-document text.

    Offsets refer to character positions in the normalized document text.
    """

    index: int
    section: str | None
    text: str
    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Chunk index must be greater than or equal to zero")

        if not self.text.strip():
            raise ValueError("Chunk text must not be empty")

        if self.start_offset < 0:
            raise ValueError("Chunk start_offset must be greater than or equal to zero")

        if self.end_offset <= self.start_offset:
            raise ValueError("Chunk end_offset must be greater than start_offset")