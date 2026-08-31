from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


_TOKEN_PATTERN = re.compile(r"[a-z0-9+#.]+")


def normalize_matching_text(value: str) -> str:
    """Normalize text for deterministic requirement matching."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")

    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.casefold()

    replacements = {
        "ros 2": "ros2",
        "ros-2": "ros2",
        "c plus plus": "c++",
        "c sharp": "c#",
        "machine-learning": "machine learning",
        "deep-learning": "deep learning",
        "computer-vision": "computer vision",
        "artificial-intelligence": "artificial intelligence",
    }

    for source, target in replacements.items():
        normalized = normalized.replace(source, target)

    normalized = re.sub(r"[_/|]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9+#.\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def tokenize_matching_text(value: str) -> set[str]:
    """Return normalized tokens suitable for matching."""

    normalized = normalize_matching_text(value)

    if not normalized:
        return set()

    return set(_TOKEN_PATTERN.findall(normalized))


def contains_normalized_phrase(
    requirement: str,
    candidate_text: str,
) -> bool:
    """Check whether candidate text contains the normalized requirement."""

    normalized_requirement = normalize_matching_text(requirement)
    normalized_candidate = normalize_matching_text(candidate_text)

    if not normalized_requirement or not normalized_candidate:
        return False

    pattern = re.compile(
        rf"(?<![a-z0-9+#.])"
        rf"{re.escape(normalized_requirement)}"
        rf"(?![a-z0-9+#.])"
    )

    return pattern.search(normalized_candidate) is not None


def token_coverage(
    requirement: str,
    candidate_text: str,
) -> float:
    """Return the proportion of requirement tokens present in evidence."""

    requirement_tokens = tokenize_matching_text(requirement)

    if not requirement_tokens:
        return 0.0

    candidate_tokens = tokenize_matching_text(candidate_text)

    if not candidate_tokens:
        return 0.0

    matched_tokens = requirement_tokens & candidate_tokens

    return len(matched_tokens) / len(requirement_tokens)


def jaccard_similarity(
    first: str,
    second: str,
) -> float:
    """Return Jaccard similarity between normalized token sets."""

    first_tokens = tokenize_matching_text(first)
    second_tokens = tokenize_matching_text(second)

    if not first_tokens and not second_tokens:
        return 1.0

    if not first_tokens or not second_tokens:
        return 0.0

    intersection = first_tokens & second_tokens
    union = first_tokens | second_tokens

    return len(intersection) / len(union)


def combine_evidence_text(
    values: Iterable[str | None],
) -> str:
    """Combine non-empty evidence fields into searchable text."""

    parts = [
        value.strip()
        for value in values
        if value is not None and value.strip()
    ]

    return "\n".join(parts)