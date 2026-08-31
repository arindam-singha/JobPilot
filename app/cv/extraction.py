from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Protocol


@dataclass(frozen=True)
class ExtractedText:
    text: str
    page_count: int | None = None
    paragraph_count: int | None = None


class DocumentTextExtractor(Protocol):
    def extract(self, file_content: bytes) -> ExtractedText:
        ...


class PDFTextExtractor:
    """Deterministic PDF text extractor using pypdf."""

    def extract(self, file_content: bytes) -> ExtractedText:
        if not file_content:
            return ExtractedText(text="")

        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_content))
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                pages.append(page_text)

        text = "\n\n".join(pages).strip()
        return ExtractedText(
            text=text,
            page_count=len(reader.pages),
            paragraph_count=len([line for line in text.splitlines() if line.strip()]),
        )


class DOCXTextExtractor:
    """Deterministic DOCX text extractor using python-docx."""

    def extract(self, file_content: bytes) -> ExtractedText:
        if not file_content:
            return ExtractedText(text="")

        from docx import Document

        document = Document(BytesIO(file_content))
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        text = "\n".join(paragraphs)
        return ExtractedText(text=text, paragraph_count=len(paragraphs))


def extract_document_text(file_content: bytes, mime_type: str | None = None) -> ExtractedText:
    normalized_mime = (mime_type or "").lower()

    if normalized_mime == "application/pdf" or file_content.startswith(b"%PDF"):
        return PDFTextExtractor().extract(file_content)
    if (
        normalized_mime
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or file_content.startswith(b"PK\x03\x04")
    ):
        return DOCXTextExtractor().extract(file_content)

    raise ValueError(f"Unsupported document content type for extraction: {mime_type or 'unknown'}")
