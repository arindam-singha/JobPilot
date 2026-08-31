from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from app.core.config import get_settings


class DocumentStorageError(Exception):
    """Base exception for document storage errors."""


class DocumentStorageNotFoundError(DocumentStorageError):
    """Raised when a requested document file does not exist."""


class DocumentStoragePathError(DocumentStorageError):
    """Raised when a storage path is unsafe or escapes the configured storage root."""


class DocumentStorageTypeError(DocumentStorageError):
    """Raised when a document content type is unsupported."""


@dataclass(frozen=True)
class StoredDocument:
    storage_path: str
    file_size: int
    filename: str


class DocumentStorageService:
    """Filesystem abstraction for candidate CV documents."""

    SUPPORTED_CONTENT_TYPES = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    }

    def __init__(self, storage_root: str | Path | None = None) -> None:
        configured_root = Path(storage_root) if storage_root is not None else Path(get_settings().cv_storage_dir)
        self.storage_root = configured_root.resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def _safe_storage_root(self) -> Path:
        return self.storage_root.resolve()

    def _validate_content_type(self, content_type: str) -> str:
        if content_type not in self.SUPPORTED_CONTENT_TYPES:
            raise DocumentStorageTypeError(f"Unsupported document content type: {content_type}")
        return self.SUPPORTED_CONTENT_TYPES[content_type]

    def _resolve_storage_path(self, storage_path: str) -> Path:
        if not storage_path:
            raise DocumentStoragePathError("Storage path is required")

        root = self._safe_storage_root()
        candidate = (root / storage_path).resolve()

        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise DocumentStoragePathError("Storage path escapes the configured storage root") from exc

        return candidate

    def _get_safe_filename(self, document_id: UUID, content_type: str) -> str:
        extension = self._validate_content_type(content_type)
        return f"{document_id}{extension}"

    def save(
        self,
        file_content: bytes,
        document_id: UUID,
        original_filename: str,
        content_type: str,
    ) -> StoredDocument:
        if not isinstance(file_content, (bytes, bytearray)):
            raise TypeError("file_content must be bytes")

        extension = self._validate_content_type(content_type)
        safe_filename = f"{document_id}{extension}"
        storage_path = self._safe_storage_root() / safe_filename

        if storage_path.exists():
            raise DocumentStorageError(f"Storage path already exists for document {document_id}")

        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_path.write_bytes(bytes(file_content))

        relative = path_relative_to_root(storage_path, self._safe_storage_root())
        return StoredDocument(
            storage_path=relative,
            file_size=len(file_content),
            filename=safe_filename,
        )

    def get(self, storage_path: str) -> bytes:
        try:
            resolved = self._resolve_storage_path(storage_path)
        except DocumentStoragePathError as exc:
            raise DocumentStorageNotFoundError(f"Document file not found at storage path: {storage_path}") from exc

        if not resolved.exists() or not resolved.is_file():
            raise DocumentStorageNotFoundError(f"Document file not found at storage path: {storage_path}")
        return resolved.read_bytes()

    def exists(self, storage_path: str) -> bool:
        try:
            resolved = self._resolve_storage_path(storage_path)
        except DocumentStoragePathError:
            return False
        return resolved.exists() and resolved.is_file()

    def delete(self, storage_path: str) -> None:
        try:
            resolved = self._resolve_storage_path(storage_path)
        except DocumentStoragePathError:
            return

        if resolved.exists() and resolved.is_file():
            resolved.unlink()


def path_relative_to_root(path: Path, root: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(root.resolve()))
    except ValueError as exc:
        raise DocumentStoragePathError("Storage path escapes the configured storage root") from exc
