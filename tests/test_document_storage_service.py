from uuid import uuid4

import pytest
from app.services.document_storage_service import (
    DocumentStorageNotFoundError,
    DocumentStorageService,
    DocumentStorageTypeError,
)


@pytest.fixture
def storage_service(tmp_path):
    return DocumentStorageService(storage_root=tmp_path / "cv-storage")


def test_save_pdf_successfully(storage_service):
    doc_id = uuid4()
    payload = b"%PDF-1.4\n%%EOF"

    saved = storage_service.save(payload, doc_id, "resume report.pdf", "application/pdf")

    assert saved.storage_path.endswith(f"{doc_id}.pdf")
    assert saved.file_size == len(payload)
    assert saved.filename == f"{doc_id}.pdf"
    assert storage_service.exists(saved.storage_path)


def test_save_docx_successfully(storage_service):
    doc_id = uuid4()
    payload = b"PK\x03\x04DOCX"

    saved = storage_service.save(payload, doc_id, "resume final.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    assert saved.storage_path.endswith(f"{doc_id}.docx")
    assert saved.file_size == len(payload)
    assert saved.filename == f"{doc_id}.docx"


def test_returned_file_size_is_correct(storage_service):
    doc_id = uuid4()
    payload = b"hello world"

    saved = storage_service.save(payload, doc_id, "resume.txt", "application/pdf")

    assert saved.file_size == 11
    assert storage_service.get(saved.storage_path) == payload


def test_stored_filename_is_based_on_uuid_not_original_filename(storage_service):
    doc_id = uuid4()
    payload = b"content"

    saved = storage_service.save(payload, doc_id, "my very large resume with spaces.pdf", "application/pdf")

    assert saved.filename == f"{doc_id}.pdf"
    assert saved.filename != "my very large resume with spaces.pdf"
    assert storage_service.exists(saved.storage_path)


def test_original_filename_containing_spaces_is_handled_safely(storage_service):
    doc_id = uuid4()
    payload = b"content"

    saved = storage_service.save(payload, doc_id, "resume final version.pdf", "application/pdf")

    assert saved.filename == f"{doc_id}.pdf"
    assert "resume final version" not in saved.storage_path


def test_original_filename_containing_path_traversal_characters_cannot_escape_storage(storage_service):
    doc_id = uuid4()
    payload = b"content"

    saved = storage_service.save(payload, doc_id, "../../../../../evil.pdf", "application/pdf")

    assert saved.filename == f"{doc_id}.pdf"
    assert "evil.pdf" not in saved.storage_path
    assert storage_service.exists(saved.storage_path)


def test_unsupported_content_type_is_rejected(storage_service):
    doc_id = uuid4()

    with pytest.raises(DocumentStorageTypeError):
        storage_service.save(b"not a pdf", doc_id, "resume.txt", "text/plain")


def test_get_retrieves_saved_bytes_exactly(storage_service):
    doc_id = uuid4()
    payload = b"\x00\x01\x02python\n"

    saved = storage_service.save(payload, doc_id, "resume.pdf", "application/pdf")

    assert storage_service.get(saved.storage_path) == payload


def test_get_for_nonexistent_file_raises(storage_service):
    with pytest.raises(DocumentStorageNotFoundError):
        storage_service.get("missing-file.pdf")


def test_exists_returns_true_for_existing_file(storage_service):
    doc_id = uuid4()
    payload = b"abc"

    saved = storage_service.save(payload, doc_id, "resume.pdf", "application/pdf")

    assert storage_service.exists(saved.storage_path) is True


def test_exists_returns_false_for_nonexistent_file(storage_service):
    assert storage_service.exists("missing-file.pdf") is False


def test_delete_removes_stored_file(storage_service):
    doc_id = uuid4()
    payload = b"abc"

    saved = storage_service.save(payload, doc_id, "resume.pdf", "application/pdf")
    storage_service.delete(saved.storage_path)

    assert storage_service.exists(saved.storage_path) is False


def test_delete_is_safe_for_already_missing_file(storage_service):
    storage_service.delete("already-missing.pdf")

    assert storage_service.exists("already-missing.pdf") is False


def test_get_cannot_access_file_outside_storage_root(storage_service):
    outside = storage_service.storage_root.parent / "outside.pdf"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"secret")

    with pytest.raises(DocumentStorageNotFoundError):
        storage_service.get("../outside.pdf")


def test_delete_cannot_delete_file_outside_storage_root(storage_service):
    outside = storage_service.storage_root.parent / "outside.pdf"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"secret")

    storage_service.delete("../outside.pdf")

    assert outside.exists()


def test_storage_directory_is_created_automatically(tmp_path):
    root = tmp_path / "new" / "nested" / "cv-store"
    service = DocumentStorageService(storage_root=root)

    assert root.exists()
    assert root.is_dir()
    assert service.storage_root == root.resolve()


def test_two_different_document_uuids_create_two_different_files(storage_service):
    first_id = uuid4()
    second_id = uuid4()

    first = storage_service.save(b"one", first_id, "one.pdf", "application/pdf")
    second = storage_service.save(b"two", second_id, "two.pdf", "application/pdf")

    assert first.storage_path != second.storage_path
    assert storage_service.get(first.storage_path) == b"one"
    assert storage_service.get(second.storage_path) == b"two"


def test_safe_path_rejected_for_escaping_storage_root(storage_service):
    assert storage_service.exists("../../secret.txt") is False
    assert storage_service.exists("..\\..\\secret.txt") is False
    assert storage_service.exists("/etc/passwd") is False
