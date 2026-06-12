from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import pytest

from app.services.dataset_archive import (
    ArchiveLimitExceeded,
    DatasetArchiveError,
    DatasetArchiveService,
    _copy_member,
)


def _zip_bytes(entries: list[tuple[str | zipfile.ZipInfo, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return output.getvalue()


def _valid_entries(prefix: str = "") -> list[tuple[str, bytes]]:
    return [
        (f"{prefix}images/train/example.jpg", b"image"),
        (f"{prefix}labels/train/example.txt", b"label"),
    ]


def _service(
    tmp_path: Path,
    *,
    max_zip_bytes: int = 1024 * 1024,
    max_extract_bytes: int = 1024 * 1024,
    max_file_count: int = 100,
) -> DatasetArchiveService:
    return DatasetArchiveService(
        upload_root=tmp_path / "uploads",
        max_zip_bytes=max_zip_bytes,
        max_extract_bytes=max_extract_bytes,
        max_file_count=max_file_count,
    )


def test_prepare_extracts_dataset_at_archive_root_and_removes_zip(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    dataset_root = service.prepare(
        io.BytesIO(_zip_bytes(_valid_entries())),
        "DATASET.ZIP",
    )

    assert dataset_root.name == "extracted"
    assert (dataset_root / "images/train/example.jpg").read_bytes() == b"image"
    assert (dataset_root / "labels/train/example.txt").read_bytes() == b"label"
    assert not (dataset_root.parent / "dataset.zip").exists()


def test_prepare_extracts_single_wrapped_dataset_directory(tmp_path: Path) -> None:
    service = _service(tmp_path)

    dataset_root = service.prepare(
        io.BytesIO(_zip_bytes(_valid_entries("wrapped/"))),
        "dataset.zip",
    )

    assert dataset_root.name == "wrapped"
    assert (dataset_root / "images/train/example.jpg").is_file()
    assert (dataset_root / "labels/train/example.txt").is_file()


def test_prepare_rejects_non_zip_filename(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(DatasetArchiveError):
        service.prepare(io.BytesIO(_zip_bytes(_valid_entries())), "dataset.tar")


def test_prepare_rejects_damaged_zip_and_removes_stored_archive(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    with pytest.raises(DatasetArchiveError):
        service.prepare(io.BytesIO(b"not a zip archive"), "dataset.zip")

    job_directories = list((tmp_path / "uploads").iterdir())
    assert len(job_directories) == 1
    assert not (job_directories[0] / "dataset.zip").exists()


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.txt",
        "/absolute.txt",
        "C:/windows.txt",
        r"\\server\share\file.txt",
    ],
)
def test_prepare_rejects_unsafe_member_paths(
    tmp_path: Path,
    member_name: str,
) -> None:
    service = _service(tmp_path)
    entries = [*_valid_entries(), (member_name, b"unsafe")]

    with pytest.raises(DatasetArchiveError):
        service.prepare(io.BytesIO(_zip_bytes(entries)), "dataset.zip")


def test_prepare_rejects_unix_symbolic_link(tmp_path: Path) -> None:
    service = _service(tmp_path)
    symlink = zipfile.ZipInfo("linked-file")
    symlink.create_system = 3
    symlink.external_attr = stat.S_IFLNK << 16
    entries: list[tuple[str | zipfile.ZipInfo, bytes]] = [
        *_valid_entries(),
        (symlink, b"target"),
    ]

    with pytest.raises(DatasetArchiveError):
        service.prepare(io.BytesIO(_zip_bytes(entries)), "dataset.zip")


def test_prepare_rejects_member_count_over_limit(tmp_path: Path) -> None:
    service = _service(tmp_path, max_file_count=1)

    with pytest.raises(ArchiveLimitExceeded):
        service.prepare(
            io.BytesIO(_zip_bytes(_valid_entries())),
            "dataset.zip",
        )


def test_prepare_rejects_declared_extract_size_over_limit(tmp_path: Path) -> None:
    service = _service(tmp_path, max_extract_bytes=9)

    with pytest.raises(ArchiveLimitExceeded):
        service.prepare(
            io.BytesIO(_zip_bytes(_valid_entries())),
            "dataset.zip",
        )


def test_prepare_rejects_upload_size_over_limit(tmp_path: Path) -> None:
    service = _service(tmp_path, max_zip_bytes=10)

    with pytest.raises(ArchiveLimitExceeded):
        service.prepare(
            io.BytesIO(_zip_bytes(_valid_entries())),
            "dataset.zip",
        )


def test_copy_member_enforces_actual_written_byte_limit() -> None:
    destination = io.BytesIO()

    with pytest.raises(ArchiveLimitExceeded):
        _copy_member(io.BytesIO(b"123456"), destination, max_bytes=5)

    assert len(destination.getvalue()) <= 5


def test_prepare_rejects_archive_without_dataset_root(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(DatasetArchiveError):
        service.prepare(
            io.BytesIO(_zip_bytes([("notes/readme.txt", b"hello")])),
            "dataset.zip",
        )


def test_prepare_rejects_multiple_direct_child_dataset_roots(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    entries = [*_valid_entries("first/"), *_valid_entries("second/")]

    with pytest.raises(DatasetArchiveError):
        service.prepare(io.BytesIO(_zip_bytes(entries)), "dataset.zip")


def test_prepare_rejects_file_directory_path_conflict(tmp_path: Path) -> None:
    service = _service(tmp_path)
    entries = [
        ("images", b"file blocks directory"),
        ("images/train/example.jpg", b"image"),
        ("labels/train/example.txt", b"label"),
    ]

    with pytest.raises(DatasetArchiveError):
        service.prepare(io.BytesIO(_zip_bytes(entries)), "dataset.zip")
