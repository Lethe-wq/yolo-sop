from __future__ import annotations

import os
import re
import stat
import uuid
import zipfile
from pathlib import Path
from typing import BinaryIO


# 每次读写的块大小（1MB）
_CHUNK_SIZE = 1024 * 1024
# 匹配 Windows 驱动器前缀，例如 C: 或 D:
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class DatasetArchiveError(ValueError):
    """数据集归档相关的基类异常。

    当上传的 ZIP 文件存在安全问题或格式错误时抛出此异常。
    """


class ArchiveLimitExceeded(DatasetArchiveError):
    """当归档或解压超出配置的资源限制时抛出此异常。"""


def _copy_member(source: BinaryIO, destination: BinaryIO, max_bytes: int) -> int:
    """安全地从归档内流式拷贝单个成员到目标文件。

    该函数会逐块读取源流并写入目标，确保写入总字节数不超过
    `max_bytes`，否则抛出 `ArchiveLimitExceeded`。
    返回实际写入的字节数。
    """
    written = 0
    while chunk := source.read(_CHUNK_SIZE):
        if written + len(chunk) > max_bytes:
            raise ArchiveLimitExceeded("Extracted data exceeds the configured limit")
        destination.write(chunk)
        written += len(chunk)
    return written


class DatasetArchiveService:
    """处理训练数据集 ZIP 上传与解压的服务类（含安全校验）。

    主要职责：
    - 保存上传的 ZIP 到磁盘（受 `max_zip_bytes` 限制）
    - 验证 ZIP 成员是否合法（禁止符号链接、路径穿越、文件/目录冲突）
    - 按块解压成员且受 `max_extract_bytes` 与 `max_file_count` 限制
    - 返回解压后识别出的训练数据集根目录（必须且只能有一个）
    """

    def __init__(
        self,
        upload_root: Path,
        max_zip_bytes: int,
        max_extract_bytes: int,
        max_file_count: int,
    ) -> None:
        # 上传临时目录
        self.upload_root = upload_root
        # 上传 ZIP 的最大字节数限制
        self.max_zip_bytes = max_zip_bytes
        # 解压后总字节数限制
        self.max_extract_bytes = max_extract_bytes
        # ZIP 成员文件数限制（包含目录）
        self.max_file_count = max_file_count

    def prepare(self, source: BinaryIO, filename: str | None) -> Path:
        """验证并准备 ZIP 上传：保存、解压并返回数据集根目录路径。

        参数：
        - `source`: 上传的二进制流
        - `filename`: 客户端提供的文件名（用于校验后缀）
        返回值：解压后的训练数据集根目录（Path）
        """
        if not filename or Path(filename).suffix.lower() != ".zip":
            raise DatasetArchiveError("Only .zip dataset archives are accepted")

        # 创建上传根目录并为本次任务创建唯一子目录
        self.upload_root.mkdir(parents=True, exist_ok=True)
        job_root = self.upload_root / str(uuid.uuid4())
        extracted_root = job_root / "extracted"
        archive_path = job_root / "dataset.zip"
        extracted_root.mkdir(parents=True)

        try:
            # 先保存上传的 ZIP，再解压并校验内容，最后返回数据集根
            self._store_upload(source, archive_path)
            self._extract_archive(archive_path, extracted_root)
            return self._find_dataset_root(extracted_root)
        finally:
            # 无论成功或失败，尝试删除临时的 ZIP 文件（不删除解压目录）
            archive_path.unlink(missing_ok=True)

    def _store_upload(self, source: BinaryIO, archive_path: Path) -> None:
        """将上传流按块写入磁盘，同时检查 `max_zip_bytes` 限制。"""
        stored = 0
        try:
            with archive_path.open("wb") as destination:
                while chunk := source.read(_CHUNK_SIZE):
                    if stored + len(chunk) > self.max_zip_bytes:
                        raise ArchiveLimitExceeded(
                            "ZIP upload exceeds the configured limit"
                        )
                    destination.write(chunk)
                    stored += len(chunk)
        except ArchiveLimitExceeded:
            # 资源限制触发：直接向上抛出，调用方负责捕获并返回 400/413 等
            raise
        except (OSError, ValueError) as exc:
            # 磁盘写入失败或输入流异常
            raise DatasetArchiveError("Unable to store ZIP upload") from exc

    def _extract_archive(
        self,
        archive_path: Path,
        extracted_root: Path,
    ) -> None:
        """解压 ZIP 文件并将成员拷贝到 `extracted_root`，包含安全校验与资源限制。

        解压时会先调用 `_validate_members` 做全量校验（成员数量/声明大小/路径安全性），
        然后逐个成员按块写入到磁盘，累计写入字节受 `max_extract_bytes` 限制。
        """
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                validated = self._validate_members(members, extracted_root)
                extracted_bytes = 0

                for member, target, is_directory in validated:
                    if is_directory:
                        target.mkdir(parents=True, exist_ok=True)
                        continue

                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as member_source, target.open(
                        "wb"
                    ) as destination:
                        extracted_bytes += _copy_member(
                            member_source,
                            destination,
                            self.max_extract_bytes - extracted_bytes,
                        )
        except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError) as exc:
            # ZIP 文件本身有问题
            raise DatasetArchiveError("Invalid ZIP archive") from exc
        except ArchiveLimitExceeded:
            # 超出限制，直接向上抛出以便调用方处理
            raise
        except OSError as exc:
            # 文件系统写入错误
            raise DatasetArchiveError("Unable to extract ZIP archive") from exc

    def _validate_members(
        self,
        members: list[zipfile.ZipInfo],
        extracted_root: Path,
    ) -> list[tuple[zipfile.ZipInfo, Path, bool]]:
        """对 ZIP 的所有成员执行静态验证，返回 (member, target_path, is_dir) 列表。

        验证包括：
        - 成员数量不超过 `max_file_count`
        - 声明的总解压大小不超过 `max_extract_bytes`
        - 禁止符号链接
        - 成员路径不会越出解压根目录（防止路径穿越）
        - 文件/目录的冲突检测
        """
        if len(members) > self.max_file_count:
            raise ArchiveLimitExceeded(
                "ZIP member count exceeds the configured limit"
            )

        declared_size = sum(
            member.file_size for member in members if not member.is_dir()
        )
        if declared_size > self.max_extract_bytes:
            raise ArchiveLimitExceeded(
                "ZIP extracted size exceeds the configured limit"
            )

        root = extracted_root.resolve()
        path_kinds: dict[tuple[str, ...], str] = {}
        validated: list[tuple[zipfile.ZipInfo, Path, bool]] = []

        for member in members:
            # 禁止 ZIP 内包含符号链接（可能指向宿主文件系统）
            if self._is_symlink(member):
                raise DatasetArchiveError("ZIP symbolic links are not allowed")

            parts = self._safe_parts(member.filename)
            target = extracted_root.joinpath(*parts)
            # 确保解析后的目标路径位于解压根目录下
            if not target.resolve(strict=False).is_relative_to(root):
                raise DatasetArchiveError("ZIP member escapes extraction root")

            is_directory = member.is_dir()
            key_parts = tuple(os.path.normcase(part) for part in parts)
            # 记录路径种类并检测文件/目录冲突
            self._record_path(path_kinds, key_parts, is_directory)
            validated.append((member, target, is_directory))

        return validated

    @staticmethod
    def _safe_parts(member_name: str) -> tuple[str, ...]:
        """将 ZIP 内部的文件名切分为路径部分并做安全检查。

        检查包括：空名、空字节、绝对路径/以斜杠开头、Windows 驱动器前缀、以及 '.'/'..' 等。
        返回路径部分的元组（适合与 `Path.joinpath` 一起使用）。
        """
        normalized = member_name.replace("\\", "/")
        if (
            not normalized
            or "\x00" in normalized
            or normalized.startswith("/")
            or _WINDOWS_DRIVE.match(normalized)
        ):
            raise DatasetArchiveError("ZIP contains an unsafe member path")

        normalized = normalized.rstrip("/")
        parts = tuple(normalized.split("/"))
        if not normalized or any(part in {"", ".", ".."} for part in parts):
            raise DatasetArchiveError("ZIP contains an unsafe member path")
        return parts

    @staticmethod
    def _is_symlink(member: zipfile.ZipInfo) -> bool:
        """判断 ZIPInfo 是否表示一个 unix 符号链接。"""
        unix_mode = member.external_attr >> 16
        return member.create_system == 3 and stat.S_ISLNK(unix_mode)

    @staticmethod
    def _record_path(
        path_kinds: dict[tuple[str, ...], str],
        parts: tuple[str, ...],
        is_directory: bool,
    ) -> None:
        """记录路径层级信息并检测文件/目录冲突。

        例如，若存在 'a/b' 为文件但之后出现 'a/b/c'，则为冲突。
        """
        for index in range(1, len(parts)):
            parent = parts[:index]
            if path_kinds.get(parent) == "file":
                raise DatasetArchiveError("ZIP contains a file/directory conflict")
            path_kinds.setdefault(parent, "directory")

        current_kind = "directory" if is_directory else "file"
        existing_kind = path_kinds.get(parts)
        if existing_kind is not None and (
            existing_kind != current_kind or current_kind == "file"
        ):
            raise DatasetArchiveError("ZIP contains a file/directory conflict")
        path_kinds[parts] = current_kind

    @staticmethod
    def _find_dataset_root(extracted_root: Path) -> Path:
        """在解压目录中寻找训练数据集的根目录。

        规定的数据集根目录必须包含 `images/train` 和 `labels/train` 两个子目录。
        如果满足条件的根目录数量不为 1，则认为归档结构不符合要求并抛出异常。
        """
        candidates: list[Path] = []
        roots = [extracted_root]
        roots.extend(path for path in extracted_root.iterdir() if path.is_dir())

        for root in roots:
            if (root / "images" / "train").is_dir() and (
                root / "labels" / "train"
            ).is_dir():
                candidates.append(root)

        if len(candidates) != 1:
            raise DatasetArchiveError(
                "Archive must contain exactly one training dataset root"
            )
        return candidates[0]
