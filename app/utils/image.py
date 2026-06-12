# ============================================================
# image.py — 图片验证、编解码工具
# ============================================================
"""
全内存操作，不写磁盘，避免 Windows 中文路径问题。
使用 cv2.imdecode / cv2.imencode 在内存中处理图片。
"""
import os

import cv2
import numpy as np

# 支持的图片格式
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# 图片尺寸上限（防止异常大图耗尽内存）
MAX_DIMENSION = 4096


def validate_upload(filename: str, content: bytes, max_size_mb: int) -> None:
    """
    校验上传的图片文件。

    Args:
        filename: 文件名（用于检查扩展名）
        content: 文件二进制内容
        max_size_mb: 文件大小上限 (MB)

    Raises:
        ValueError: 文件格式不支持或大小超限
    """
    if not filename:
        raise ValueError("缺少文件名")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"不支持的图片格式: {ext}，支持的格式: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    max_bytes = max_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise ValueError(
            f"图片大小 {len(content) / 1024 / 1024:.1f}MB 超过限制 {max_size_mb}MB"
        )


def decode_image(content: bytes) -> np.ndarray:
    """
    将图片二进制数据解码为 BGR numpy 数组。

    Args:
        content: 图片文件的二进制内容

    Returns:
        BGR 格式的 numpy 数组

    Raises:
        ValueError: 图片解码失败或尺寸异常
    """
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图片解码失败，文件可能已损坏")

    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        raise ValueError(f"图片尺寸 {w}x{h} 超过最大限制 {MAX_DIMENSION}")

    return img


def encode_image(img: np.ndarray, ext: str = ".jpg", quality: int = 90) -> bytes:
    """
    将 BGR numpy 数组编码为图片二进制数据。

    Args:
        img: BGR 格式的 numpy 数组
        ext: 目标格式（.jpg / .png）
        quality: JPEG 编码质量（1-100）

    Returns:
        编码后的二进制数据

    Raises:
        ValueError: 编码失败
    """
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    success, buf = cv2.imencode(ext, img, params)
    if not success:
        raise ValueError("图片编码失败")

    return buf.tobytes()
