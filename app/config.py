# ============================================================
# config.py - 配置管理
# ============================================================
"""
基于 pydantic-settings 的配置管理，从 .env 文件加载配置。
提供模型路径、训练输出目录、设备检测等工具方法。
"""
import os
from pathlib import Path

import torch
from pydantic_settings import BaseSettings

# 类名映射：默认不覆盖模型自身的类别名称。
DEFAULT_CLASS_NAMES: dict[int, str] = {}


class Settings(BaseSettings):
    """应用配置，自动从 .env 文件加载。"""

    # 服务配置
    SOP_HOST: str = "0.0.0.0"
    SOP_PORT: int = 10000

    # 推理设备: auto / cpu / cuda:0
    SOP_DEVICE: str = "auto"

    # 兼容旧配置保留；模型和训练目录不再依赖该值。
    SOP_YOLO_DIR: str = ""

    # 当前模型权重，支持绝对路径或相对项目根目录的相对路径
    SOP_MODEL_WEIGHTS: str = "yolov8s.pt"

    # 训练基础预训练权重
    SOP_BASE_WEIGHTS: str = "yolov8s.pt"

    # 默认置信度阈值
    SOP_DEFAULT_CONFIDENCE: float = 0.25

    # 上传图片大小限制 (MB)
    SOP_MAX_IMAGE_SIZE_MB: int = 10

    SOP_MAX_DATASET_ZIP_SIZE_MB: int = 4096
    SOP_MAX_DATASET_EXTRACT_SIZE_GB: int = 20
    SOP_MAX_DATASET_FILE_COUNT: int = 100000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def get_project_root(self) -> Path:
        """返回当前项目根目录。"""
        return Path(__file__).resolve().parents[1]

    def resolve_project_path(self, path_value: str) -> str:
        """将路径解析为绝对路径；相对路径相对于当前项目根目录。"""
        raw_path = Path(path_value)
        if raw_path.is_absolute():
            return str(raw_path.resolve())
        return str((self.get_project_root() / raw_path).resolve())

    def get_model_path(self) -> str:
        """获取当前模型权重的完整绝对路径。"""
        return self.resolve_project_path(self.SOP_MODEL_WEIGHTS)

    def get_base_weights_path(self) -> str:
        """获取训练基础权重的完整绝对路径。"""
        return self.resolve_project_path(self.SOP_BASE_WEIGHTS)

    def get_runs_train_dir(self) -> str:
        """获取本地训练输出根目录。"""
        return self.resolve_project_path("runs/train")

    def get_upload_root(self) -> str:
        """获取 ZIP 数据集上传解压根目录。"""
        return self.resolve_project_path("datasets/uploads")

    def get_resolved_device(self) -> str:
        """解析实际使用的推理设备，auto 时自动检测 CUDA。"""
        if self.SOP_DEVICE == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return self.SOP_DEVICE

    def get_class_names(self) -> dict:
        """获取类名映射（覆盖模型原始类名）。"""
        return dict(DEFAULT_CLASS_NAMES)

    def derive_model_name(self, weights_path: str | None) -> str:
        """从权重路径推导模型名称。"""
        if not weights_path:
            return "unknown"

        normalized = Path(weights_path)
        parts = normalized.parts
        for i, part in enumerate(parts):
            if part == "train" and i > 0 and parts[i - 1] == "runs" and i + 1 < len(parts):
                return parts[i + 1]
        return normalized.stem or "unknown"

    def get_model_name(self) -> str:
        """从当前模型权重路径推导模型名称。"""
        return self.derive_model_name(self.get_model_path())

    def get_gpu_name(self) -> str:
        """获取 GPU 名称。"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"


settings = Settings()
