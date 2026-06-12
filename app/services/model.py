# ============================================================
# model.py - YOLO 模型加载与推理
# ============================================================
"""
单例模式的 YOLO 推理服务，启动时加载模型，全局复用。
"""
import logging
import os
import time

import numpy as np
from ultralytics import YOLO

from app.config import settings

logger = logging.getLogger(__name__)


class YOLOService:
    """YOLO 推理服务，启动时加载模型，全局复用。"""

    def __init__(self):
        model_path = settings.get_model_path()
        if not os.path.isfile(model_path):
            fallback_path = settings.get_base_weights_path()
            logger.warning(
                "当前模型权重不存在，回退到基础权重: %s -> %s",
                model_path,
                fallback_path,
            )
            model_path = fallback_path

        device = settings.get_resolved_device()
        logger.info("正在加载 YOLO 模型: %s", model_path)
        logger.info("推理设备: %s", device)

        self.model = YOLO(model_path)
        self.device = device
        self.model_path = model_path
        self.class_names_override = settings.get_class_names()

        logger.info("模型加载完成，原始类别: %s", self.model.names)

    def predict(self, image: np.ndarray, conf: float = 0.25) -> tuple:
        """
        对单张图片运行推理。
        Returns:
            (detections, elapsed_seconds)
        """
        t0 = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf,
            device=self.device,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0

        detections = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names_override.get(
                    cls_id, results[0].names.get(cls_id, f"class_{cls_id}")
                )
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(confidence, 4),
                        "bbox": {
                            "x1": round(x1, 1),
                            "y1": round(y1, 1),
                            "x2": round(x2, 1),
                            "y2": round(y2, 1),
                        },
                    }
                )
        return detections, elapsed

    def reload_model(self, new_weights_path: str, class_names: dict | None = None):
        """热重载模型。"""
        if not os.path.isfile(new_weights_path):
            raise FileNotFoundError(f"权重文件不存在: {new_weights_path}")

        logger.info("正在切换模型: %s -> %s", self.model_path, new_weights_path)
        self.model = YOLO(new_weights_path)
        self.model_path = new_weights_path
        self.class_names_override = class_names or {}
        logger.info("模型切换完成，原始类别: %s", self.model.names)

    @property
    def native_names(self) -> dict:
        return dict(self.model.names)

    @property
    def device_name(self) -> str:
        return self.device


_service: YOLOService | None = None


def init_service() -> YOLOService:
    """初始化 YOLO 服务。"""
    global _service
    _service = YOLOService()
    return _service


def get_service() -> YOLOService:
    """获取 YOLO 服务实例。"""
    if _service is None:
        raise RuntimeError("YOLOService 未初始化，请先调用 init_service()")
    return _service
