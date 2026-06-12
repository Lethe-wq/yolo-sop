# ============================================================
# detection.py - 检测接口路由
# ============================================================
"""
SOP 工件检测 API 路由。
端点:
    POST /api/v1/detection/detect    - 检测接口（返回 JSON）
    POST /api/v1/detection/annotated - 标注图片接口（返回图片二进制流）
    GET  /api/v1/health              - 健康检查
    GET  /api/v1/model               - 模型信息
    PUT  /api/v1/model               - 切换模型
"""
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.schemas.detection import (
    ApiResponse,
    Detection,
    DetectionData,
    HealthData,
    HealthResponse,
    ModelInfoData,
    ModelInfoResponse,
    ModelSwitchRequest,
    ModelSwitchResponse,
)
from app.services.annotator import draw_results
from app.services.model import get_service
from app.utils.image import decode_image, encode_image, validate_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

# 线程池：单线程，防止 GPU 推理并发冲突（本地单用户场景）
_executor = ThreadPoolExecutor(max_workers=1)


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _resolve_requested_model_path(model_path: str | None) -> str:
    if model_path is None or not model_path.strip():
        raise HTTPException(status_code=400, detail="model_path is required")
    return settings.resolve_project_path(model_path.strip())


def _ensure_requested_model(service, model_path: str | None) -> str:
    resolved_path = _resolve_requested_model_path(model_path)
    current_path = getattr(service, "model_path", None)
    if current_path and _normalize_path(current_path) == _normalize_path(resolved_path):
        return resolved_path

    if not os.path.isfile(resolved_path):
        raise HTTPException(status_code=404, detail=f"模型权重不存在: {resolved_path}")

    try:
        service.reload_model(resolved_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"模型切换失败: {exc}") from exc

    return resolved_path


@router.post("/detection/detect", response_model=ApiResponse)
async def detect(
    file: UploadFile = File(..., description="工件图片 (JPG/PNG/BMP, <= 10MB)"),
    product_type: str = Form(..., description="工件类型名称"),
    expected_count: int = Form(..., gt=0, description="期望数量"),
    model_path: str | None = Form(default=None, description="本次推理使用的模型权重路径"),
    confidence: float = Form(default=None, gt=0, lt=1, description="置信度阈值（默认 0.25）"),
):
    """工件检测接口，返回 JSON 格式检测结果。"""
    _resolve_requested_model_path(model_path)
    service = get_service()
    _ensure_requested_model(service, model_path)

    conf = confidence if confidence is not None else settings.SOP_DEFAULT_CONFIDENCE

    content = await file.read()
    try:
        validate_upload(file.filename, content, settings.SOP_MAX_IMAGE_SIZE_MB)
        image = decode_image(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    loop = asyncio.get_event_loop()
    detections, elapsed = await loop.run_in_executor(
        _executor, service.predict, image, conf
    )

    actual_count = len(detections)
    passed = actual_count == expected_count
    avg_conf = (
        sum(d["confidence"] for d in detections) / len(detections)
        if detections
        else 0.0
    )

    return ApiResponse(
        data=DetectionData(
            product_type=product_type,
            expected_count=expected_count,
            actual_count=actual_count,
            result="pass" if passed else "fail",
            confidence=round(avg_conf, 4),
            inference_time_ms=round(elapsed * 1000, 1),
            detections=[Detection(**d) for d in detections],
        )
    )


@router.post("/detection/annotated")
async def annotated(
    file: UploadFile = File(..., description="工件图片 (JPG/PNG/BMP, <= 10MB)"),
    product_type: str = Form(..., description="工件类型名称"),
    expected_count: int = Form(..., gt=0, description="期望数量"),
    model_path: str | None = Form(default=None, description="本次推理使用的模型权重路径"),
    confidence: float = Form(default=None, gt=0, lt=1, description="置信度阈值"),
):
    """标注图片接口，返回检测并绘制后的 JPEG 图片。"""
    _resolve_requested_model_path(model_path)
    service = get_service()
    _ensure_requested_model(service, model_path)

    conf = confidence if confidence is not None else settings.SOP_DEFAULT_CONFIDENCE

    content = await file.read()
    try:
        validate_upload(file.filename, content, settings.SOP_MAX_IMAGE_SIZE_MB)
        image = decode_image(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    loop = asyncio.get_event_loop()
    detections, _elapsed = await loop.run_in_executor(
        _executor, service.predict, image, conf
    )

    actual_count = len(detections)
    passed = actual_count == expected_count
    annotated_img = draw_results(
        image, detections, passed, expected_count, actual_count, product_type
    )

    img_bytes = encode_image(annotated_img)
    return Response(content=img_bytes, media_type="image/jpeg")


@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查接口，返回服务状态和模型信息。"""
    service = get_service()
    return HealthResponse(
        data=HealthData(
            status="healthy",
            model_loaded=True,
            model_name=settings.derive_model_name(getattr(service, "model_path", None)),
            device=service.device_name,
            gpu_name=settings.get_gpu_name(),
        )
    )


@router.get("/model", response_model=ModelInfoResponse)
async def model_info():
    """获取当前模型信息。"""
    service = get_service()
    native = service.native_names
    override = service.class_names_override
    merged = {str(k): override.get(k, v) for k, v in native.items()}

    actual_path = service.model_path
    return ModelInfoResponse(
        data=ModelInfoData(
            name=settings.derive_model_name(actual_path),
            classes=merged,
            device=service.device_name,
            weights_path=actual_path or settings.get_model_path(),
        )
    )


@router.put("/model", response_model=ModelSwitchResponse)
async def switch_model(req: ModelSwitchRequest):
    """切换到指定模型权重路径。"""
    service = get_service()
    weights_path = _ensure_requested_model(service, req.model_path)
    model_name = settings.derive_model_name(weights_path)

    return ModelSwitchResponse(
        msg=f"已切换到模型: {model_name}",
        data={
            "new_model_name": model_name,
            "new_weights_path": weights_path,
        },
    )
