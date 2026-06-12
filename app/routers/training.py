# ============================================================
# training.py - 训练接口路由
# ============================================================
"""
YOLO 模型训练 API 路由。

端点:
    POST /api/v1/training/start        - 从服务器目录启动训练任务
    POST /api/v1/training/start-upload - 上传 ZIP 并启动训练任务
    GET  /api/v1/training/status       - 查询训练状态
    POST /api/v1/training/stop         - 停止训练
    GET  /api/v1/training/list         - 列出所有训练实验
"""
import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.config import settings
from app.schemas.training import (
    TrainListData,
    TrainListItem,
    TrainListResponse,
    TrainParameters,
    TrainStartRequest,
    TrainStartResponse,
    TrainStatusData,
    TrainStatusResponse,
    TrainStopResponse,
)
from app.services.dataset_archive import (
    ArchiveLimitExceeded,
    DatasetArchiveError,
    DatasetArchiveService,
)
from app.services.trainer import get_trainer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/training")


def _create_archive_service() -> DatasetArchiveService:
    """Create an archive service from the current application settings."""
    return DatasetArchiveService(
        upload_root=Path(settings.get_upload_root()),
        max_zip_bytes=settings.SOP_MAX_DATASET_ZIP_SIZE_MB * 1024 * 1024,
        max_extract_bytes=settings.SOP_MAX_DATASET_EXTRACT_SIZE_GB * 1024**3,
        max_file_count=settings.SOP_MAX_DATASET_FILE_COUNT,
    )


def _training_config(dataset_path: str, train_id: str, params: TrainParameters) -> dict:
    return {
        "dataset_path": dataset_path,
        "trainId": train_id,
        "name": train_id,
        **params.model_dump(),
    }


def _parse_upload_parameters(
    train_id: str,
    classes: str,
    epochs: int,
    batch: int,
    imgsz: int,
    patience: int,
    lr0: float,
) -> tuple[str, TrainParameters]:
    if not train_id or not train_id.strip():
        raise HTTPException(status_code=400, detail="trainId is required")

    try:
        parsed_classes = json.loads(classes)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="classes 必须是 JSON 字符串数组",
        ) from exc

    if not isinstance(parsed_classes, list):
        raise HTTPException(status_code=400, detail="classes 必须是 JSON 数组")

    try:
        params = TrainParameters(
            classes=parsed_classes,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            patience=patience,
            lr0=lr0,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_context=False),
        ) from exc
    return train_id.strip(), params


def _response_code_for_training_status(status: str) -> int:
    return 201 if status == "failed" else 200


@router.post("/start", response_model=TrainStartResponse)
async def start_training(req: TrainStartRequest):
    """启动模型训练任务。"""
    trainer = get_trainer()
    params = TrainParameters.model_validate(req.model_dump(exclude={"dataset_path", "trainId"}))
    config = _training_config(req.dataset_path, req.trainId, params)

    try:
        result = trainer.start_training(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TrainStartResponse(
        code=_response_code_for_training_status(result["status"]),
        data=TrainStatusData(**result),
    )


@router.post("/start-upload", response_model=TrainStartResponse)
async def start_training_from_upload(
    file: UploadFile = File(..., description="YOLO 数据集 ZIP"),
    trainId: str = Form(..., description="训练任务业务主键，同时作为实验目录名"),
    classes: str = Form(..., description="JSON class_id 数组，如 [0, 1]"),
    epochs: int = Form(50),
    batch: int = Form(16),
    imgsz: int = Form(640),
    patience: int = Form(20),
    lr0: float = Form(0.01),
):
    """Upload, safely extract, validate, and start a training dataset."""
    train_id, params = _parse_upload_parameters(
        trainId,
        classes,
        epochs,
        batch,
        imgsz,
        patience,
        lr0,
    )

    try:
        dataset_path = await asyncio.to_thread(
            _create_archive_service().prepare,
            file.file,
            file.filename,
        )
        result = get_trainer().start_training(
            _training_config(str(dataset_path), train_id, params)
        )
    except ArchiveLimitExceeded as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except DatasetArchiveError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("保存或解压训练数据集失败")
        raise HTTPException(
            status_code=500,
            detail="保存或解压训练数据集失败",
        ) from exc
    finally:
        await file.close()

    return TrainStartResponse(
        code=_response_code_for_training_status(result["status"]),
        data=TrainStatusData(**result),
    )


@router.get("/status", response_model=TrainStatusResponse)
async def training_status():
    """查询当前训练任务状态。"""
    trainer = get_trainer()
    result = trainer.get_status()

    if result is None:
        raise HTTPException(status_code=404, detail="当前没有训练任务")

    return TrainStatusResponse(
        code=_response_code_for_training_status(result["status"]),
        data=TrainStatusData(**result),
    )


@router.post("/stop", response_model=TrainStopResponse)
async def stop_training():
    """停止当前训练任务。"""
    trainer = get_trainer()
    stopped = trainer.stop_training()

    if not stopped:
        raise HTTPException(status_code=404, detail="没有正在运行的训练任务")

    return TrainStopResponse(msg="训练停止请求已发送")


@router.get("/list", response_model=TrainListResponse)
async def list_experiments():
    """列出所有训练实验。"""
    trainer = get_trainer()
    experiments = trainer.list_experiments()
    try:
        from app.services.model import get_service

        current_model = settings.derive_model_name(get_service().model_path)
    except RuntimeError:
        current_model = settings.derive_model_name(settings.get_model_path())

    return TrainListResponse(
        data=TrainListData(
            experiments=[TrainListItem(**exp) for exp in experiments],
            current_model=current_model,
        )
    )
