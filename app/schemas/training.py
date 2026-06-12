# ============================================================
# training.py - 训练接口数据模型
# ============================================================
"""
Pydantic v2 模型，用于训练相关的请求和响应。
"""
from typing import Optional

from pydantic import BaseModel, Field, StrictInt, field_validator


def _validate_class_ids(value: list[int]) -> list[int]:
    if len(value) == 0:
        raise ValueError("classes must contain at least one class id")
    if any(class_id < 0 for class_id in value):
        raise ValueError("classes must contain non-negative integers")
    if len(set(value)) != len(value):
        raise ValueError("classes must not contain duplicate class ids")
    return value


class TrainParameters(BaseModel):
    """训练参数。"""

    classes: list[StrictInt] = Field(
        ...,
        min_length=1,
        description="训练数据中使用的 class_id 数组（array），不要求连续编号，如 [0]、[0, 2, 5]",
    )
    epochs: int = Field(50, ge=1, le=1000, description="训练轮数（默认 50）")
    batch: int = Field(16, ge=1, le=128, description="批次大小（默认 16）")
    imgsz: int = Field(640, ge=128, le=1280, description="输入图像尺寸（默认 640）")
    patience: int = Field(20, ge=0, le=200, description="早停耐心值（默认 20）")
    lr0: float = Field(0.01, gt=0, le=1.0, description="初始学习率（默认 0.01）")

    @field_validator("classes", mode="after")
    @classmethod
    def validate_classes(cls, value: list[int]) -> list[int]:
        return _validate_class_ids(value)


class TrainStartRequest(TrainParameters):
    """训练启动请求。"""

    model_config = {
        "json_schema_extra": {
            "example": {
                "classes": [0, 2],
                "trainId": "train-20260610-001",
                "dataset_path": "E:/datasets/screws_v2",
            }
        }
    }

    trainId: str = Field(..., min_length=1, description="训练任务业务主键，同时作为实验目录名")
    dataset_path: str = Field(
        ...,
        min_length=1,
        description="数据集目录绝对路径，需包含 images/train 和 labels/train",
    )


class TrainProgress(BaseModel):
    """训练进度信息。"""

    current_epoch: int = Field(0, description="当前轮次")
    total_epochs: int = Field(0, description="总轮次")
    box_loss: Optional[float] = Field(None, description="边界框损失")
    cls_loss: Optional[float] = Field(None, description="分类损失")
    dfl_loss: Optional[float] = Field(None, description="分布焦点损失")
    map50: Optional[float] = Field(None, description="mAP@0.5")
    map50_95: Optional[float] = Field(None, description="mAP@0.5:0.95")
    progress_pct: float = Field(0.0, description="进度百分比")


class TrainStatusData(BaseModel):
    """训练状态数据。"""

    task_id: str = Field(..., description="运行态任务 ID")
    trainId: str = Field(..., description="训练任务业务主键")
    status: str = Field(..., description="状态: pending / running / completed / failed")
    name: str = Field(..., description="实验目录名，固定等于 trainId")
    config: dict = Field(default_factory=dict, description="训练配置快照")
    progress: TrainProgress = Field(default_factory=TrainProgress, description="训练进度")
    best_weights_path: Optional[str] = Field(None, description="最佳权重路径")
    error: Optional[str] = Field(None, description="失败时的错误信息")
    started_at: Optional[str] = Field(None, description="开始时间")
    finished_at: Optional[str] = Field(None, description="结束时间")


class TrainStartResponse(BaseModel):
    """训练启动响应。"""

    code: int = 200
    msg: str = "success"
    data: TrainStatusData


class TrainStatusResponse(BaseModel):
    """训练状态响应。"""

    code: int = 200
    msg: str = "success"
    data: TrainStatusData


class TrainListItem(BaseModel):
    """单个训练实验。"""

    trainId: str = Field(..., description="训练任务业务主键")
    name: str = Field(..., description="实验目录名")
    status: str = Field(..., description="状态: completed / failed / incomplete")
    best_weights: Optional[str] = Field(None, description="最佳权重路径")
    created: Optional[str] = Field(None, description="创建时间")


class TrainListData(BaseModel):
    """训练实验列表数据。"""

    experiments: list[TrainListItem] = Field(default_factory=list, description="实验列表")
    current_model: str = Field(..., description="当前使用的模型名称")


class TrainListResponse(BaseModel):
    """训练实验列表响应。"""

    code: int = 200
    msg: str = "success"
    data: TrainListData


class TrainStopResponse(BaseModel):
    """停止训练响应。"""

    code: int = 200
    msg: str = "success"
    data: Optional[dict] = None
