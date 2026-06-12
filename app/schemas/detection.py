# ============================================================
# detection.py - 请求/响应数据模型
# ============================================================
"""
Pydantic v2 模型，用于响应序列化和 Swagger 文档生成。
检测接口的请求参数通过 Form() 接收（multipart/form-data），
不使用 Pydantic request body。
"""
from typing import Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """检测框坐标（左上角 + 右下角）。"""

    x1: float = Field(..., description="左上角 X")
    y1: float = Field(..., description="左上角 Y")
    x2: float = Field(..., description="右下角 X")
    y2: float = Field(..., description="右下角 Y")


class Detection(BaseModel):
    """单个检测结果。"""

    class_id: int = Field(..., description="类别 ID")
    class_name: str = Field(..., description="类别名称")
    confidence: float = Field(..., description="置信度")
    bbox: BBox = Field(..., description="检测框坐标")


class DetectionData(BaseModel):
    """检测接口返回的业务数据。"""

    product_type: str = Field(..., description="工件类型")
    expected_count: int = Field(..., description="期望数量")
    actual_count: int = Field(..., description="实际检测数量")
    result: str = Field(..., description="检测结果 pass 或 fail", pattern="^(pass|fail)$")
    confidence: float = Field(..., description="所有检测目标的平均置信度")
    inference_time_ms: float = Field(..., description="推理耗时（毫秒）")
    detections: list[Detection] = Field(..., description="检测目标列表")


class ApiResponse(BaseModel):
    """标准 API 响应格式。"""

    code: int = 200
    msg: str = "success"
    data: Optional[DetectionData] = None


class HealthData(BaseModel):
    """健康检查数据。"""

    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    model_name: str = Field(..., description="模型名称")
    device: str = Field(..., description="推理设备")
    gpu_name: str = Field(..., description="GPU 名称")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    code: int = 200
    msg: str = "success"
    data: HealthData


class ModelInfoData(BaseModel):
    """模型信息数据。"""

    name: str = Field(..., description="模型名称")
    classes: dict[str, str] = Field(..., description="类别映射 {id: name}")
    device: str = Field(..., description="推理设备")
    weights_path: str = Field(..., description="模型权重路径")


class ModelInfoResponse(BaseModel):
    """模型信息响应。"""

    code: int = 200
    msg: str = "success"
    data: ModelInfoData


class ModelSwitchRequest(BaseModel):
    """模型切换请求。"""

    model_path: str = Field(..., description="要切换到的模型权重路径")


class ModelSwitchResponse(BaseModel):
    """模型切换响应。"""

    code: int = 200
    msg: str
    data: Optional[dict] = None
