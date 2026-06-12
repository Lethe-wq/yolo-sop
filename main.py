# ============================================================
# main.py — FastAPI 应用入口
# ============================================================
"""
SOP 工件检测服务，启动时加载 YOLO 模型。

启动方式:
    cd yolo-api
    python main.py

    # 或使用 uvicorn（支持热重载）
    uvicorn main:app --reload --host 127.0.0.1 --port 10000
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import detection
from app.routers import training
from app.services.model import init_service, get_service

# ---- 日志配置 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型，关闭时清理资源"""
    # Startup
    logger.info("正在初始化 YOLO 服务...")
    service = init_service()
    logger.info(f"模型加载完成，设备: {service.device_name}")
    logger.info(f"服务地址: http://{settings.SOP_HOST}:{settings.SOP_PORT}")
    logger.info(f"API 文档: http://{settings.SOP_HOST}:{settings.SOP_PORT}/docs")
    yield
    # Shutdown（模型随进程退出自动释放，无需额外清理）
    logger.info("服务关闭")


# ---- FastAPI 应用 ----
app = FastAPI(
    title="SOP 工件检测服务",
    description="基于 YOLOv8 的 SOP 工件检测 FastAPI 服务",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS：允许 Electron 前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(detection.router)
app.include_router(training.router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.SOP_HOST,
        port=settings.SOP_PORT,
        reload=False,  # 模型加载较慢，默认不热重载
    )
