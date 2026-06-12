# ============================================================
# annotator.py — 标注绘制
# ============================================================
"""
在检测图片上绘制标注框和合格/不合格结果。
复用 yolo/predict_dual.py 的 PIL 中文字体渲染模式。

绘制内容:
1. 每个检测目标画框 + 类别标签
2. 图片顶部增加结果横幅（检测数量 / 期望数量 → 合格/不合格）
"""
import logging
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---- 颜色方案（BGR for OpenCV, RGB for PIL）----
COLOR_PASS = (0, 200, 0)       # 合格 - 绿色
COLOR_FAIL = (0, 0, 220)       # 不合格 - 红色
BOX_THICKNESS = 2
FONT_SIZE = 24

# ---- 中文字体路径（Windows 系统字体）----
_FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",      # 黑体
    "C:/Windows/Fonts/simsun.ttc",      # 宋体
]
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont:
    """获取支持中文的 PIL 字体，自动查找 Windows 系统字体"""
    if size in _font_cache:
        return _font_cache[size]

    for fp in _FONT_PATHS:
        if os.path.isfile(fp):
            _font_cache[size] = ImageFont.truetype(fp, size)
            return _font_cache[size]

    logger.warning("未找到中文字体，使用默认字体（不支持中文）")
    _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def draw_results(
    image: np.ndarray,
    detections: list,
    passed: bool,
    expected_count: int,
    actual_count: int,
    product_type: str = "",
) -> np.ndarray:
    """
    在图片上绘制检测框和结果信息。

    Args:
        image: BGR 格式 numpy 数组（原图）
        detections: 检测结果列表
        passed: 是否合格
        expected_count: 期望数量
        actual_count: 实际数量
        product_type: 工件类型名称

    Returns:
        标注后的 BGR numpy 数组（新数组，不修改原图）
    """
    color = COLOR_PASS if passed else COLOR_FAIL
    color_rgb = (color[2], color[1], color[0])  # BGR → RGB

    # OpenCV BGR → PIL RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    font = _get_font(FONT_SIZE)

    # ---- 1. 绘制每个检测目标的框 + 标签 ----
    draw = ImageDraw.Draw(pil_img)
    for det in detections:
        bbox = det["bbox"]
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        label = f"{det['class_name']} {det['confidence']:.1%}"

        # 画框
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=BOX_THICKNESS)

        # 计算文字尺寸并画标签背景
        text_bbox = font.getbbox(label)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - th - 8, x1 + tw + 8, y1], fill=color_rgb)
        draw.text((x1 + 4, y1 - th - 6), label, fill=(255, 255, 255), font=font)

    # ---- 2. 绘制顶部结果横幅 ----
    status_text = "合格" if passed else "不合格"
    if product_type:
        header = f"[{product_type}] 检测: {actual_count} / 期望: {expected_count} -> {status_text}"
    else:
        header = f"检测: {actual_count} / 期望: {expected_count} -> {status_text}"

    header_bbox = font.getbbox(header)
    hw = header_bbox[2] - header_bbox[0]
    hh = header_bbox[3] - header_bbox[1]

    # 创建新画布（比原图高一截，放横幅）
    img_w = image.shape[1]
    banner_h = hh + 16
    new_h = image.shape[0] + banner_h + 4
    new_pil = Image.new("RGB", (img_w, new_h), (40, 40, 40))
    new_pil.paste(pil_img, (0, banner_h + 4))

    # 在横幅区域画背景色 + 文字
    draw_new = ImageDraw.Draw(new_pil)
    draw_new.rectangle([0, 0, img_w, banner_h], fill=color_rgb)
    draw_new.text((8, 4), header, fill=(255, 255, 255), font=font)

    # PIL RGB → OpenCV BGR
    result = cv2.cvtColor(np.array(new_pil), cv2.COLOR_RGB2BGR)
    return result
