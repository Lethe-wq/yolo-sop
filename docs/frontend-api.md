# SOP 工件检测服务前端接口文档

本文档给前端同事直接对接使用，基于当前 FastAPI 服务的真实接口整理。

## 1. 基础信息

- 本地调试地址: `http://127.0.0.1:10000`
- 局域网访问: 启动服务时需监听 `0.0.0.0`，然后使用机器内网 IP 访问，例如 `http://192.168.0.122:10000`
- Swagger 文档: `http://127.0.0.1:10000/docs`
- 统一前缀: `/api/v1`

## 2. 通用约定

### 2.1 响应格式

大部分接口返回统一结构：

```json
{
  "code": 200,
  "msg": "success",
  "data": {}
}
```

### 2.2 常见状态码

- `200` 成功
- `400` 参数错误、文件格式错误、数据集格式错误
- `404` 资源不存在、当前无训练任务
- `409` 状态冲突，例如已有训练任务正在运行
- `413` 上传文件或解压数据超过限制
- `500` 服务器内部错误

### 2.3 连接测试接口

用于确认前端是否能连到后端：

- `GET /test`

返回纯文本：`ok`

## 3. 接口列表

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/test` | 连通性测试，返回 `ok` |
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/model` | 获取当前模型信息 |
| PUT | `/api/v1/model` | 切换模型 |
| POST | `/api/v1/detection/detect` | 图片检测，返回 JSON 结果 |
| POST | `/api/v1/detection/annotated` | 图片检测并返回标注图 |
| POST | `/api/v1/training/start` | 从本地数据集目录启动训练 |
| POST | `/api/v1/training/start-upload` | 上传 ZIP 并启动训练 |
| GET | `/api/v1/training/status` | 查询当前训练状态 |
| POST | `/api/v1/training/stop` | 停止当前训练 |
| GET | `/api/v1/training/list` | 列出训练实验 |

## 4. 接口说明

### 4.1 健康检查

**GET** `/api/v1/health`

返回示例：

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "status": "healthy",
    "model_loaded": true,
    "model_name": "exp-2",
    "device": "cuda:0",
    "gpu_name": "NVIDIA GeForce RTX 4060"
  }
}
```

### 4.2 模型信息

**GET** `/api/v1/model`

返回示例：

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "name": "exp-2",
    "classes": {
      "0": "螺丝"
    },
    "device": "cuda:0",
    "weights_path": "E:/shixi WORK/SOP/yolo/runs/train/exp-2/weights/best.pt"
  }
}
```

### 4.3 模型切换

**PUT** `/api/v1/model`

请求体：

```json
{
  "model_name": "exp-3"
}
```

返回示例：

```json
{
  "code": 200,
  "msg": "已切换到模型: exp-3",
  "data": {
    "new_model_name": "exp-3",
    "new_weights_path": "E:/shixi WORK/SOP/yolo/runs/train/exp-3/weights/best.pt"
  }
}
```

### 4.4 图片检测

**POST** `/api/v1/detection/detect`

请求类型：`multipart/form-data`

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| file | file | 是 | 图片文件，支持 JPG/PNG/BMP，最大 10MB |
| product_type | string | 是 | 工件类型名称 |
| expected_count | int | 是 | 期望螺丝数量，必须大于 0 |
| confidence | float | 否 | 置信度阈值，默认 0.25，范围 0 到 1 之间 |

返回示例：

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "product_type": "A型支架",
    "expected_count": 4,
    "actual_count": 4,
    "result": "pass",
    "confidence": 0.9123,
    "inference_time_ms": 36.8,
    "detections": [
      {
        "class_id": 0,
        "class_name": "螺丝",
        "confidence": 0.93,
        "bbox": {
          "x1": 12.4,
          "y1": 33.1,
          "x2": 56.7,
          "y2": 88.2
        }
      }
    ]
  }
}
```

### 4.5 标注图片

**POST** `/api/v1/detection/annotated`

请求参数与检测接口相同，但返回的是图片二进制流。

返回类型：`image/jpeg`

前端建议：直接把响应当作图片 blob 显示。

### 4.6 从服务器目录启动训练

**POST** `/api/v1/training/start`

请求体：

```json
{
  "dataset_path": "E:/datasets/screws_v2",
  "classes": [0, 1],
  "epochs": 50,
  "batch": 16,
  "imgsz": 640,
  "name": "exp",
  "patience": 20,
  "lr0": 0.01
}
```

说明：

- `dataset_path` 必须包含 `images/train` 和 `labels/train`
- 如果没有 `val` 集，后端会自动按 80/20 划分
- `classes` 是 class_id 数组，不要求连续编号，例如 `[0]`、`[0, 2, 5]`

### 4.7 上传 ZIP 并启动训练

**POST** `/api/v1/training/start-upload`

请求类型：`multipart/form-data`

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| file | file | 是 | YOLO 数据集 ZIP |
| classes | string | 是 | JSON 数组字符串，例如 `[0,2,5]` |
| epochs | int | 否 | 默认 50 |
| batch | int | 否 | 默认 16 |
| imgsz | int | 否 | 默认 640 |
| name | string | 否 | 默认 `exp` |
| patience | int | 否 | 默认 20 |
| lr0 | float | 否 | 默认 0.01 |

`classes` 示例（不要求连续）：

```text
[0,2]
```

返回结构与 `/api/v1/training/start` 一致。

### 4.8 查询训练状态

**GET** `/api/v1/training/status`

返回示例：

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "task_id": "a1b2c3d4",
    "status": "running",
    "name": "exp",
    "config": {
      "dataset_path": "E:/datasets/screws_v2",
      "classes": [0, 2],
      "epochs": 50,
      "batch": 16,
      "imgsz": 640,
      "name": "exp",
      "patience": 20,
      "lr0": 0.01
    },
    "progress": {
      "current_epoch": 12,
      "total_epochs": 50,
      "box_loss": 0.123,
      "cls_loss": 0.045,
      "dfl_loss": 0.067,
      "map50": 0.82,
      "map50_95": 0.61,
      "progress_pct": 24.0
    },
    "best_weights_path": "E:/shixi WORK/SOP/yolo/runs/train/exp/weights/best.pt",
    "error": null,
    "started_at": "2026-06-10 10:10:00",
    "finished_at": null
  }
}
```

如果当前没有训练任务，返回 `404`。

### 4.9 停止训练

**POST** `/api/v1/training/stop`

返回示例：

```json
{
  "code": 200,
  "msg": "训练停止请求已发送",
  "data": null
}
```

如果没有正在运行的任务，返回 `404`。

### 4.10 训练实验列表

**GET** `/api/v1/training/list`

返回示例：

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "experiments": [
      {
        "name": "exp-2",
        "status": "completed",
        "best_weights": "E:/shixi WORK/SOP/yolo/runs/train/exp-2/weights/best.pt",
        "created": "2026-06-08 14:20:33"
      }
    ],
    "current_model": "exp-2"
  }
}
```

## 5. 前端调用建议

- 图片检测接口用 `multipart/form-data` 发送，前端可用 `FormData`。
- 标注图片接口返回 `image/jpeg`，前端需要把响应转成 `Blob` 后显示。
- 训练接口返回的是统一 JSON，可直接按 `code/msg/data` 解析。
- 若前端和后端不在同一台机器上，后端必须监听 `0.0.0.0`，并确保 Windows 防火墙放行 `10000` 端口。

## 6. 推荐联调顺序

1. 先访问 `GET /test`，确认返回 `ok`
2. 再访问 `GET /api/v1/health`，确认后端和模型正常
3. 然后调用 `POST /api/v1/detection/detect`
4. 需要看效果时再调用 `POST /api/v1/detection/annotated`
