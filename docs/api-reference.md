# SOP YOLO API 接口文档

本文档面向 Electron、Web 前端和 Java 业务系统对接人员，内容以当前 FastAPI
路由、Pydantic Schema 和测试为准。

## 1. 服务信息

| 项目 | 地址 |
|---|---|
| 默认基础地址 | `http://127.0.0.1:10000` |
| Swagger UI | `http://127.0.0.1:10000/docs` |
| OpenAPI JSON | `http://127.0.0.1:10000/openapi.json` |
| API 前缀 | `/api/v1` |

如果 Docker 将宿主机端口映射为 `18000:10000`，客户端基础地址应改为
`http://127.0.0.1:18000`。

## 2. 通用约定

### 2.1 请求格式

接口使用以下三种请求格式：

| 格式 | 使用场景 |
|---|---|
| `application/json` | 模型切换、服务器目录训练 |
| `multipart/form-data` | 图片检测、标注图、ZIP 数据集上传 |
| 无请求体 | 健康检查、模型信息、训练状态、停止训练、实验列表 |

### 2.2 JSON 成功响应

除标注图接口外，成功响应通常使用以下结构：

```json
{
  "code": 200,
  "msg": "success",
  "data": {}
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 业务状态码，不一定等于 HTTP 状态码 |
| `msg` | string | 业务消息 |
| `data` | object/null | 接口业务数据 |

训练任务已经进入 `failed` 状态时，接口仍返回 HTTP `200`，但 JSON 中的
`code` 为 `201`。调用方应同时检查 HTTP 状态、`code` 和 `data.status`。

### 2.3 错误响应

FastAPI 错误不使用成功响应外壳，常见格式为：

```json
{
  "detail": "错误原因"
}
```

参数校验失败时，`detail` 也可能是错误对象数组：

```json
{
  "detail": [
    {
      "type": "greater_than",
      "loc": [
        "body",
        "expected_count"
      ],
      "msg": "Input should be greater than 0",
      "input": "0",
      "ctx": {
        "gt": 0
      }
    }
  ]
}
```

### 2.4 路径规则

- 图片和 ZIP 文件由客户端上传，文件路径属于客户端。
- `model_path` 和 `dataset_path` 是服务端路径，必须能被 FastAPI 服务进程访问。
- 相对模型路径以项目根目录为基准解析。
- Windows 本地服务可以使用 `C:/models/best.pt` 形式。
- Docker 容器不能直接访问宿主机任意路径。内置模型路径为
  `/app/yolov8s.pt`；额外模型或数据集需要放在容器内或通过 Docker 卷挂载。

## 3. 接口索引

| 模块 | 方法与路径 | 用途 |
|---|---|---|
| 检测 | `POST /api/v1/detection/detect` | 返回 JSON 检测结果 |
| 检测 | `POST /api/v1/detection/annotated` | 返回 JPEG 标注图 |
| 服务 | `GET /api/v1/health` | 查询服务和模型健康状态 |
| 模型 | `GET /api/v1/model` | 查询当前模型信息 |
| 模型 | `PUT /api/v1/model` | 热切换模型 |
| 训练 | `POST /api/v1/training/start` | 使用服务端数据集目录启动训练 |
| 训练 | `POST /api/v1/training/start-upload` | 上传 ZIP 并启动训练 |
| 训练 | `GET /api/v1/training/status` | 查询当前训练任务 |
| 训练 | `POST /api/v1/training/stop` | 请求停止当前训练 |
| 训练 | `GET /api/v1/training/list` | 查询训练实验列表 |

## 4. 检测接口

### 4.1 图片检测

`POST /api/v1/detection/detect`

上传图片并返回目标列表、数量判断、平均置信度和推理耗时。

**Content-Type：** `multipart/form-data`

#### 请求参数

| 参数 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `file` | file | 是 | - | JPG、JPEG、PNG 或 BMP |
| `product_type` | string | 是 | - | 业务侧工件名称 |
| `expected_count` | integer | 是 | - | 必须大于 `0` |
| `model_path` | string | 是 | - | 服务端可访问的 `.pt` 权重路径 |
| `confidence` | number | 否 | `0.25` | 必须大于 `0` 且小于 `1` |

图片默认大小上限为 `10 MB`，解码后的最大宽度或高度为 `4096` 像素。大小上限可由
服务配置修改。

虽然 OpenAPI 表单定义中 `model_path` 允许空值，但当前业务逻辑要求该字段必传；
缺失或空字符串返回 HTTP `400`。

如果 `model_path` 与当前模型不同，服务会先热切换模型，再执行本次推理。

#### curl 示例

```powershell
curl.exe -X POST "http://127.0.0.1:10000/api/v1/detection/detect" `
  -F "file=@C:/images/sample.jpg" `
  -F "product_type=螺丝组件" `
  -F "expected_count=4" `
  -F "model_path=C:/models/best.pt" `
  -F "confidence=0.25"
```

Docker CPU 镜像使用内置模型时：

```powershell
curl.exe -X POST "http://127.0.0.1:10000/api/v1/detection/detect" `
  -F "file=@C:/images/sample.jpg" `
  -F "product_type=通用目标" `
  -F "expected_count=1" `
  -F "model_path=/app/yolov8s.pt"
```

#### 成功响应

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "product_type": "螺丝组件",
    "expected_count": 4,
    "actual_count": 4,
    "result": "pass",
    "confidence": 0.9132,
    "inference_time_ms": 85.4,
    "detections": [
      {
        "class_id": 0,
        "class_name": "bolt",
        "confidence": 0.9321,
        "bbox": {
          "x1": 120.5,
          "y1": 88.0,
          "x2": 205.3,
          "y2": 176.8
        }
      }
    ]
  }
}
```

#### 响应字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `product_type` | string | 原样返回的工件类型 |
| `expected_count` | integer | 期望目标数量 |
| `actual_count` | integer | 实际检测数量 |
| `result` | string | 数量相等为 `pass`，否则为 `fail` |
| `confidence` | number | 所有目标的平均置信度，无目标时为 `0.0` |
| `inference_time_ms` | number | 模型推理耗时，单位毫秒 |
| `detections` | array | 检测目标列表 |
| `detections[].class_id` | integer | 模型类别 ID |
| `detections[].class_name` | string | 模型类别名称 |
| `detections[].confidence` | number | 单个目标置信度 |
| `detections[].bbox` | object | 检测框坐标 |
| `bbox.x1/y1` | number | 左上角坐标 |
| `bbox.x2/y2` | number | 右下角坐标 |

#### 主要错误

| HTTP 状态 | 场景 |
|---|---|
| `400` | `model_path` 缺失、扩展名不支持、文件超限、图片损坏或尺寸超限 |
| `404` | 模型权重不存在 |
| `422` | 表单字段缺失或数值边界不合法 |
| `500` | 模型切换或推理内部错误 |

### 4.2 获取标注图

`POST /api/v1/detection/annotated`

请求参数与图片检测接口相同，服务返回绘制检测框和数量结果后的 JPEG 图片。

**Content-Type：** `multipart/form-data`

**成功响应 Content-Type：** `image/jpeg`

#### 请求参数

| 参数 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `file` | file | 是 | - | JPG、JPEG、PNG 或 BMP |
| `product_type` | string | 是 | - | 业务侧工件名称 |
| `expected_count` | integer | 是 | - | 必须大于 `0` |
| `model_path` | string | 是 | - | 服务端可访问的模型路径 |
| `confidence` | number | 否 | `0.25` | 必须大于 `0` 且小于 `1` |

#### curl 示例

```powershell
curl.exe -X POST "http://127.0.0.1:10000/api/v1/detection/annotated" `
  -F "file=@C:/images/sample.jpg" `
  -F "product_type=螺丝组件" `
  -F "expected_count=4" `
  -F "model_path=C:/models/best.pt" `
  --output "C:/images/sample-annotated.jpg"
```

成功时响应体是图片二进制，不是 JSON。失败时返回 JSON `detail` 错误。

主要错误状态与图片检测接口一致：`400`、`404`、`422`、`500`。

## 5. 服务与模型接口

### 5.1 健康检查

`GET /api/v1/health`

用于应用启动检查和客户端连接检测。

#### curl 示例

```powershell
curl.exe "http://127.0.0.1:10000/api/v1/health"
```

#### 成功响应

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "status": "healthy",
    "model_loaded": true,
    "model_name": "yolov8s",
    "device": "cpu",
    "gpu_name": "CPU"
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | 当前固定返回 `healthy` |
| `model_loaded` | boolean | 当前固定返回 `true` |
| `model_name` | string | 从当前权重路径推导的模型名称 |
| `device` | string | 例如 `cpu` 或 `cuda:0` |
| `gpu_name` | string | GPU 名称；CPU 模式返回 `CPU` |

服务在启动阶段加载模型。如果模型无法加载，应用通常无法正常完成启动，客户端可能表现为
连接失败，而不是从此接口收到不健康 JSON。

### 5.2 查询当前模型

`GET /api/v1/model`

#### curl 示例

```powershell
curl.exe "http://127.0.0.1:10000/api/v1/model"
```

#### 成功响应

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "name": "train-20260610-001",
    "classes": {
      "0": "bolt",
      "1": "nut"
    },
    "device": "cpu",
    "weights_path": "C:/models/best.pt"
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | 权重文件名，或 `runs/train/<实验名>/weights` 中的实验名 |
| `classes` | object | 类别 ID 到类别名称的映射 |
| `device` | string | 当前推理设备 |
| `weights_path` | string | 当前实际加载的服务端权重路径 |

### 5.3 切换模型

`PUT /api/v1/model`

热切换到指定权重。请求完成后，后续检测默认复用新的模型实例。

**Content-Type：** `application/json`

#### 请求体

```json
{
  "model_path": "C:/models/best.pt"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| `model_path` | string | 是 | 服务端绝对路径，或相对项目根目录的路径 |

#### curl 示例

```powershell
curl.exe -X PUT "http://127.0.0.1:10000/api/v1/model" `
  -H "Content-Type: application/json" `
  -d '{"model_path":"runs/train/train-20260610-001/weights/best.pt"}'
```

#### 成功响应

```json
{
  "code": 200,
  "msg": "已切换到模型: train-20260610-001",
  "data": {
    "new_model_name": "train-20260610-001",
    "new_weights_path": "C:/service/runs/train/train-20260610-001/weights/best.pt"
  }
}
```

当前接口只切换运行中模型，不修改配置文件。服务重启后使用启动配置指定的模型。

#### 主要错误

| HTTP 状态 | 场景 |
|---|---|
| `400` | `model_path` 为空 |
| `404` | 权重文件不存在 |
| `422` | JSON 字段缺失或类型不合法 |
| `500` | 模型加载失败 |

## 6. 训练接口

### 6.1 训练参数

两个启动训练接口共享以下参数：

| 参数 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `trainId` | string | 是 | - | 非空；同时作为实验目录名 |
| `classes` | integer[] | 是 | - | 非空、非负、不得重复、元素必须为整数 |
| `epochs` | integer | 否 | `50` | `1` 到 `1000` |
| `batch` | integer | 否 | `16` | `1` 到 `128` |
| `imgsz` | integer | 否 | `640` | `128` 到 `1280` |
| `patience` | integer | 否 | `20` | `0` 到 `200` |
| `lr0` | number | 否 | `0.01` | 大于 `0` 且不超过 `1.0` |

`classes` 可以使用不连续业务类别 ID，例如 `[0,2,5]`。训练准备阶段会将标签类别重映射为
连续索引，并生成 `class_0`、`class_2`、`class_5` 形式的占位类别名称。

同一进程设计为同时只运行一个训练任务。已有运行中任务时返回 HTTP `409`。

### 6.2 使用服务端目录启动训练

`POST /api/v1/training/start`

适用于数据集已经位于服务端文件系统的场景。

**Content-Type：** `application/json`

#### 请求体

```json
{
  "trainId": "train-20260610-001",
  "classes": [
    0,
    2
  ],
  "dataset_path": "C:/sop-data/screws-v2",
  "epochs": 50,
  "batch": 16,
  "imgsz": 640,
  "patience": 20,
  "lr0": 0.01
}
```

| 特有字段 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| `dataset_path` | string | 是 | 服务端数据集根目录 |

数据集至少需要：

```text
dataset-root/
├── images/
│   └── train/
└── labels/
    └── train/
```

`images/train` 和 `labels/train` 必须存在且非空。如果缺少验证集，服务会从训练集自动划分
约 20% 文件到 `images/val` 和 `labels/val`。该操作会移动文件，调用前应确保数据集允许
被服务修改。

#### curl 示例

```powershell
curl.exe -X POST "http://127.0.0.1:10000/api/v1/training/start" `
  -H "Content-Type: application/json" `
  -d '{"trainId":"train-20260610-001","classes":[0,2],"dataset_path":"C:/sop-data/screws-v2","epochs":50,"batch":16,"imgsz":640,"patience":20,"lr0":0.01}'
```

#### 成功响应

训练在线程中异步执行，启动响应通常为 `pending`：

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "task_id": "a1b2c3d4",
    "trainId": "train-20260610-001",
    "status": "pending",
    "name": "train-20260610-001",
    "config": {
      "dataset_path": "C:/sop-data/screws-v2",
      "trainId": "train-20260610-001",
      "name": "train-20260610-001",
      "classes": [
        0,
        2
      ],
      "epochs": 50,
      "batch": 16,
      "imgsz": 640,
      "patience": 20,
      "lr0": 0.01
    },
    "progress": {
      "current_epoch": 0,
      "total_epochs": 50,
      "box_loss": null,
      "cls_loss": null,
      "dfl_loss": null,
      "map50": null,
      "map50_95": null,
      "progress_pct": 0.0
    },
    "best_weights_path": null,
    "error": null,
    "started_at": null,
    "finished_at": null
  }
}
```

#### 主要错误

| HTTP 状态 | 场景 |
|---|---|
| `400` | 数据集目录不存在、目录结构错误、图片或标签为空、标签内容错误 |
| `409` | 已有训练任务正在运行 |
| `422` | JSON 字段缺失、类型或参数边界不合法 |
| `500` | 服务端基础权重或内部文件错误 |

基础预训练权重在后台线程中加载。如果权重缺失，启动请求可能已经返回成功，随后任务状态
变为 `failed`；调用方必须轮询训练状态并检查 `error`。

### 6.3 上传 ZIP 并启动训练

`POST /api/v1/training/start-upload`

上传 ZIP，服务安全解压、验证数据集并启动训练。

**Content-Type：** `multipart/form-data`

#### 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|:---:|---|---|
| `file` | file | 是 | - | 扩展名必须为 `.zip` |
| `trainId` | string | 是 | - | 训练业务 ID 和实验目录名 |
| `classes` | string | 是 | - | JSON 数组字符串，例如 `[0,2]` |
| `epochs` | integer | 否 | `50` | `1` 到 `1000` |
| `batch` | integer | 否 | `16` | `1` 到 `128` |
| `imgsz` | integer | 否 | `640` | `128` 到 `1280` |
| `patience` | integer | 否 | `20` | `0` 到 `200` |
| `lr0` | number | 否 | `0.01` | 大于 `0` 且不超过 `1.0` |

ZIP 默认资源限制：

| 限制 | 默认值 |
|---|---:|
| 上传 ZIP 大小 | 4096 MB |
| 解压后总大小 | 20 GB |
| ZIP 成员数量 | 100000 |

ZIP 禁止路径穿越、绝对路径、Windows 驱动器路径、符号链接以及文件/目录冲突。解压后必须
恰好识别到一个包含 `images/train` 和 `labels/train` 的数据集根目录。

支持以下两种 ZIP 结构：

```text
dataset.zip
├── images/
│   └── train/
└── labels/
    └── train/
```

```text
dataset.zip
└── screws-v2/
    ├── images/
    │   └── train/
    └── labels/
        └── train/
```

#### curl 示例

```powershell
curl.exe -X POST "http://127.0.0.1:10000/api/v1/training/start-upload" `
  -F "file=@C:/datasets/screws-v2.zip" `
  -F "trainId=train-20260610-001" `
  -F "classes=[0,2]" `
  -F "epochs=50" `
  -F "batch=16" `
  -F "imgsz=640" `
  -F "patience=20" `
  -F "lr0=0.01"
```

成功响应结构与服务器目录训练接口相同。

#### 主要错误

| HTTP 状态 | 场景 |
|---|---|
| `400` | 非 ZIP、ZIP 损坏、危险路径、数据集根目录无效、`classes` 不是 JSON 数组 |
| `409` | 已有训练任务正在运行 |
| `413` | ZIP 大小、解压大小或成员数量超过配置限制 |
| `422` | 表单参数类型、训练参数边界或类别 ID 校验失败 |
| `500` | 保存、解压或服务端文件系统错误 |

### 6.4 查询当前训练状态

`GET /api/v1/training/status`

返回当前进程内最近一次训练任务的实时状态。

#### curl 示例

```powershell
curl.exe "http://127.0.0.1:10000/api/v1/training/status"
```

#### 运行中响应

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "task_id": "a1b2c3d4",
    "trainId": "train-20260610-001",
    "status": "running",
    "name": "train-20260610-001",
    "config": {
      "classes": [
        0,
        2
      ],
      "epochs": 50,
      "batch": 16,
      "imgsz": 640,
      "patience": 20,
      "lr0": 0.01
    },
    "progress": {
      "current_epoch": 12,
      "total_epochs": 50,
      "box_loss": 0.3214,
      "cls_loss": 0.1082,
      "dfl_loss": 0.7421,
      "map50": 0.9143,
      "map50_95": 0.6812,
      "progress_pct": 24.0
    },
    "best_weights_path": null,
    "error": null,
    "started_at": "2026-06-10 18:30:00",
    "finished_at": null
  }
}
```

#### 训练状态

| `status` | 说明 |
|---|---|
| `pending` | 任务对象已创建，后台线程尚未正式开始训练 |
| `running` | 正在训练 |
| `completed` | 训练流程完成 |
| `failed` | 训练失败，查看 `error` |

失败状态示例：

```json
{
  "code": 201,
  "msg": "success",
  "data": {
    "task_id": "a1b2c3d4",
    "trainId": "train-20260610-001",
    "status": "failed",
    "name": "train-20260610-001",
    "config": {},
    "progress": {
      "current_epoch": 0,
      "total_epochs": 50,
      "box_loss": null,
      "cls_loss": null,
      "dfl_loss": null,
      "map50": null,
      "map50_95": null,
      "progress_pct": 0.0
    },
    "best_weights_path": null,
    "error": "预训练权重不存在",
    "started_at": "2026-06-10 18:30:00",
    "finished_at": "2026-06-10 18:30:01"
  }
}
```

该失败响应的 HTTP 状态仍为 `200`。

没有训练任务时返回 HTTP `404`：

```json
{
  "detail": "当前没有训练任务"
}
```

### 6.5 停止当前训练

`POST /api/v1/training/stop`

设置停止标记。训练器通常在当前 epoch 结束回调时处理停止请求，因此不是强制立即终止。

#### curl 示例

```powershell
curl.exe -X POST "http://127.0.0.1:10000/api/v1/training/stop"
```

#### 成功响应

```json
{
  "code": 200,
  "msg": "训练停止请求已发送",
  "data": null
}
```

没有 `running` 状态的任务时返回 HTTP `404`：

```json
{
  "detail": "没有正在运行的训练任务"
}
```

### 6.6 查询训练实验列表

`GET /api/v1/training/list`

扫描服务端 `runs/train` 目录，返回训练实验和当前模型名称。

#### curl 示例

```powershell
curl.exe "http://127.0.0.1:10000/api/v1/training/list"
```

#### 成功响应

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "experiments": [
      {
        "trainId": "train-20260610-001",
        "name": "train-20260610-001",
        "status": "completed",
        "best_weights": "C:/service/runs/train/train-20260610-001/weights/best.pt",
        "created": "2026-06-10 18:45:30"
      }
    ],
    "current_model": "train-20260610-001"
  }
}
```

#### 实验状态

| `status` | 判定条件 |
|---|---|
| `completed` | 存在 `weights/best.pt` |
| `failed` | 存在 `weights` 目录，但没有 `best.pt` |
| `incomplete` | 实验目录存在，但没有 `weights` 目录 |

`created` 来自实验目录修改时间，读取失败时为 `null`。

## 7. HTTP 状态码汇总

| HTTP 状态 | 含义 | 常见接口 |
|---|---|---|
| `200` | 请求已处理 | 所有成功 JSON 接口；训练失败状态查询也使用 200 |
| `400` | 业务输入、文件或数据集不合法 | 检测、模型切换、训练启动 |
| `404` | 模型或训练任务不存在 | 检测、模型切换、训练状态、停止训练 |
| `409` | 训练状态冲突 | 两个训练启动接口 |
| `413` | ZIP 资源超限 | ZIP 上传训练 |
| `422` | FastAPI/Pydantic 参数校验失败 | JSON 或表单接口 |
| `500` | 模型加载、文件系统或内部错误 | 模型切换、训练启动 |

## 8. 推荐调用流程

### 8.1 客户端启动检查

1. 调用健康检查。
2. HTTP 200 后读取 `data.device` 和 `data.model_name`。
3. 调用模型信息接口，记录 `weights_path` 和 `classes`。
4. 任一步连接失败时，提示用户检测服务未启动。

### 8.2 图片检测

1. 确定服务端模型路径。
2. 调用 JSON 检测接口。
3. 同时检查 HTTP 状态和 `data.result`。
4. 需要展示检测框时，再调用标注图接口并显示返回的 JPEG。

### 8.3 ZIP 训练并切换模型

1. 组织数据集并压缩为 ZIP。
2. 调用 ZIP 上传训练接口。
3. 保存返回的 `task_id` 和 `trainId`。
4. 每 1 到 3 秒轮询训练状态。
5. `completed` 时读取 `best_weights_path`。
6. 调用模型切换接口加载最佳权重。
7. 再调用模型信息接口确认切换结果。
8. `failed` 时展示 `error`，不要仅判断 HTTP 状态。

## 9. Java 对接示例

以下示例使用 JDK 11+ 自带的 `java.net.http.HttpClient`。

### 9.1 JSON 请求：切换模型

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class SwitchModelExample {
    public static void main(String[] args) throws Exception {
        HttpClient client = HttpClient.newHttpClient();
        String json = """
                {"model_path":"runs/train/train-20260610-001/weights/best.pt"}
                """;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://127.0.0.1:10000/api/v1/model"))
                .header("Content-Type", "application/json")
                .PUT(HttpRequest.BodyPublishers.ofString(json))
                .build();

        HttpResponse<String> response = client.send(
                request,
                HttpResponse.BodyHandlers.ofString()
        );

        System.out.println(response.statusCode());
        System.out.println(response.body());
    }
}
```

Java 文本块需要 JDK 15+。使用 JDK 11 时，可以把 `json` 改为普通转义字符串：

```java
String json = "{\"model_path\":\"runs/train/train-20260610-001/weights/best.pt\"}";
```

### 9.2 multipart 工具方法

JDK 标准库没有高级 multipart 构造器，可以使用以下辅助方法：

```java
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

static byte[] multipartBody(
        String boundary,
        Map<String, String> fields,
        String fileField,
        Path file,
        String contentType
) throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    byte[] crlf = "\r\n".getBytes(StandardCharsets.UTF_8);

    for (Map.Entry<String, String> entry : fields.entrySet()) {
        out.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
        out.write(("Content-Disposition: form-data; name=\"" + entry.getKey()
                + "\"\r\n\r\n").getBytes(StandardCharsets.UTF_8));
        out.write(entry.getValue().getBytes(StandardCharsets.UTF_8));
        out.write(crlf);
    }

    out.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
    out.write(("Content-Disposition: form-data; name=\"" + fileField
            + "\"; filename=\"" + file.getFileName() + "\"\r\n")
            .getBytes(StandardCharsets.UTF_8));
    out.write(("Content-Type: " + contentType + "\r\n\r\n")
            .getBytes(StandardCharsets.UTF_8));
    out.write(Files.readAllBytes(file));
    out.write(crlf);
    out.write(("--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
    return out.toByteArray();
}
```

### 9.3 Java 图片检测

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;

HttpClient client = HttpClient.newHttpClient();
String boundary = "----SopBoundary" + UUID.randomUUID();
Map<String, String> fields = new LinkedHashMap<>();
fields.put("product_type", "螺丝组件");
fields.put("expected_count", "4");
fields.put("model_path", "C:/models/best.pt");
fields.put("confidence", "0.25");

byte[] body = multipartBody(
        boundary,
        fields,
        "file",
        Path.of("C:/images/sample.jpg"),
        "image/jpeg"
);

HttpRequest request = HttpRequest.newBuilder()
        .uri(URI.create("http://127.0.0.1:10000/api/v1/detection/detect"))
        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
        .POST(HttpRequest.BodyPublishers.ofByteArray(body))
        .build();

HttpResponse<String> response = client.send(
        request,
        HttpResponse.BodyHandlers.ofString()
);
System.out.println(response.statusCode());
System.out.println(response.body());
```

### 9.4 Java 保存标注图

```java
import java.net.URI;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;

HttpRequest request = HttpRequest.newBuilder()
        .uri(URI.create("http://127.0.0.1:10000/api/v1/detection/annotated"))
        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
        .POST(HttpRequest.BodyPublishers.ofByteArray(body))
        .build();

HttpResponse<byte[]> response = client.send(
        request,
        HttpResponse.BodyHandlers.ofByteArray()
);

String responseType = response.headers()
        .firstValue("Content-Type")
        .orElse("");

if (response.statusCode() == 200 && responseType.startsWith("image/jpeg")) {
    Files.write(Path.of("C:/images/sample-annotated.jpg"), response.body());
} else {
    String error = new String(response.body(), java.nio.charset.StandardCharsets.UTF_8);
    throw new IllegalStateException(error);
}
```

### 9.5 Java 上传 ZIP 训练

```java
import java.net.URI;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;

String boundary = "----SopBoundary" + UUID.randomUUID();
Map<String, String> fields = new LinkedHashMap<>();
fields.put("trainId", "train-20260610-001");
fields.put("classes", "[0,2]");
fields.put("epochs", "50");
fields.put("batch", "16");
fields.put("imgsz", "640");
fields.put("patience", "20");
fields.put("lr0", "0.01");

byte[] body = multipartBody(
        boundary,
        fields,
        "file",
        Path.of("C:/datasets/screws-v2.zip"),
        "application/zip"
);

HttpRequest request = HttpRequest.newBuilder()
        .uri(URI.create("http://127.0.0.1:10000/api/v1/training/start-upload"))
        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
        .POST(HttpRequest.BodyPublishers.ofByteArray(body))
        .build();

HttpResponse<String> response = client.send(
        request,
        HttpResponse.BodyHandlers.ofString()
);
System.out.println(response.statusCode());
System.out.println(response.body());
```

生产代码应为 HTTP 请求设置连接和请求超时，并使用 JSON 库解析响应，不建议通过字符串
查找判断 `code` 或 `status`。
