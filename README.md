# yolo-api

`yolo-api` 是一个基于 FastAPI 与 Ultralytics YOLO 的本地工件检测服务，面向本地部署场景提供目标检测、标注图返回、模型信息查询与热切换，以及训练任务管理能力。面向 GitHub 的开源发布分支仅公开源码与文档；模型权重、数据集、训练输出等运行工件需要由使用方单独提供。

## 功能特性

- 提供图片目标检测接口，返回统一 JSON 结果。
- 提供标注图接口，直接返回带框选与结果横幅的图片。
- 提供健康检查、模型信息查询与模型热切换接口。
- 提供训练任务启动、ZIP 数据集上传启动、状态查询、停止和实验列表接口。
- 使用环境变量管理主机、端口、设备、权重路径与上传限制。
- 通过全局模型实例与串行推理线程，避免每次请求重复加载模型。

## 项目结构

```text
yolo-api/
├─ main.py
├─ requirements.txt
├─ requirements-dev.txt
├─ environment.yml
├─ app/
│  ├─ config.py
│  ├─ routers/
│  ├─ schemas/
│  ├─ services/
│  └─ utils/
├─ docs/
│  ├─ api-reference.md
│  └─ frontend-api.md
└─ tests/
```

- `main.py`：FastAPI 应用入口、生命周期、CORS 与路由注册。
- `app/config.py`：环境变量读取、路径解析、设备与默认配置。
- `app/routers/`：HTTP 接口层。
- `app/schemas/`：Pydantic 请求与响应模型。
- `app/services/`：推理、训练、标注与归档处理逻辑。
- `app/utils/`：图片校验、解码和编码工具。
- `docs/`：接口与前端对接文档。
- `tests/`：自动化测试。

## 快速开始

### 环境要求

- Python 3.10+
- 可用的 YOLO 运行环境，通常需要 `torch`、`ultralytics`、`numpy`、`opencv-python`、`PyYAML`
- Windows、Linux 均可，路径处理以 `pathlib.Path` 为主

如需参考现有环境定义，可查看 `environment.yml`；它是可选参考文件，不是必须的安装入口。

### 安装

```powershell
python -m pip install -r requirements.txt
```

如果当前环境尚未安装 YOLO 相关依赖，请先准备包含 `torch` 和 `ultralytics` 的 Python 环境，再安装本项目 API 依赖。

### 配置

复制示例配置并按本地环境调整：

```powershell
Copy-Item .env.example .env
```

`.env.example` 中的 `SOP_HOST`、`SOP_PORT`、`SOP_DEVICE`、`SOP_MODEL_WEIGHTS`、`SOP_BASE_WEIGHTS` 都可以按需修改。默认值由 `app/config.py` 定义，权重路径支持绝对路径或相对项目根目录的相对路径。由于服务会在启动时立即加载模型，首次启动前必须先准备可用的 YOLO 权重文件，并确保 `SOP_MODEL_WEIGHTS` 与 `SOP_BASE_WEIGHTS` 指向有效路径。

### 运行

```powershell
python main.py
```

如果未先配置有效权重，服务会在启动阶段加载模型时失败。

开发时也可以使用：

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 10000
```

- 默认本地访问地址：`http://127.0.0.1:10000`
- Swagger UI：`http://127.0.0.1:10000/docs`

## API 概览

服务统一使用 `/api/v1` 前缀，主要接口如下：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 健康检查与模型加载状态 |
| `GET` | `/api/v1/model` | 获取当前模型信息 |
| `PUT` | `/api/v1/model` | 热切换当前模型 |
| `POST` | `/api/v1/detection/detect` | 执行检测并返回 JSON 结果 |
| `POST` | `/api/v1/detection/annotated` | 执行检测并返回标注图片 |
| `POST` | `/api/v1/training/start` | 以现有数据集目录启动训练 |
| `POST` | `/api/v1/training/start-upload` | 上传 ZIP 数据集并启动训练 |
| `GET` | `/api/v1/training/status` | 查询当前训练状态 |
| `POST` | `/api/v1/training/stop` | 停止当前训练任务 |
| `GET` | `/api/v1/training/list` | 查询训练实验列表 |

更详细的请求参数、响应结构与调用示例见：

- [docs/api-reference.md](docs/api-reference.md)
- [docs/frontend-api.md](docs/frontend-api.md)

## 模型与数据集说明

- 面向 GitHub 的开源发布分支不会发布模型权重文件，例如 `*.pt`、`*.onnx`、`*.engine`。
- 面向 GitHub 的开源发布分支不会发布训练数据集、训练输出目录 `runs/`、上传归档或生成图片。
- `SOP_MODEL_WEIGHTS` 和 `SOP_BASE_WEIGHTS` 需要指向你本地或外部存储中的实际权重文件。
- 训练数据集同样需要由使用方自行准备，并在训练接口请求或本地配置中提供路径。
- 公开内容以源码、测试和文档为主，便于二次开发、部署和接口集成。

## 开发验证

基础验证命令：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
python -m compileall main.py app
```

服务可正常加载本地权重时，可进一步验证：

```powershell
Invoke-RestMethod http://127.0.0.1:10000/api/v1/health
Invoke-RestMethod http://127.0.0.1:10000/api/v1/model
```

如果涉及检测接口，请使用真实的小型 JPG 或 PNG 文件补充验证上传、边界参数和错误响应。

## License

本项目采用 [MIT License](LICENSE)。
