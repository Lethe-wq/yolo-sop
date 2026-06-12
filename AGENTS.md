# AGENTS.md

本文件为在此仓库中工作的编码代理提供项目背景、开发约定和验证要求。

## 项目概览

这是一个基于 FastAPI 和 Ultralytics YOLO 的本地工件检测服务，主要提供：

- 图片目标检测及 JSON 结果
- 检测结果标注图
- 模型信息查询与热切换
- YOLO 训练任务启动、状态查询、停止和实验列表

服务默认运行在 `http://127.0.0.1:10000`，API 路由统一使用 `/api/v1` 前缀。

## 目录结构

- `main.py`：FastAPI 应用入口、生命周期、CORS 和路由注册
- `app/config.py`：环境变量、模型路径和设备配置
- `app/routers/`：HTTP 接口层
- `app/schemas/`：Pydantic 请求与响应模型
- `app/services/`：推理、训练和图片标注业务逻辑
- `app/utils/`：图片校验、解码和编码工具
- `README.md`：完整架构、配置、接口和对接文档
- `.env`：本地运行配置，不应提交或输出其中的敏感内容
- `*.pt`：模型权重，不要无故修改、复制或提交新的大文件

## 开发环境

项目使用 Python 3.10+。现有 YOLO 环境还需要提供 `torch`、`ultralytics`、`numpy`、`opencv-python` 和 `PyYAML`。

安装 API 依赖：

```powershell
python -m pip install -r requirements.txt
```

启动服务：

```powershell
python main.py
```

开发时可使用热重载，但每次重载都会重新加载模型：

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 10000
```

## 配置约定

配置由 `app.config.Settings` 从 `.env` 读取。新增配置时：

1. 在 `Settings` 中提供类型和合理默认值。
2. 使用 `SOP_` 前缀。
3. 同步更新 `README.md` 的配置说明。
4. 不要硬编码开发者机器的绝对路径。
5. 不要在日志、测试输出或文档中泄露 `.env` 内容。

模型和训练数据通常位于仓库外部，由 `SOP_YOLO_DIR` 指向。代码不得假设这些外部文件在 CI 或其他开发环境中一定存在。

## 编码约定

- 遵循现有分层：路由负责 HTTP 交互，Schema 负责数据契约，Service 负责业务逻辑。
- 新接口应使用 Pydantic 模型声明请求和响应，并保持统一响应格式。
- 用户输入错误返回 `400`，资源不存在返回 `404`，状态冲突返回 `409`。
- 文件上传必须经过大小、扩展名和图片解码校验。
- 阻塞型推理或训练工作不得直接阻塞 FastAPI 事件循环。
- GPU 推理当前通过单工作线程串行执行；修改并发方式前需考虑 CUDA 和模型实例的线程安全。
- 全局模型通过 `init_service()` 初始化、`get_service()` 获取，不要在每次请求中重复加载。
- 训练服务同一时间只允许一个运行中的任务，修改时应保持锁和状态更新的一致性。
- 训练 ZIP 必须通过 `DatasetArchiveService` 解压；不得使用 `ZipFile.extractall()`。
- 修改归档逻辑时必须覆盖路径穿越、符号链接、文件数和解压大小限制测试。
- 路径操作优先使用 `pathlib.Path` 或 `os.path`，兼容 Windows 路径。
- 保持类型标注，注释只说明不明显的约束和原因。
- 保留现有中文文档和接口说明，所有文本文件使用 UTF-8 编码。

## API 变更要求

修改接口时，同时检查并更新：

- `app/routers/` 中的端点
- `app/schemas/` 中的请求和响应模型
- `README.md` 中的接口文档和调用示例
- 依赖该接口的健康检查、模型管理或训练流程

除非需求明确要求，否则不要更改现有 URL、字段名称、状态码或响应结构，以免破坏 Electron 和 Java 调用方。

## 测试与验证

自动化测试位于 `tests/`。每次修改至少执行与改动范围匹配的验证。

基础语法检查：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
python -m compileall main.py app
```

服务可正常加载模型时，启动后验证：

```powershell
Invoke-RestMethod http://127.0.0.1:10000/api/v1/health
Invoke-RestMethod http://127.0.0.1:10000/api/v1/model
```

涉及检测接口时，应使用真实的小型 JPG 或 PNG 验证：

- 正常图片能够返回检测结果
- 非图片或不支持的扩展名返回 `400`
- 超过大小限制的文件返回 `400`
- `expected_count` 和 `confidence` 的边界校验正确

涉及训练时，避免默认运行耗时的完整训练。优先测试数据集校验、任务冲突、状态转换和停止逻辑；只有需求明确时才启动实际训练。

新增可独立测试的逻辑时，在 `tests/` 中添加 `pytest` 测试，并通过依赖注入或 mock 避免真实加载模型和占用 GPU。

## 修改边界

- 不要修改或删除用户已有的未提交更改。
- 不要将 `.env`、数据集、训练输出、缓存或新增模型权重提交到版本控制。
- 不要自动移动训练数据；`TrainingService` 的自动 train/val 划分会移动文件，相关改动必须格外谨慎。
- 不要为了局部需求进行无关重构。
- 新增依赖时同步更新 `requirements.txt`，并说明其用途。
- 完成改动前查看 `git diff`，运行可行的验证，并明确说明未能执行的验证。
