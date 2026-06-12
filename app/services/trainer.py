# ============================================================
# trainer.py - YOLO 模型训练服务
# ============================================================
"""
后台训练管理：数据集校验、YAML 配置生成、异步训练、进度跟踪。
"""
import logging
import os
import random
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

import yaml

from app.config import settings

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _build_class_index_map(classes: list[int]) -> dict[int, int]:
    """把原始 class_id 映射成 YOLO 训练所需的连续索引。"""
    return {class_id: index for index, class_id in enumerate(classes)}


def _remap_label_text(label_text: str, class_index_map: dict[int, int], source: Path) -> str:
    """把单个 YOLO 标签文件中的 class_id 重映射为连续索引。"""
    remapped_lines: list[str] = []
    for line_no, raw_line in enumerate(label_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"标注文件格式错误: {source} 第 {line_no} 行")

        try:
            original_class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"标注文件类别 ID 非整数: {source} 第 {line_no} 行") from exc

        if original_class_id not in class_index_map:
            raise ValueError(
                f"标注文件包含未声明的类别 ID: {source} 第 {line_no} 行, class_id={original_class_id}"
            )

        parts[0] = str(class_index_map[original_class_id])
        remapped_lines.append(" ".join(parts))

    return "\n".join(remapped_lines) + ("\n" if remapped_lines else "")


def _build_train_kwargs(config: dict, train_dir: str, device) -> dict:
    """Build Ultralytics arguments for training inside the API background thread."""
    return {
        "data": config["_dataset_yaml"],
        "epochs": config["epochs"],
        "batch": config["batch"],
        "imgsz": config["imgsz"],
        "name": config["name"],
        "patience": config["patience"],
        "lr0": config["lr0"],
        "project": train_dir,
        "device": device,
        "exist_ok": True,
        # Windows DataLoader subprocesses are unstable when training starts
        # from the API's background thread.
        "workers": 0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
    }


def _apply_runtime_task_state(task, trainer) -> None:
    """Expose the actual run name after Ultralytics resolves save_dir."""
    actual_name = Path(trainer.save_dir).name
    task.name = actual_name
    task.train_id = actual_name
    task.config["name"] = actual_name
    task.config["trainId"] = actual_name


def _get_best_weights_path(model) -> str | None:
    """Return the checkpoint path selected by the active Ultralytics trainer."""
    best = getattr(getattr(model, "trainer", None), "best", None)
    return str(best) if best and Path(best).is_file() else None


def _request_stop(trainer) -> None:
    """Set the stop flag used by current Ultralytics trainer versions."""
    trainer.stop = True


class TrainingTask:
    """单个训练任务的状态容器。"""

    def __init__(self, task_id: str, train_id: str, name: str, config: dict):
        self.task_id = task_id
        self.train_id = train_id
        self.name = name
        self.status = "pending"
        self.config = config
        self.current_epoch = 0
        self.total_epochs = config.get("epochs", 50)
        self.box_loss = None
        self.cls_loss = None
        self.dfl_loss = None
        self.map50 = None
        self.map50_95 = None
        self.best_weights_path = None
        self.error = None
        self.started_at = None
        self.finished_at = None
        self._stop_flag = False

    def to_dict(self) -> dict:
        progress_pct = (
            round(self.current_epoch / self.total_epochs * 100, 1)
            if self.total_epochs > 0
            else 0.0
        )
        return {
            "task_id": self.task_id,
            "trainId": self.train_id,
            "status": self.status,
            "name": self.name,
            "config": self.config,
            "progress": {
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "box_loss": self.box_loss,
                "cls_loss": self.cls_loss,
                "dfl_loss": self.dfl_loss,
                "map50": self.map50,
                "map50_95": self.map50_95,
                "progress_pct": progress_pct,
            },
            "best_weights_path": self.best_weights_path,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class TrainingService:
    """训练服务，管理训练任务生命周期。"""

    def __init__(self):
        self._current_task: TrainingTask | None = None
        self._lock = threading.Lock()

    def start_training(self, config: dict) -> dict:
        """启动训练任务。"""
        with self._lock:
            if self._current_task and self._current_task.status == "running":
                raise RuntimeError(
                    f"已有训练任务在运行: {self._current_task.name} "
                    f"(epoch {self._current_task.current_epoch}/{self._current_task.total_epochs})"
                )

            dataset_path = config["dataset_path"]
            ds_info = self._validate_dataset(dataset_path)

            if not ds_info["has_val"]:
                logger.info("验证集不存在，自动从训练集划分 80/20 ...")
                self._auto_split_train_val(dataset_path)

            dataset_path = self._normalize_training_dataset(dataset_path, config["classes"])
            yaml_path = self._generate_dataset_yaml(dataset_path, config["classes"])
            config["_dataset_yaml"] = yaml_path
            config["dataset_info"] = ds_info

            task_id = uuid.uuid4().hex[:8]
            task = TrainingTask(task_id, config["trainId"], config["name"], config)
            self._current_task = task

            thread = threading.Thread(
                target=self._run_training,
                args=(task, config),
                daemon=True,
            )
            thread.start()

            return task.to_dict()

    def get_status(self) -> dict | None:
        with self._lock:
            if self._current_task is None:
                return None
            return self._current_task.to_dict()

    def stop_training(self) -> bool:
        with self._lock:
            if self._current_task is None or self._current_task.status != "running":
                return False
            self._current_task._stop_flag = True
            return True

    def list_experiments(self) -> list[dict]:
        """扫描本地 runs/train/ 目录，列出所有训练实验。"""
        train_dir = settings.get_runs_train_dir()
        if not os.path.isdir(train_dir):
            return []

        experiments = []
        for name in sorted(os.listdir(train_dir)):
            exp_path = os.path.join(train_dir, name)
            if not os.path.isdir(exp_path):
                continue

            best_pt = os.path.join(exp_path, "weights", "best.pt")
            info = {
                "trainId": name,
                "name": name,
                "status": "incomplete",
                "best_weights": None,
                "created": None,
            }

            if os.path.isfile(best_pt):
                info["status"] = "completed"
                info["best_weights"] = best_pt
            elif os.path.isdir(os.path.join(exp_path, "weights")):
                info["status"] = "failed"

            try:
                mtime = os.path.getmtime(exp_path)
                info["created"] = datetime.fromtimestamp(mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except OSError:
                pass

            experiments.append(info)

        return experiments

    def _validate_dataset(self, dataset_path: str) -> dict:
        """校验数据集目录结构。"""
        if not os.path.isdir(dataset_path):
            raise ValueError(f"数据集目录不存在: {dataset_path}")

        img_train = os.path.join(dataset_path, "images", "train")
        lbl_train = os.path.join(dataset_path, "labels", "train")

        if not os.path.isdir(img_train):
            raise ValueError(f"训练图片目录不存在: {dataset_path}/images/train")
        if not os.path.isdir(lbl_train):
            raise ValueError(f"训练标注目录不存在: {dataset_path}/labels/train")

        train_images = [
            f for f in os.listdir(img_train) if Path(f).suffix.lower() in _IMAGE_EXTS
        ]
        train_labels = [f for f in os.listdir(lbl_train) if f.endswith(".txt")]

        if len(train_images) == 0:
            raise ValueError(f"训练图片目录为空: {img_train}")
        if len(train_labels) == 0:
            raise ValueError(f"训练标注目录为空: {lbl_train}")

        img_stems = {Path(f).stem for f in train_images}
        lbl_stems = {Path(f).stem for f in train_labels}
        missing_labels = img_stems - lbl_stems
        if missing_labels:
            logger.warning("有 %s 张图片没有对应的标注文件", len(missing_labels))

        img_val = os.path.join(dataset_path, "images", "val")
        lbl_val = os.path.join(dataset_path, "labels", "val")
        has_val = os.path.isdir(img_val) and os.path.isdir(lbl_val)

        val_images = 0
        if has_val:
            val_images = len(
                [f for f in os.listdir(img_val) if Path(f).suffix.lower() in _IMAGE_EXTS]
            )

        info = {
            "train_images": len(train_images),
            "train_labels": len(train_labels),
            "has_val": has_val,
            "val_images": val_images,
        }
        logger.info("数据集校验通过: %s", info)
        return info

    def _auto_split_train_val(self, dataset_path: str, ratio: float = 0.2):
        """从 train 中随机移动 20% 的文件到 val。"""
        img_train_dir = os.path.join(dataset_path, "images", "train")
        lbl_train_dir = os.path.join(dataset_path, "labels", "train")
        img_val_dir = os.path.join(dataset_path, "images", "val")
        lbl_val_dir = os.path.join(dataset_path, "labels", "val")

        os.makedirs(img_val_dir, exist_ok=True)
        os.makedirs(lbl_val_dir, exist_ok=True)

        train_images = sorted(
            [f for f in os.listdir(img_train_dir) if Path(f).suffix.lower() in _IMAGE_EXTS]
        )

        random.seed(42)
        random.shuffle(train_images)
        val_count = max(1, int(len(train_images) * ratio))
        val_set = set(train_images[:val_count])

        moved = 0
        for img_name in val_set:
            stem = Path(img_name).stem
            src_img = os.path.join(img_train_dir, img_name)
            dst_img = os.path.join(img_val_dir, img_name)
            shutil.move(src_img, dst_img)

            lbl_name = stem + ".txt"
            src_lbl = os.path.join(lbl_train_dir, lbl_name)
            dst_lbl = os.path.join(lbl_val_dir, lbl_name)
            if os.path.isfile(src_lbl):
                shutil.move(src_lbl, dst_lbl)

            moved += 1

        logger.info("自动划分完成: 移动 %s 张图片到验证集", moved)

    def _generate_dataset_yaml(self, dataset_path: str, classes: list[int]) -> str:
        """生成 YOLO 数据集配置 YAML 文件。"""
        nc = len(classes)
        names = [f"class_{class_id}" for class_id in classes]

        yaml_content = {
            "path": os.path.abspath(dataset_path),
            "train": "images/train",
            "val": "images/val",
            "nc": nc,
            "names": names,
        }

        yaml_path = os.path.join(dataset_path, "dataset.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)

        logger.info("数据集 YAML 已生成: %s", yaml_path)
        return yaml_path

    def _normalize_training_dataset(self, dataset_path: str, classes: list[int]) -> str:
        """复制一份训练数据集，并把标签中的 class_id 重映射为连续索引。

        Ultralytics 训练时要求标签类别必须从 0 开始连续编号。
        当外部传入的 `classes` 使用原始业务 class_id（例如 [1, 2]）时，
        这里会生成一份中间数据集：
        - 图片原样复制
        - 标签文件第一列重映射为 0..n-1
        - YAML 只引用这份中间数据集
        """
        class_index_map = _build_class_index_map(classes)
        normalized_root = os.path.join(dataset_path, "_normalized")

        if os.path.isdir(normalized_root):
            shutil.rmtree(normalized_root)

        for split in ("train", "val"):
            src_img_dir = os.path.join(dataset_path, "images", split)
            src_lbl_dir = os.path.join(dataset_path, "labels", split)
            if not os.path.isdir(src_img_dir) or not os.path.isdir(src_lbl_dir):
                continue

            dst_img_dir = os.path.join(normalized_root, "images", split)
            dst_lbl_dir = os.path.join(normalized_root, "labels", split)
            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_lbl_dir, exist_ok=True)

            for name in sorted(os.listdir(src_img_dir)):
                if Path(name).suffix.lower() not in _IMAGE_EXTS:
                    continue
                shutil.copy2(os.path.join(src_img_dir, name), os.path.join(dst_img_dir, name))

            for name in sorted(os.listdir(src_lbl_dir)):
                if not name.endswith(".txt"):
                    continue

                src_lbl_path = os.path.join(src_lbl_dir, name)
                dst_lbl_path = os.path.join(dst_lbl_dir, name)
                with open(src_lbl_path, "r", encoding="utf-8") as f:
                    remapped = _remap_label_text(
                        f.read(),
                        class_index_map,
                        Path(src_lbl_path),
                    )
                with open(dst_lbl_path, "w", encoding="utf-8") as f:
                    f.write(remapped)

        logger.info("训练数据集已重映射到连续索引目录: %s", normalized_root)
        return normalized_root

    def _run_training(self, task: TrainingTask, config: dict):
        """后台线程：执行 YOLO 训练。"""
        try:
            from ultralytics import YOLO

            task.status = "running"
            task.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            task.total_epochs = config["epochs"]
            logger.info("[训练开始] %s (task_id=%s)", task.name, task.task_id)

            weights_path = settings.get_base_weights_path()
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    f"预训练权重不存在: {weights_path}，请确认 {settings.SOP_BASE_WEIGHTS} 已放在当前项目内"
                )

            model = YOLO(weights_path)

            def on_train_epoch_end(trainer):
                _apply_runtime_task_state(task, trainer)
                task.current_epoch = trainer.epoch + 1

                metrics = getattr(trainer, "metrics", None)
                if metrics:
                    task.box_loss = round(metrics.get("train/box_loss", 0) or 0, 4)
                    task.cls_loss = round(metrics.get("train/cls_loss", 0) or 0, 4)
                    task.dfl_loss = round(metrics.get("train/dfl_loss", 0) or 0, 4)
                    task.map50 = round(metrics.get("metrics/mAP50(B)", 0) or 0, 4)
                    task.map50_95 = round(
                        metrics.get("metrics/mAP50-95(B)", 0) or 0, 4
                    )

                logger.info(
                    "[训练进度] %s epoch %s/%s box_loss=%s cls_loss=%s",
                    task.name,
                    task.current_epoch,
                    task.total_epochs,
                    task.box_loss,
                    task.cls_loss,
                )

                if task._stop_flag:
                    logger.info("[训练停止] 用户请求停止: %s", task.name)
                    _request_stop(trainer)

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            train_dir = settings.get_runs_train_dir()
            device = settings.get_resolved_device()
            if device.startswith("cuda:"):
                device = int(device.split(":")[-1])

            train_kwargs = _build_train_kwargs(config, train_dir, device)
            model.train(**train_kwargs)

            _apply_runtime_task_state(task, model.trainer)
            task.best_weights_path = _get_best_weights_path(model)
            task.status = "completed"
            task.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                "[训练完成] %s best_weights=%s",
                task.name,
                task.best_weights_path,
            )

        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
            task.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.error("[训练失败] %s: %s", task.name, exc, exc_info=True)


_training_service: TrainingService | None = None


def get_trainer() -> TrainingService:
    """获取训练服务实例。"""
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service
