from pathlib import Path
from types import SimpleNamespace

import yaml

from app.services.trainer import (
    TrainingService,
    _apply_runtime_task_state,
    _build_class_index_map,
    _build_train_kwargs,
    _get_best_weights_path,
    _remap_label_text,
    _request_stop,
)


def test_build_train_kwargs_disables_dataloader_workers_for_background_thread():
    config = {
        "_dataset_yaml": "E:/datasets/screws/dataset.yaml",
        "epochs": 50,
        "batch": 16,
        "imgsz": 640,
        "name": "exp",
        "patience": 20,
        "lr0": 0.01,
    }

    kwargs = _build_train_kwargs(
        config,
        train_dir="E:/yolo/runs/train",
        device=0,
    )

    assert kwargs["workers"] == 0


def test_apply_runtime_task_state_uses_ultralytics_incremented_directory():
    task = SimpleNamespace(name="exp", config={"name": "exp"})
    trainer = SimpleNamespace(
        save_dir=Path("E:/yolo/runs/train/exp-3"),
        best=Path("E:/yolo/runs/train/exp-3/weights/best.pt"),
    )

    _apply_runtime_task_state(task, trainer)

    assert task.name == "exp-3"
    assert task.config["name"] == "exp-3"


def test_get_best_weights_path_uses_ultralytics_actual_checkpoint(tmp_path):
    best = tmp_path / "exp-3" / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"checkpoint")
    model = SimpleNamespace(trainer=SimpleNamespace(best=best))

    assert _get_best_weights_path(model) == str(best)


def test_request_stop_sets_ultralytics_stop_flag():
    trainer = SimpleNamespace(stop=False)

    _request_stop(trainer)

    assert trainer.stop is True


def test_generate_dataset_yaml_uses_generated_class_names_for_ids(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    trainer = TrainingService()

    yaml_path = trainer._generate_dataset_yaml(
        str(dataset_path),
        [0, 1],
    )

    assert Path(yaml_path).is_file()
    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    assert data["nc"] == 2
    assert data["names"] == ["class_0", "class_1"]


def test_remap_label_text_maps_business_class_ids_to_dense_training_indices(tmp_path):
    class_index_map = _build_class_index_map([1, 2])

    remapped = _remap_label_text(
        "1 0.5 0.5 0.2 0.2\n2 0.4 0.4 0.1 0.1\n",
        class_index_map,
        tmp_path / "sample.txt",
    )

    assert remapped == "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n"


def test_normalize_training_dataset_rewrites_label_ids_for_yolo(tmp_path):
    dataset_path = tmp_path / "dataset"
    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "labels" / "train").mkdir(parents=True)
    (dataset_path / "images" / "val").mkdir(parents=True)
    (dataset_path / "labels" / "val").mkdir(parents=True)

    (dataset_path / "images" / "train" / "a.jpg").write_bytes(b"image-a")
    (dataset_path / "images" / "val" / "b.jpg").write_bytes(b"image-b")
    (dataset_path / "labels" / "train" / "a.txt").write_text(
        "1 0.5 0.5 0.2 0.2\n2 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )
    (dataset_path / "labels" / "val" / "b.txt").write_text(
        "2 0.3 0.3 0.1 0.1\n",
        encoding="utf-8",
    )

    trainer = TrainingService()
    normalized = Path(trainer._normalize_training_dataset(str(dataset_path), [1, 2]))

    assert normalized.is_dir()
    assert (normalized / "images" / "train" / "a.jpg").read_bytes() == b"image-a"
    assert (normalized / "images" / "val" / "b.jpg").read_bytes() == b"image-b"
    assert (normalized / "labels" / "train" / "a.txt").read_text(encoding="utf-8") == (
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n"
    )
    assert (normalized / "labels" / "val" / "b.txt").read_text(encoding="utf-8") == (
        "1 0.3 0.3 0.1 0.1\n"
    )
