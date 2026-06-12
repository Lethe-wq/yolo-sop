import io
import json
import zipfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import training
from app.services.dataset_archive import (
    ArchiveLimitExceeded,
    DatasetArchiveError,
)


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("images/train/sample.jpg", b"image")
        archive.writestr("labels/train/sample.txt", b"label")
    return buffer.getvalue()


def _task_result(config: dict) -> dict:
    return {
        "task_id": "task1234",
        "status": "pending",
        "name": config["name"],
        "trainId": config["trainId"],
        "config": config,
        "progress": {
            "current_epoch": 0,
            "total_epochs": config["epochs"],
            "box_loss": None,
            "cls_loss": None,
            "dfl_loss": None,
            "map50": None,
            "map50_95": None,
            "progress_pct": 0.0,
        },
        "best_weights_path": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
    }


class ArchiveStub:
    def __init__(
        self,
        result: Path = Path("E:/datasets/uploaded"),
        error: Exception | None = None,
    ):
        self.result = result
        self.error = error
        self.filename = None

    def prepare(self, source, filename):
        self.filename = filename
        if self.error:
            raise self.error
        return self.result


class TrainerStub:
    def __init__(self, error: Exception | None = None, result: dict | None = None):
        self.error = error
        self.config = None
        self.result = result
        self.status_result = None

    def start_training(self, config):
        self.config = config
        if self.error:
            raise self.error
        return self.result or _task_result(config)

    def get_status(self):
        return self.status_result


def _client(monkeypatch, archive_service, trainer) -> TestClient:
    app = FastAPI()
    app.include_router(training.router)
    monkeypatch.setattr(
        training,
        "_create_archive_service",
        lambda: archive_service,
    )
    monkeypatch.setattr(training, "get_trainer", lambda: trainer)
    return TestClient(app)


def test_start_accepts_class_ids_and_omits_optional_fields(monkeypatch):
    trainer = TrainerStub()
    client = _client(monkeypatch, ArchiveStub(), trainer)

    response = client.post(
        "/api/v1/training/start",
        json={
            "trainId": "train-001",
            "classes": [0, 1],
            "dataset_path": "E:/datasets/screws_v2",
        },
    )

    assert response.status_code == 200
    assert trainer.config == {
        "dataset_path": "E:/datasets/screws_v2",
        "trainId": "train-001",
        "classes": [0, 1],
        "epochs": 50,
        "batch": 16,
        "imgsz": 640,
        "name": "train-001",
        "patience": 20,
        "lr0": 0.01,
    }
    assert response.json()["data"]["trainId"] == "train-001"


def test_start_upload_maps_form_fields_to_training_config(monkeypatch):
    archive = ArchiveStub()
    trainer = TrainerStub()
    client = _client(monkeypatch, archive, trainer)

    response = client.post(
        "/api/v1/training/start-upload",
        files={"file": ("dataset.zip", _zip_bytes(), "application/zip")},
        data={
            "trainId": "train-upload-001",
            "classes": json.dumps([0, 1]),
            "epochs": "25",
            "batch": "8",
            "imgsz": "512",
            "patience": "10",
            "lr0": "0.005",
        },
    )

    assert response.status_code == 200
    assert trainer.config == {
        "dataset_path": str(archive.result),
        "trainId": "train-upload-001",
        "classes": [0, 1],
        "epochs": 25,
        "batch": 8,
        "imgsz": 512,
        "name": "train-upload-001",
        "patience": 10,
        "lr0": 0.005,
    }
    assert archive.filename == "dataset.zip"


def test_start_upload_rejects_invalid_classes_json(monkeypatch):
    client = _client(monkeypatch, ArchiveStub(), TrainerStub())

    response = client.post(
        "/api/v1/training/start-upload",
        files={"file": ("dataset.zip", _zip_bytes(), "application/zip")},
        data={"trainId": "train-001", "classes": "not-json"},
    )

    assert response.status_code == 400
    assert "classes" in response.json()["detail"]


def test_start_upload_rejects_non_array_classes(monkeypatch):
    client = _client(monkeypatch, ArchiveStub(), TrainerStub())

    response = client.post(
        "/api/v1/training/start-upload",
        files={"file": ("dataset.zip", _zip_bytes(), "application/zip")},
        data={"trainId": "train-001", "classes": '{"0": 1}'},
    )

    assert response.status_code == 400


def test_start_upload_accepts_non_contiguous_business_class_ids(monkeypatch):
    client = _client(monkeypatch, ArchiveStub(), TrainerStub())

    response = client.post(
        "/api/v1/training/start-upload",
        files={"file": ("dataset.zip", _zip_bytes(), "application/zip")},
        data={"trainId": "train-001", "classes": "[0,2]"},
    )

    assert response.status_code == 200


def test_start_upload_maps_archive_validation_to_400(monkeypatch):
    archive = ArchiveStub(error=DatasetArchiveError("bad zip"))
    client = _client(monkeypatch, archive, TrainerStub())

    response = client.post(
        "/api/v1/training/start-upload",
        files={"file": ("dataset.zip", b"bad", "application/zip")},
        data={"trainId": "train-001", "classes": "[0]"},
    )

    assert response.status_code == 400


def test_start_upload_maps_archive_limit_to_413(monkeypatch):
    archive = ArchiveStub(error=ArchiveLimitExceeded("archive too large"))
    client = _client(monkeypatch, archive, TrainerStub())

    response = client.post(
        "/api/v1/training/start-upload",
        files={"file": ("dataset.zip", b"large", "application/zip")},
        data={"trainId": "train-001", "classes": "[0]"},
    )

    assert response.status_code == 413


def test_start_upload_maps_training_conflict_to_409(monkeypatch):
    trainer = TrainerStub(error=RuntimeError("training already running"))
    client = _client(monkeypatch, ArchiveStub(), trainer)

    response = client.post(
        "/api/v1/training/start-upload",
        files={"file": ("dataset.zip", _zip_bytes(), "application/zip")},
        data={"trainId": "train-001", "classes": "[0]"},
    )

    assert response.status_code == 409


def test_start_upload_enforces_shared_parameter_boundaries(monkeypatch):
    client = _client(monkeypatch, ArchiveStub(), TrainerStub())

    response = client.post(
        "/api/v1/training/start-upload",
        files={"file": ("dataset.zip", _zip_bytes(), "application/zip")},
        data={"trainId": "train-001", "classes": "[0]", "epochs": "0"},
    )

    assert response.status_code == 422


def test_start_returns_business_code_201_when_training_is_already_failed(monkeypatch):
    failed_result = _task_result(
        {
            "dataset_path": "E:/datasets/screws_v2",
            "trainId": "train-001",
            "classes": [0],
            "epochs": 50,
            "batch": 16,
            "imgsz": 640,
            "name": "train-001",
            "patience": 20,
            "lr0": 0.01,
        }
    )
    failed_result["status"] = "failed"
    failed_result["error"] = "boom"
    trainer = TrainerStub(result=failed_result)
    client = _client(monkeypatch, ArchiveStub(), trainer)

    response = client.post(
        "/api/v1/training/start",
        json={
            "trainId": "train-001",
            "classes": [0],
            "dataset_path": "E:/datasets/screws_v2",
        },
    )

    assert response.status_code == 200
    assert response.json()["code"] == 201


def test_status_returns_business_code_201_when_current_task_failed(monkeypatch):
    trainer = TrainerStub()
    trainer.status_result = _task_result(
        {
            "dataset_path": "E:/datasets/screws_v2",
            "trainId": "train-001",
            "classes": [0],
            "epochs": 50,
            "batch": 16,
            "imgsz": 640,
            "name": "train-001",
            "patience": 20,
            "lr0": 0.01,
        }
    )
    trainer.status_result["status"] = "failed"
    client = _client(monkeypatch, ArchiveStub(), trainer)

    response = client.get("/api/v1/training/status")

    assert response.status_code == 200
    assert response.json()["code"] == 201
