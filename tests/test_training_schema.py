import pytest
from pydantic import ValidationError
import os

from app.config import Settings
from app.schemas.training import TrainParameters, TrainStartRequest


def test_settings_training_archive_limits_default_values(monkeypatch):
    monkeypatch.delenv("SOP_MAX_DATASET_ZIP_SIZE_MB", raising=False)
    monkeypatch.delenv("SOP_MAX_DATASET_EXTRACT_SIZE_GB", raising=False)
    monkeypatch.delenv("SOP_MAX_DATASET_FILE_COUNT", raising=False)

    settings = Settings(_env_file=None)

    assert settings.SOP_MAX_DATASET_ZIP_SIZE_MB == 4096
    assert settings.SOP_MAX_DATASET_EXTRACT_SIZE_GB == 20
    assert settings.SOP_MAX_DATASET_FILE_COUNT == 100000


def test_train_start_request_keeps_existing_defaults():
    request = TrainStartRequest(
        dataset_path="E:/datasets/screws",
        classes=[0],
        trainId="train-001",
    )

    assert request.epochs == 50
    assert request.batch == 16
    assert request.imgsz == 640
    assert request.patience == 20
    assert request.lr0 == 0.01
    assert request.trainId == "train-001"


@pytest.mark.parametrize(
    "classes",
    [
        [],
        [-1],
        [0, 0],
        [0, "1"],
    ],
)
def test_train_parameters_rejects_invalid_classes(classes):
    with pytest.raises(ValidationError):
        TrainParameters(classes=classes)


@pytest.mark.parametrize("classes", ([0, 1], [1, 2], [0, 2], [0, 2, 5]))
def test_train_parameters_accepts_declared_business_class_ids(classes):
    request = TrainParameters(classes=classes)

    assert request.classes == classes


def test_train_parameters_rejects_zero_epochs():
    with pytest.raises(ValidationError):
        TrainParameters(classes=[0], epochs=0)


def test_train_start_request_openapi_example_only_shows_required_fields():
    schema = TrainStartRequest.model_json_schema()
    example = schema["example"]

    assert schema["required"] == ["classes", "trainId", "dataset_path"]
    assert example == {
        "classes": [0, 2],
        "trainId": "train-20260610-001",
        "dataset_path": "E:/datasets/screws_v2",
    }


def test_settings_resolve_project_relative_paths():
    settings = Settings(
        _env_file=None,
        SOP_MODEL_WEIGHTS="models/current.pt",
        SOP_BASE_WEIGHTS="yolo26n.pt",
    )

    project_root = __import__("pathlib").Path(__file__).resolve().parents[1]

    assert settings.get_model_path() == str(project_root / "models" / "current.pt")
    assert settings.get_base_weights_path() == str(project_root / "yolo26n.pt")


def test_settings_keep_absolute_paths_unchanged():
    settings = Settings(
        _env_file=None,
        SOP_MODEL_WEIGHTS="E:/models/current.pt",
        SOP_BASE_WEIGHTS="E:/models/base.pt",
    )

    assert settings.get_model_path() == os.path.normpath("E:/models/current.pt")
    assert settings.get_base_weights_path() == os.path.normpath("E:/models/base.pt")
