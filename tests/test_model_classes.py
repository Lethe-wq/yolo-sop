import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _install_ultralytics_stub(monkeypatch):
    module = ModuleType("ultralytics")
    module.YOLO = lambda path: SimpleNamespace(names={0: "bolt", 1: "nut"}, path=path)
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def _load_detection_module(monkeypatch):
    _install_ultralytics_stub(monkeypatch)
    import app.routers.detection as detection_module

    return importlib.reload(detection_module)


def _load_model_module(monkeypatch):
    _install_ultralytics_stub(monkeypatch)
    import app.services.model as model_module

    return importlib.reload(model_module)


def test_model_info_uses_model_native_class_names_without_default_override(monkeypatch):
    detection = _load_detection_module(monkeypatch)
    service = SimpleNamespace(
        native_names={0: "bolt", 1: "nut"},
        class_names_override={},
        device_name="cpu",
        model_path="E:/models/custom.pt",
    )
    app = FastAPI()
    app.include_router(detection.router)
    monkeypatch.setattr(detection, "get_service", lambda: service)

    response = TestClient(app).get("/api/v1/model")

    assert response.status_code == 200
    assert response.json()["data"]["classes"] == {
        "0": "bolt",
        "1": "nut",
    }
    assert response.json()["data"]["name"] == "custom"


def test_model_info_uses_run_directory_name_for_training_weights(monkeypatch):
    detection = _load_detection_module(monkeypatch)
    service = SimpleNamespace(
        native_names={0: "bolt"},
        class_names_override={},
        device_name="cpu",
        model_path="E:/workspace/yolo-api/runs/train/train-007/weights/best.pt",
    )
    app = FastAPI()
    app.include_router(detection.router)
    monkeypatch.setattr(detection, "get_service", lambda: service)

    response = TestClient(app).get("/api/v1/model")

    assert response.status_code == 200
    assert response.json()["data"]["name"] == "train-007"


def test_reload_model_clears_old_class_override_when_no_new_override(monkeypatch, tmp_path):
    model = _load_model_module(monkeypatch)
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"weights")
    service = model.YOLOService.__new__(model.YOLOService)
    service.model_path = "E:/runs/train/train-old/weights/best.pt"
    service.class_names_override = {0: "螺丝"}

    service.reload_model(str(weights_path))

    assert service.class_names_override == {}


def test_switch_model_accepts_model_path(monkeypatch, tmp_path):
    detection = _load_detection_module(monkeypatch)
    weights_path = tmp_path / "weights" / "best.pt"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"weights")
    service = SimpleNamespace(reload_model=lambda path: setattr(service, "loaded_path", path))
    app = FastAPI()
    app.include_router(detection.router)
    monkeypatch.setattr(detection, "get_service", lambda: service)

    response = TestClient(app).put(
        "/api/v1/model",
        json={"model_path": str(weights_path)},
    )

    assert response.status_code == 200
    assert response.json()["data"]["new_weights_path"] == str(weights_path)
    assert service.loaded_path == str(weights_path)


def test_detect_rejects_missing_model_path_with_400(monkeypatch):
    detection = _load_detection_module(monkeypatch)
    app = FastAPI()
    app.include_router(detection.router)

    response = TestClient(app).post(
        "/api/v1/detection/detect",
        files={"file": ("sample.jpg", b"image", "image/jpeg")},
        data={"product_type": "bolt", "expected_count": "1"},
    )

    assert response.status_code == 400
    assert "model_path" in response.json()["detail"]


def test_detect_switches_model_before_inference_when_model_path_changes(monkeypatch, tmp_path):
    detection = _load_detection_module(monkeypatch)
    weights_path = tmp_path / "weights" / "best.pt"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"weights")

    class ServiceStub:
        def __init__(self):
            self.model_path = "E:/old/best.pt"
            self.reload_calls = []

        def reload_model(self, path):
            self.reload_calls.append(path)
            self.model_path = path

        def predict(self, image, conf):
            return (
                [
                    {
                        "class_id": 0,
                        "class_name": "bolt",
                        "confidence": 0.9,
                        "bbox": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
                    }
                ],
                0.01,
            )

    service = ServiceStub()
    app = FastAPI()
    app.include_router(detection.router)
    monkeypatch.setattr(detection, "get_service", lambda: service)
    monkeypatch.setattr(detection, "validate_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(detection, "decode_image", lambda content: content)

    response = TestClient(app).post(
        "/api/v1/detection/detect",
        files={"file": ("sample.jpg", b"image", "image/jpeg")},
        data={
            "product_type": "bolt",
            "expected_count": "1",
            "model_path": str(weights_path),
        },
    )

    assert response.status_code == 200
    assert service.reload_calls == [str(weights_path)]
    assert response.json()["data"]["actual_count"] == 1
