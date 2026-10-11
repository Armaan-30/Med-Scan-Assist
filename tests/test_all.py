"""Unit and integration test suite for MedScan Assist."""

import io
import json
import os
import pytest
from PIL import Image
import numpy as np

from app import app
from config.config import Config
from services import image_validator
from services import preprocessing
from services.image_validator import ImageValidator
from services.prediction_service import PredictionService
from models.model_loader import ModelLoader


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image_bytes():
    """Create synthetic test image bytes (PNG)."""
    img = Image.new("RGB", (256, 256), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_config_paths():
    """Verify configuration initialization and defaults."""
    assert os.path.exists(Config.BASE_DIR)
    assert len(Config.CLASS_NAMES) == 2
    assert "NORMAL" in Config.CLASS_NAMES
    assert "PNEUMONIA" in Config.CLASS_NAMES
    assert Config.IMAGE_SIZE == 224


def test_validator_allowed_extension():
    """Test file extension validation."""
    valid, err = image_validator.validate_file_extension("chest_xray.png")
    assert valid is True
    assert err is None

    invalid, err = image_validator.validate_file_extension("notes.pdf")
    assert invalid is False
    assert "Unsupported file extension" in err


def test_validator_magic_bytes(sample_image_bytes):
    """Test MIME type verification from magic bytes."""
    valid, mime, err = image_validator.validate_mime_type(sample_image_bytes)
    assert valid is True
    assert mime == "image/png"
    assert err is None

    fake_bytes = b"NOT_AN_IMAGE_HEADER_AT_ALL"
    invalid, _, err = image_validator.validate_mime_type(fake_bytes)
    assert invalid is False
    assert "does not match any accepted image format" in err


def test_validator_size_limit():
    """Test file size limit check."""
    # 5 MB test bytes
    small_bytes = b"0" * (5 * 1024 * 1024)
    valid, size_mb, err = image_validator.validate_file_size(small_bytes, max_size_mb=10)
    assert valid is True
    assert err is None

    # 15 MB test bytes
    large_bytes = b"0" * (15 * 1024 * 1024)
    invalid, size_mb, err = image_validator.validate_file_size(large_bytes, max_size_mb=10)
    assert invalid is False
    assert "exceeds maximum allowed size" in err


def test_validator_run_all_checks(sample_image_bytes):
    """Test full technical quality verification suite."""
    res = image_validator.run_all_checks(file_bytes=sample_image_bytes)
    assert res["is_valid"] is True
    assert len(res["errors"]) == 0
    assert "quality_info" in res
    assert "disclaimer" in res


def test_preprocessing():
    """Test preprocessed tensor shape and value ranges."""
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    tensor = preprocessing.preprocess_pil_image(img, device="cpu")
    assert tensor.shape == (1, 3, 224, 224)
    assert not np.isnan(tensor.numpy()).any()


def test_api_health(client):
    """Test GET /api/health endpoint."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "online"
    assert "model_loaded" in data
    assert "device" in data


def test_api_metrics(client):
    """Test GET /api/metrics endpoint."""
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "available" in data
    if data["available"]:
        # Verify accuracy either at root level or in nested structure
        accuracy = data.get("accuracy") or data.get("metrics", {}).get("accuracy") or data.get("metrics", {}).get("overall_metrics", {}).get("accuracy")
        assert accuracy is not None
        assert 0.0 <= accuracy <= 1.0


def test_api_sample_images(client):
    """Test GET /api/sample-images endpoint."""
    resp = client.get("/api/sample-images")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "samples" in data
    assert len(data["samples"]) > 0


def test_api_analyze_valid_image(client, sample_image_bytes):
    """Test POST /api/analyze with valid image."""
    data = {
        "image": (io.BytesIO(sample_image_bytes), "test_xray.png")
    }
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["prediction"] in Config.CLASS_NAMES
    assert "NORMAL" in payload["scores"]
    assert "PNEUMONIA" in payload["scores"]
    assert "confidence" in payload
    assert "gradcam_url" in payload
    assert "disclaimer" in payload


def test_api_analyze_invalid_extension(client):
    """Test POST /api/analyze with forbidden file extension."""
    data = {
        "image": (io.BytesIO(b"dummy text content"), "report.txt")
    }
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["success"] is False
