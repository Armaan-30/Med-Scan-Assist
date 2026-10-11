"""Services package for MedScan Assist.

Provides modular services for image validation, preprocessing,
Grad-CAM explainability, model prediction orchestration, and report generation.
"""

from services.image_validator import (
    validate_file_extension,
    validate_mime_type,
    validate_file_size,
    validate_image_readable,
    validate_dimensions,
    validate_image_mode,
    check_brightness,
    check_blur,
    run_all_checks,
)
from services.preprocessing import (
    get_training_transforms,
    get_eval_transforms,
    preprocess_for_inference,
    preprocess_pil_image,
)
from services.gradcam_service import (
    get_gradcam_visualization,
    create_side_by_side_image,
)
from services.prediction_service import (
    PredictionService,
    DISCLAIMER_TEXT,
)
from services.report_service import (
    generate_html_report,
    image_file_to_base64_data_uri,
)

__all__ = [
    "validate_file_extension",
    "validate_mime_type",
    "validate_file_size",
    "validate_image_readable",
    "validate_dimensions",
    "validate_image_mode",
    "check_brightness",
    "check_blur",
    "run_all_checks",
    "get_training_transforms",
    "get_eval_transforms",
    "preprocess_for_inference",
    "preprocess_pil_image",
    "get_gradcam_visualization",
    "create_side_by_side_image",
    "PredictionService",
    "DISCLAIMER_TEXT",
    "generate_html_report",
    "image_file_to_base64_data_uri",
]
