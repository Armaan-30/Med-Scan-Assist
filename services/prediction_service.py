"""Prediction Service Orchestration for MedScan Assist.

Coordinates input validation, preprocessing, deep learning inference,
uncertainty/abstention thresholding, and Grad-CAM saliency generation.

DISCLAIMER: This is a research prototype output. It is NOT a medical diagnosis.
All medical decisions require evaluation by qualified healthcare professionals.
"""

import logging
import os
import shutil
import uuid
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import Config
from services import image_validator
from services import preprocessing
from services import gradcam_service

logger = logging.getLogger("medscan.prediction_service")

DISCLAIMER_TEXT = (
    "This is a research prototype output. It is NOT a medical diagnosis. "
    "All medical decisions require evaluation by qualified healthcare professionals."
)


class PredictionService:
    """Orchestrates end-to-end clinical scan inference pipeline."""

    def __init__(
        self,
        model_loader: Optional[Any] = None,
        confidence_threshold: Optional[float] = None
    ) -> None:
        """Initialize the PredictionService.

        Args:
            model_loader: Object or callable providing access to the loaded PyTorch model.
                          Can be a model loader instance, an nn.Module, or None.
            confidence_threshold: Minimum confidence [0.0 - 1.0] required before triggering
                                  clinical review flags. Defaults to Config.CONFIDENCE_THRESHOLD or 0.70.
        """
        self.model_loader = model_loader
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(Config, "CONFIDENCE_THRESHOLD", 0.70)
        )
        self.class_names = getattr(Config, "CLASS_NAMES", ["NORMAL", "PNEUMONIA"])
        self.device = getattr(Config, "DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def get_model(self) -> Optional[nn.Module]:
        """Extract or resolve the active PyTorch model from the model loader dependency."""
        if self.model_loader is None:
            return None

        if isinstance(self.model_loader, nn.Module):
            return self.model_loader

        if hasattr(self.model_loader, "get_model") and callable(self.model_loader.get_model):
            return self.model_loader.get_model()

        if hasattr(self.model_loader, "model") and isinstance(self.model_loader.model, nn.Module):
            return self.model_loader.model

        if hasattr(self.model_loader, "load_model") and callable(self.model_loader.load_model):
            return self.model_loader.load_model()

        return None

    def predict(self, image_path: str) -> Dict[str, Any]:
        """Run the complete inference pipeline on an input chest radiograph.

        Pipeline Stages:
        1. Validate technical file integrity, mode, size, and blur/brightness.
        2. Preprocess scan using evaluation transforms.
        3. Execute deep neural network inference.
        4. Calculate class probabilities via softmax.
        5. Evaluate confidence threshold and assign abstention flags if uncertain.
        6. Compute Grad-CAM activation map and side-by-side visualization.
        7. Assemble comprehensive result payload.

        Args:
            image_path: Absolute or relative path to the image file on disk.

        Returns:
            Dictionary containing prediction outcome, probabilities, Grad-CAM paths,
            quality report, model info, and mandatory medical research disclaimers.
        """
        logger.info("Initiating prediction pipeline for input file.")

        # Structure default response
        result: Dict[str, Any] = {
            "success": False,
            "prediction": "UNKNOWN",
            "scores": {cls_name: 0.0 for cls_name in self.class_names},
            "confidence": 0.0,
            "needs_review": True,
            "review_reason": None,
            "quality_checks": {},
            "gradcam_path": None,
            "original_image_path": None,
            "model_info": {
                "architecture": getattr(Config, "MODEL_NAME", "efficientnet_b0"),
                "device": str(self.device),
                "threshold": self.confidence_threshold,
            },
            "disclaimer": DISCLAIMER_TEXT,
        }

        # Step 1: Technical validation
        try:
            val_results = image_validator.run_all_checks(file_path=image_path)
            result["quality_checks"] = val_results
        except Exception as exc:
            logger.error("Validation error: %s", exc)
            result["review_reason"] = f"Validation subsystem exception: {exc}"
            return result

        if not val_results.get("is_valid", False):
            errors = val_results.get("errors", [])
            err_msg = "; ".join(errors) if errors else "Image failed technical validation."
            logger.warning("Scan rejected by validator: %s", err_msg)
            result["prediction"] = "INVALID_SCAN"
            result["review_reason"] = f"Image quality check failed: {err_msg}"
            return result

        # Ensure image is saved/copied to uploads directory if needed
        safe_uploads_dir = getattr(Config, "UPLOADS_DIR", "uploads")
        os.makedirs(safe_uploads_dir, exist_ok=True)
        filename = f"scan_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
        persisted_orig_path = os.path.join(safe_uploads_dir, filename)
        try:
            if os.path.abspath(image_path) != os.path.abspath(persisted_orig_path):
                shutil.copy2(image_path, persisted_orig_path)
            result["original_image_path"] = persisted_orig_path
        except Exception as exc:
            logger.warning("Could not persist image copy: %s. Using source path.", exc)
            result["original_image_path"] = image_path

        # Step 2: Retrieve model
        model = self.get_model()
        if model is None:
            logger.error("Inference aborted: Model is not loaded or unavailable.")
            result["prediction"] = "MODEL_UNAVAILABLE"
            result["review_reason"] = "Deep learning model checkpoint is not loaded."
            return result

        model.eval()
        model.to(self.device)

        # Step 3: Preprocess image
        try:
            input_tensor = preprocessing.preprocess_for_inference(image_path, device=self.device)
        except Exception as exc:
            logger.error("Preprocessing failed: %s", exc)
            result["prediction"] = "PREPROCESSING_ERROR"
            result["review_reason"] = f"Failed to transform image for model input: {exc}"
            return result

        # Step 4: Model inference & Softmax
        try:
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        except Exception as exc:
            logger.error("Model forward pass failed: %s", exc)
            result["prediction"] = "INFERENCE_ERROR"
            result["review_reason"] = f"Error during neural network forward pass: {exc}"
            return result

        # Map probabilities
        scores: Dict[str, float] = {}
        for idx, cls_name in enumerate(self.class_names):
            if idx < len(probs):
                scores[cls_name] = float(round(probs[idx], 4))
            else:
                scores[cls_name] = 0.0

        result["scores"] = scores
        top_idx = int(probs.argmax())
        top_class = self.class_names[top_idx] if top_idx < len(self.class_names) else "UNKNOWN"
        top_confidence = float(round(probs[top_idx], 4))

        result["prediction"] = top_class
        result["confidence"] = top_confidence

        # Step 5: Confidence thresholding and quality warning flags
        needs_review = False
        review_reasons = []

        if top_confidence < self.confidence_threshold:
            needs_review = True
            review_reasons.append(
                f"Confidence score ({top_confidence:.1%}) is below the required assurance threshold ({self.confidence_threshold:.1%})."
            )

        if val_results.get("warnings"):
            needs_review = True
            review_reasons.extend(val_results["warnings"])

        result["needs_review"] = needs_review
        result["review_reason"] = " | ".join(review_reasons) if review_reasons else None

        # Step 6: Grad-CAM Explainability
        try:
            # Enable gradients specifically for the single Grad-CAM evaluation pass
            gradcam_out_dir = os.path.join(getattr(Config, "REPORTS_DIR", "reports"), "gradcam")
            os.makedirs(gradcam_out_dir, exist_ok=True)
            cam_filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
            cam_path = os.path.join(gradcam_out_dir, cam_filename)

            _, _, _, saved_cam_path = gradcam_service.get_gradcam_visualization(
                model=model,
                input_tensor=input_tensor.clone().detach(),
                original_image=result["original_image_path"],
                target_class=top_idx,
                save_path=cam_path
            )
            result["gradcam_path"] = saved_cam_path
        except Exception as exc:
            logger.warning("Grad-CAM generation failed: %s", exc)
            result["gradcam_path"] = None

        # Step 7: Clinical Guidance, Radiological Explanation & Emergency Protocols
        clinical_guidance = {}
        if top_class == "PNEUMONIA":
            clinical_guidance = {
                "summary": "Radiographic analysis demonstrates patterns consistent with alveolar consolidation / infiltrative opacities.",
                "radiological_findings": [
                    "Increased radiodensity (whiteness/clouding) within the lung fields suggesting inflammatory exudate or alveolar fluid.",
                    "Loss of normal sharp broncho-vascular margins in affected lobes (silhouette sign).",
                    "Grad-CAM activation highlights focal or diffuse feature density where the deep learning network identified pathological textures."
                ],
                "immediate_precautions": [
                    "Strict Medical Consultation: Schedule an in-person clinical evaluation with a pulmonologist or physician promptly.",
                    "Do Not Self-Medicate: Bacterial and viral pneumonia require completely different therapeutic regimens (antibiotics vs. supportive/antiviral therapy).",
                    "Rest & Hydration: Maintain adequate fluid intake to thin respiratory secretions and prevent dehydration.",
                    "Isolation & Masking: Wear a protective mask around vulnerable individuals, young children, and elderly family members.",
                    "Monitor Vital Signs: Regularly measure peripheral oxygen saturation (SpO2), heart rate, and body temperature with a pulse oximeter."
                ],
                "emergency_red_flags": [
                    "Severe shortness of breath, rapid labored breathing (tachypnea), or inability to speak in full sentences.",
                    "Pulse oximeter reading dropping below 92% (or below 90% in individuals with chronic lung disease).",
                    "Persistent chest pain, especially when inhaling or coughing (pleuritic chest pain).",
                    "Bluish or grayish lips, nail beds, or skin (cyanosis / hypoxemia).",
                    "Confusion, severe dizziness, drowsiness, or altered mental status (especially in elderly patients).",
                    "Coughing up significant amounts of blood or rust-colored sputum."
                ],
                "emergency_action": "IF ANY RED FLAG SYMPTOMS ARE PRESENT, CALL EMERGENCY SERVICES (911 / 112) OR PROCEED IMMEDIATELY TO THE NEAREST HOSPITAL EMERGENCY DEPARTMENT."
            }
        else:
            clinical_guidance = {
                "summary": "Radiographic analysis demonstrates clear lung parenchyma without prominent focal consolidation or opacities.",
                "radiological_findings": [
                    "Preserved bilateral radiolucency throughout upper, middle, and lower lung zones.",
                    "Sharp, clear costophrenic and cardiophrenic angles without evident pleural blunting.",
                    "Expected anatomical heart silhouette and mediastinal contours without pathological widening."
                ],
                "immediate_precautions": [
                    "Correlate with Clinical Symptoms: An early-stage respiratory infection or atypical presentation may precede visible radiographic changes.",
                    "If experiencing persistent fever, deep cough, or wheezing, consult a doctor despite clear imaging.",
                    "Practice respiratory hygiene and monitor for developing symptoms."
                ],
                "emergency_red_flags": [
                    "Sudden onset of dyspnea (difficulty breathing) or sudden drop in oxygen levels.",
                    "Stabbing chest pain radiating to the neck or left shoulder.",
                    "Sudden fainting, confusion, or severe persistent high fever (>103\u00b0F / 39.4\u00b0C)."
                ],
                "emergency_action": "SEEK URGENT MEDICAL ATTENTION IF ACUTE SHORTNESS OF BREATH OR UNEXPLAINED CHEST DISCOMFORT OCCURS."
            }

        result["clinical_guidance"] = clinical_guidance
        result["success"] = True

        # Sanitized logging: log prediction class and confidence only (NO PHI)
        logger.info(
            "Inference completed successfully. Predicted: %s (Confidence: %.2f%%). Needs Review: %s",
            result["prediction"],
            result["confidence"] * 100,
            result["needs_review"]
        )

        return result
