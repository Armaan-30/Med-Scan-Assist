"""Report Generation Service for MedScan Assist.

Generates self-contained, downloadable HTML diagnostic summary reports
incorporating model predictions, uncertainty scores, technical validation results,
embedded base64 Grad-CAM visualizations, and clinical research disclaimers.

NO Protected Health Information (PHI) is included or processed.
"""

import base64
from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, Optional
import uuid
from jinja2 import Template

from config.config import Config

logger = logging.getLogger("medscan.report_service")

REPORT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MedScan Assist - Analysis Report #{{ report_id }}</title>
  <style>
    :root {
      --primary: #1e3a8a;
      --primary-light: #3b82f6;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --danger: #dc2626;
      --danger-bg: #fef2f2;
      --warning: #d97706;
      --warning-bg: #fffbeb;
      --success: #16a34a;
      --success-bg: #f0fdf4;
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 30px 20px;
    }
    .container {
      max-width: 960px;
      margin: 0 auto;
      background: var(--card-bg);
      border-radius: 12px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
      overflow: hidden;
      border: 1px solid var(--border);
    }
    /* Clinical Research Banner */
    .banner {
      background: #7f1d1d;
      color: #fef2f2;
      text-align: center;
      padding: 12px 16px;
      font-weight: 700;
      font-size: 0.875rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    header {
      padding: 28px 36px 20px;
      border-bottom: 2px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 16px;
    }
    .brand h1 {
      font-size: 1.75rem;
      font-weight: 800;
      color: var(--primary);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .brand p {
      color: var(--text-muted);
      font-size: 0.875rem;
      margin-top: 4px;
    }
    .meta-box {
      text-align: right;
      font-size: 0.8125rem;
      color: var(--text-muted);
    }
    .meta-box strong {
      color: var(--text);
    }
    .content {
      padding: 32px 36px;
    }
    .section-title {
      font-size: 1.125rem;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 16px;
      border-bottom: 2px solid #f1f5f9;
      padding-bottom: 8px;
    }
    /* Prediction Summary Card */
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 20px;
      margin-bottom: 32px;
    }
    .card {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      background: #fdfdfd;
    }
    .prediction-badge {
      display: inline-block;
      font-size: 1.375rem;
      font-weight: 800;
      padding: 6px 16px;
      border-radius: 6px;
      margin: 10px 0;
    }
    .badge-normal {
      background-color: var(--success-bg);
      color: var(--success);
      border: 1px solid #bbf7d0;
    }
    .badge-pneumonia {
      background-color: var(--danger-bg);
      color: var(--danger);
      border: 1px solid #fecaca;
    }
    .badge-uncertain {
      background-color: var(--warning-bg);
      color: var(--warning);
      border: 1px solid #fde68a;
    }
    .review-flag {
      margin-top: 10px;
      font-size: 0.8125rem;
      padding: 8px 12px;
      border-radius: 6px;
      background: var(--warning-bg);
      color: #92400e;
      border: 1px solid #fde68a;
    }
    /* Scores Table & Progress Bars */
    .score-row {
      margin-bottom: 12px;
    }
    .score-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.875rem;
      font-weight: 600;
      margin-bottom: 4px;
    }
    .progress-bar-bg {
      height: 10px;
      background: #e2e8f0;
      border-radius: 5px;
      overflow: hidden;
    }
    .progress-bar-fill {
      height: 100%;
      border-radius: 5px;
    }
    .fill-pneumonia {
      background: linear-gradient(90deg, #f87171, #dc2626);
    }
    .fill-normal {
      background: linear-gradient(90deg, #4ade80, #16a34a);
    }
    /* Image Visualizations Grid */
    .visualizations {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 24px;
      margin-bottom: 32px;
    }
    .img-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background: #000;
      text-align: center;
    }
    .img-card-header {
      background: #1e293b;
      color: #f8fafc;
      font-size: 0.875rem;
      font-weight: 600;
      padding: 8px 12px;
      text-align: left;
    }
    .img-card img {
      max-width: 100%;
      height: auto;
      display: block;
      margin: 0 auto;
    }
    .img-card-footer {
      background: #0f172a;
      color: #94a3b8;
      font-size: 0.75rem;
      padding: 6px 12px;
      text-align: left;
    }
    /* Quality Checks Table */
    .quality-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
      margin-bottom: 32px;
    }
    .quality-table th, .quality-table td {
      border: 1px solid var(--border);
      padding: 10px 14px;
      text-align: left;
    }
    .quality-table th {
      background: #f8fafc;
      font-weight: 600;
    }
    .status-pass {
      color: var(--success);
      font-weight: 600;
    }
    .status-warning {
      color: var(--warning);
      font-weight: 600;
    }
    /* Disclaimer Footer */
    footer {
      background: #f8fafc;
      padding: 24px 36px;
      border-top: 1px solid var(--border);
      font-size: 0.8125rem;
      color: var(--text-muted);
      line-height: 1.6;
    }
    .disclaimer-title {
      font-weight: 700;
      color: #991b1b;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    @media print {
      body {
        padding: 0;
        background: #fff;
      }
      .container {
        border: none;
        box-shadow: none;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="banner">
      ⚠ Research Prototype Only — Not Cleared for Standalone Clinical Diagnostic Use
    </div>

    <header>
      <div class="brand">
        <h1>MedScan Assist</h1>
        <p>Deep Learning Pulmonary Radiograph Analysis & Saliency Report</p>
      </div>
      <div class="meta-box">
        <div><strong>Report ID:</strong> {{ report_id }}</div>
        <div><strong>Generated (UTC):</strong> {{ timestamp }}</div>
        <div><strong>Architecture:</strong> {{ model_info.architecture }}</div>
        <div><strong>Version:</strong> {{ model_version }}</div>
        <div><strong>Device:</strong> {{ model_info.device }}</div>
      </div>
    </header>

    <div class="content">
      <div class="section-title">Diagnostic Classification Overview</div>
      <div class="summary-grid">
        <!-- Classification Card -->
        <div class="card">
          <div style="font-size: 0.875rem; color: var(--text-muted); font-weight: 600;">PRIMARY ASSESSMENT</div>
          <div>
            {% if prediction == 'NORMAL' %}
              <span class="prediction-badge badge-normal">NORMAL (NO PNEUMONIA)</span>
            {% elif prediction == 'PNEUMONIA' %}
              <span class="prediction-badge badge-pneumonia">PNEUMONIA SUSPECTED</span>
            {% else %}
              <span class="prediction-badge badge-uncertain">{{ prediction }}</span>
            {% endif %}
          </div>
          <div style="font-size: 0.9375rem; margin-top: 4px;">
            Confidence: <strong>{{ "%.2f"|format(confidence * 100) }}%</strong>
          </div>

          {% if needs_review %}
            <div class="review-flag">
              ⚠ <strong>Clinical Review Flagged:</strong> {{ review_reason or "Confidence threshold or quality checks require verification." }}
            </div>
          {% endif %}
        </div>

        <!-- Class Probabilities -->
        <div class="card">
          <div style="font-size: 0.875rem; color: var(--text-muted); font-weight: 600; margin-bottom: 12px;">SOFTMAX CLASS PROBABILITIES</div>
          {% for class_name, score in scores.items() %}
            <div class="score-row">
              <div class="score-label">
                <span>{{ class_name }}</span>
                <span>{{ "%.2f"|format(score * 100) }}%</span>
              </div>
              <div class="progress-bar-bg">
                <div class="progress-bar-fill {% if class_name == 'PNEUMONIA' %}fill-pneumonia{% else %}fill-normal{% endif %}"
                     style="width: {{ score * 100 }}%;"></div>
              </div>
            </div>
          {% endfor %}
          <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">
            Assurance Threshold: {{ "%.0f"|format(model_info.threshold * 100) }}%
          </div>
        </div>
      </div>

      <!-- Saliency Visualizations -->
      <div class="section-title">Visual Saliency & Explainability (Grad-CAM)</div>
      <div class="visualizations">
        {% if original_image_b64 %}
          <div class="img-card">
            <div class="img-card-header">Input Chest Radiograph</div>
            <img src="{{ original_image_b64 }}" alt="Input Chest Radiograph">
            <div class="img-card-footer">Original image without model alterations</div>
          </div>
        {% endif %}

        {% if gradcam_image_b64 %}
          <div class="img-card">
            <div class="img-card-header">Grad-CAM Feature Saliency</div>
            <img src="{{ gradcam_image_b64 }}" alt="Grad-CAM Activation Overlay">
            <div class="img-card-footer">Warm colors highlight regions influencing model output</div>
          </div>
        {% endif %}
      </div>

      <!-- Technical Quality Checks -->
      <div class="section-title">Image Quality & Technical Pre-Checks</div>
      <table class="quality-table">
        <thead>
          <tr>
            <th>Validation Check</th>
            <th>Measured Value</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>File Format & Magic Bytes</td>
            <td>{{ quality_checks.quality_info.mime_type or "N/A" }}</td>
            <td><span class="status-pass">PASS</span></td>
          </tr>
          <tr>
            <td>File Size</td>
            <td>{{ quality_checks.quality_info.size_mb or "0.0" }} MB</td>
            <td><span class="status-pass">PASS</span></td>
          </tr>
          <tr>
            <td>Image Resolution</td>
            <td>
              {% if quality_checks.quality_info.dimensions %}
                {{ quality_checks.quality_info.dimensions.width }} x {{ quality_checks.quality_info.dimensions.height }} px
              {% else %}
                N/A
              {% endif %}
            </td>
            <td><span class="status-pass">PASS</span></td>
          </tr>
          <tr>
            <td>Mean Luminance Brightness</td>
            <td>{{ quality_checks.quality_info.brightness_mean or "N/A" }} / 255</td>
            <td>
              {% if quality_checks.warnings and "dark" in quality_checks.warnings|string or "bright" in quality_checks.warnings|string %}
                <span class="status-warning">FLAGGED</span>
              {% else %}
                <span class="status-pass">PASS</span>
              {% endif %}
            </td>
          </tr>
          <tr>
            <td>Sharpness (Laplacian Variance)</td>
            <td>{{ quality_checks.quality_info.blur_laplacian_var or "N/A" }}</td>
            <td>
              {% if quality_checks.warnings and "blurry" in quality_checks.warnings|string %}
                <span class="status-warning">LOW CONTRAST/BLUR</span>
              {% else %}
                <span class="status-pass">PASS</span>
              {% endif %}
            </td>
          </tr>
        </tbody>
      </table>
    <!-- Clinical Guidance, Precautions & Emergency Red Flags -->
    {% if clinical_guidance %}
    <div class="section-title">Clinical Findings &amp; Health Guidance</div>
    <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 30px;">
      <div style="font-weight: 700; color: var(--primary); margin-bottom: 8px;">
        Radiological Impression Summary:
      </div>
      <div style="font-size: 0.9375rem; color: #334155; margin-bottom: 16px;">
        {{ clinical_guidance.summary }}
      </div>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px;">
        <!-- Observed Findings -->
        <div style="background: white; border: 1px solid var(--border); border-radius: 6px; padding: 14px;">
          <div style="font-weight: 700; color: var(--primary); font-size: 0.875rem; margin-bottom: 8px;">
            Radiological Patterns:
          </div>
          <ul style="font-size: 0.8125rem; color: #475569; padding-left: 18px; margin: 0;">
            {% for item in clinical_guidance.radiological_findings %}
              <li style="margin-bottom: 4px;">{{ item }}</li>
            {% endfor %}
          </ul>
        </div>

        <!-- Precautions -->
        <div style="background: white; border: 1px solid var(--border); border-radius: 6px; padding: 14px;">
          <div style="font-weight: 700; color: #0284c7; font-size: 0.875rem; margin-bottom: 8px;">
            Immediate Precautions:
          </div>
          <ul style="font-size: 0.8125rem; color: #475569; padding-left: 18px; margin: 0;">
            {% for item in clinical_guidance.immediate_precautions %}
              <li style="margin-bottom: 4px;">{{ item }}</li>
            {% endfor %}
          </ul>
        </div>

        <!-- Emergency Red Flags -->
        <div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: 6px; padding: 14px;">
          <div style="font-weight: 700; color: var(--danger); font-size: 0.875rem; margin-bottom: 8px;">
            Emergency Red Flags:
          </div>
          <ul style="font-size: 0.8125rem; color: #9f1239; padding-left: 18px; margin: 0;">
            {% for item in clinical_guidance.emergency_red_flags %}
              <li style="margin-bottom: 4px;">{{ item }}</li>
            {% endfor %}
          </ul>
        </div>
      </div>

      {% if clinical_guidance.emergency_action %}
      <div style="margin-top: 14px; padding: 10px 14px; background: #fee2e2; border-left: 4px solid var(--danger); border-radius: 4px; font-size: 0.8125rem; font-weight: 700; color: #991b1b;">
        EMERGENCY PROTOCOL: {{ clinical_guidance.emergency_action }}
      </div>
      {% endif %}
    </div>
    {% endif %}

    <footer>
      <div class="disclaimer-title">Regulatory & Research Disclosure</div>
      <p>{{ disclaimer }}</p>
      <p style="margin-top: 8px;">
        <strong>Privacy Assurance:</strong> This document contains zero Protected Health Information (PHI) or patient identifiers.
        Images are processed anonymously for research validation and model evaluation purposes.
      </p>
    </footer>
  </div>
</body>
</html>
"""


def image_file_to_base64_data_uri(image_path: str) -> Optional[str]:
    """Convert an image file on disk to an HTML-embeddable base64 data URI."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/png"
        if ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext == ".bmp":
            mime_type = "image/bmp"

        with open(image_path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
    except Exception as exc:
        logger.warning("Failed to encode image %s to base64: %s", image_path, exc)
        return None


def generate_html_report(
    prediction_result: Dict[str, Any],
    output_dir: Optional[str] = None
) -> str:
    """Generate a self-contained, downloadable HTML medical diagnostic report.

    Args:
        prediction_result: Dictionary returned from PredictionService.predict().
        output_dir: Destination folder. Defaults to Config.REPORTS_DIR / 'generated'.

    Returns:
        Absolute file path to the generated HTML report file.
    """
    report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    model_version = getattr(Config, "VERSION", "1.0.0")

    # Embed original and Grad-CAM images as base64
    orig_path = prediction_result.get("original_image_path")
    grad_path = prediction_result.get("gradcam_path")

    orig_b64 = image_file_to_base64_data_uri(orig_path) if orig_path else None
    grad_b64 = image_file_to_base64_data_uri(grad_path) if grad_path else None

    template = Template(REPORT_HTML_TEMPLATE)
    html_content = template.render(
        report_id=report_id,
        timestamp=timestamp,
        model_version=model_version,
        prediction=prediction_result.get("prediction", "UNKNOWN"),
        confidence=prediction_result.get("confidence", 0.0),
        scores=prediction_result.get("scores", {}),
        needs_review=prediction_result.get("needs_review", False),
        review_reason=prediction_result.get("review_reason"),
        quality_checks=prediction_result.get("quality_checks", {}),
        model_info=prediction_result.get("model_info", {
            "architecture": getattr(Config, "MODEL_NAME", "efficientnet_b0"),
            "device": "cpu",
            "threshold": 0.70
        }),
        disclaimer=prediction_result.get("disclaimer", Config.RESEARCH_DISCLAIMER),
        original_image_b64=orig_b64,
        gradcam_image_b64=grad_b64,
    )

    if output_dir is None:
        output_dir = os.path.join(getattr(Config, "REPORTS_DIR", "reports"), "generated")

    os.makedirs(output_dir, exist_ok=True)
    report_filename = f"medscan_report_{report_id}.html"
    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Generated HTML clinical report at: %s", report_path)
    return os.path.abspath(report_path)


class ReportService:
    """Service wrapper for generating downloadable screening reports."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = output_dir

    def generate_report(self, prediction_result: Dict[str, Any]) -> str:
        """Generate and save report from prediction dictionary."""
        return generate_html_report(prediction_result, output_dir=self.output_dir)

