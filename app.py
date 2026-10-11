"""
MedScan Assist - Flask Application Entry Point

AI-Powered Chest X-Ray Screening Research Prototype.

DISCLAIMER: This is a research prototype for educational purposes only.
It is NOT a clinically validated diagnostic system.
All medical decisions require evaluation by qualified healthcare professionals.
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, request, jsonify, render_template, send_file,
    send_from_directory, abort
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import Config
from models.model_loader import ModelLoader
from services.prediction_service import PredictionService
from services.image_validator import ImageValidator
from services.report_service import ReportService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_FILE_SIZE_BYTES
app.config['UPLOAD_FOLDER'] = Config.UPLOADS_DIR
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'medscan-dev-key-change-in-production')
CORS(app)

# Ensure directories exist
os.makedirs(Config.UPLOADS_DIR, exist_ok=True)
os.makedirs(os.path.join(Config.REPORTS_DIR, 'generated'), exist_ok=True)
os.makedirs(os.path.join(Config.UPLOADS_DIR, 'gradcam'), exist_ok=True)
os.makedirs(os.path.join('static', 'evaluation'), exist_ok=True)

# Initialize services
model_loader = ModelLoader()
prediction_service = None
report_service = ReportService()
image_validator = ImageValidator()


def get_prediction_service():
    """Lazy-load prediction service and load model checkpoint if available."""
    global prediction_service
    if not model_loader.is_ready():
        if os.path.exists(Config.BEST_MODEL_PATH):
            model_loader.load_model(Config.BEST_MODEL_PATH)
    if prediction_service is None:
        prediction_service = PredictionService(model_loader)
    return prediction_service


def cleanup_old_uploads(max_age_hours=1):
    """Remove uploaded files older than max_age_hours."""
    try:
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        upload_dir = Config.UPLOADS_DIR
        if os.path.exists(upload_dir):
            for f in os.listdir(upload_dir):
                filepath = os.path.join(upload_dir, f)
                if os.path.isfile(filepath) and f != '.gitkeep':
                    if os.path.getmtime(filepath) < cutoff:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old upload: {f}")
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")


# ============================================================
# Page Routes
# ============================================================

@app.route('/')
def index():
    """Landing page."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')


# ============================================================
# API Routes
# ============================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Return application status, model availability, and device info."""
    # Load the checkpoint on the first health check so the dashboard reports
    # actual model readiness instead of marking a healthy, lazily initialized
    # backend as offline.
    if not model_loader.is_ready() and os.path.isfile(Config.BEST_MODEL_PATH):
        get_prediction_service()

    model_info = model_loader.get_model_info()
    return jsonify({
        'status': 'online',
        'model_loaded': model_info['is_loaded'],
        'model_info': model_info,
        'device': str(Config.DEVICE),
        'timestamp': datetime.now().isoformat(),
        'version': Config.VERSION
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze an uploaded chest X-ray image.
    
    Validates the image, runs quality checks, performs model inference,
    generates Grad-CAM visualization, and returns structured results.
    """
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided. Please upload a chest X-ray image.',
                'error_code': 'NO_FILE'
            }), 400

        file = request.files['image']
        if file.filename == '' or file.filename is None:
            return jsonify({
                'success': False,
                'error': 'No file selected.',
                'error_code': 'EMPTY_FILENAME'
            }), 400

        # Generate safe filename
        original_filename = secure_filename(file.filename)
        unique_id = uuid.uuid4().hex[:8]
        safe_filename = f"{unique_id}_{original_filename}"
        filepath = os.path.join(Config.UPLOADS_DIR, safe_filename)

        # Save uploaded file
        file.save(filepath)
        logger.info(f"Image uploaded: {safe_filename}")

        # Run prediction pipeline
        service = get_prediction_service()
        result = service.predict(filepath)

        if not result['success']:
            # Clean up on failure
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify(result), 400

        # Convert paths to URLs for frontend
        if result.get('gradcam_path') and os.path.exists(result['gradcam_path']):
            result['gradcam_url'] = f"/uploads/{os.path.basename(result['gradcam_path'])}"
        if result.get('original_image_path') and os.path.exists(result['original_image_path']):
            result['original_image_url'] = f"/uploads/{os.path.basename(result['original_image_path'])}"
        
        # Remove internal file paths from response
        result.pop('gradcam_path', None)
        result.pop('original_image_path', None)
        result.pop('upload_path', None)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'An internal error occurred during analysis. Please try again.',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Return saved evaluation metrics if available."""
    try:
        metrics_path = os.path.join(Config.REPORTS_DIR, 'evaluation', 'test_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Check for evaluation plots
            plots = {}
            plot_files = [
                'confusion_matrix.png', 'roc_curve.png', 
                'precision_recall_curve.png', 'training_curves.png'
            ]
            eval_dir = os.path.join(Config.REPORTS_DIR, 'evaluation')
            for pf in plot_files:
                plot_path = os.path.join(eval_dir, pf)
                if os.path.exists(plot_path):
                    # Copy to static for serving
                    static_eval_dir = os.path.join('static', 'evaluation')
                    os.makedirs(static_eval_dir, exist_ok=True)
                    import shutil
                    dest = os.path.join(static_eval_dir, pf)
                    if not os.path.exists(dest) or \
                       os.path.getmtime(plot_path) > os.path.getmtime(dest):
                        shutil.copy2(plot_path, dest)
                    plots[pf.replace('.png', '')] = f"/static/evaluation/{pf}"
            
            # Normalize metric fields for both direct and nested consumers
            response_payload = {
                'available': True,
                'metrics': metrics,
                'plots': plots
            }
            # Flatten standard evaluation keys if inside overall_metrics
            overall = metrics.get('overall_metrics', {})
            for k in ['accuracy', 'precision', 'recall', 'recall_sensitivity', 'sensitivity', 'specificity', 'f1_score', 'roc_auc']:
                if k in overall:
                    response_payload[k] = overall[k]
                elif k in metrics:
                    response_payload[k] = metrics[k]
            
            # Confusion matrix normalization
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                if 'raw_matrix' in cm:
                    response_payload['confusion_matrix'] = cm['raw_matrix']
                else:
                    response_payload['confusion_matrix'] = cm

            return jsonify(response_payload)
        else:
            return jsonify({
                'available': False,
                'message': 'Evaluation not yet available. Run the evaluation script first: python training/evaluate.py'
            })
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return jsonify({
            'available': False,
            'error': 'Failed to load evaluation metrics.'
        }), 500


@app.route('/api/sample-images', methods=['GET'])
def get_sample_images():
    """Return the two curated clinical radiograph examples."""
    try:
        sample_dir = os.path.join('static', 'images', 'samples')
        curated_samples = [
            ('normal_chest_radiograph.png', 'X-ray Normal'),
            ('pneumonia_chest_radiograph.png', 'X-ray Pneumonia'),
        ]
        samples = [
            {
                'filename': filename,
                'url': f"/static/images/samples/{filename}",
                'label': label,
            }
            for filename, label in curated_samples
            if os.path.isfile(os.path.join(sample_dir, filename))
        ]
        return jsonify({
            'samples': samples,
            'note': 'Curated clinical examples for research demonstration only.'
        })
    except Exception as e:
        logger.error(f"Sample images error: {str(e)}")
        return jsonify({'samples': [], 'error': str(e)}), 500


@app.route('/api/report', methods=['POST'])
def generate_report():
    """Generate a downloadable screening research report."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No result data provided for report generation.'
            }), 400

        # Generate report
        report_path = report_service.generate_report(data)
        
        if report_path and os.path.exists(report_path):
            return send_file(
                report_path,
                as_attachment=True,
                download_name=f"medscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mimetype='text/html'
            )
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate report.'
            }), 500

    except Exception as e:
        logger.error(f"Report generation error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Report generation failed.'
        }), 500


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded/generated files (images, heatmaps)."""
    safe_name = secure_filename(filename)
    upload_dir = os.path.abspath(Config.UPLOADS_DIR)
    gradcam_dir = os.path.abspath(os.path.join(Config.REPORTS_DIR, 'gradcam'))
    
    file_path = os.path.join(upload_dir, safe_name)
    if os.path.exists(file_path) and os.path.abspath(file_path).startswith(upload_dir):
        return send_from_directory(upload_dir, safe_name)
        
    gradcam_file_path = os.path.join(gradcam_dir, safe_name)
    if os.path.exists(gradcam_file_path) and os.path.abspath(gradcam_file_path).startswith(gradcam_dir):
        return send_from_directory(gradcam_dir, safe_name)

    abort(404)


# ============================================================
# Error Handlers
# ============================================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size is {Config.MAX_FILE_SIZE_BYTES // (1024*1024)}MB.',
        'error_code': 'FILE_TOO_LARGE'
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Resource not found.',
        'error_code': 'NOT_FOUND'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error.',
        'error_code': 'INTERNAL_ERROR'
    }), 500


# ============================================================
# Startup
# ============================================================

if __name__ == '__main__':
    # Cleanup old uploads on startup
    cleanup_old_uploads()
    
    # Log startup info
    logger.info("=" * 60)
    logger.info("MedScan Assist - AI Chest X-Ray Screening Research Prototype")
    logger.info("=" * 60)
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Checkpoint path: {Config.BEST_MODEL_PATH}")
    
    model_info = model_loader.get_model_info()
    if model_info['is_loaded']:
        logger.info("Model loaded successfully!")
    else:
        logger.warning("Model NOT loaded. Train the model first:")
        logger.warning("  python training/train.py")
        logger.warning("  python training/evaluate.py")
    
    logger.info("")
    logger.info("Starting server at http://127.0.0.1:5000")
    logger.info("DISCLAIMER: Research prototype only. Not for clinical use.")
    logger.info("=" * 60)
    
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    )

