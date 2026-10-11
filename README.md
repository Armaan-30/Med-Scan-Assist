# MedScan Assist — AI-Powered Chest X-Ray Screening Research Prototype

> **IMPORTANT CLINICAL RESEARCH DISCLAIMER**  
> **RESEARCH USE ONLY:** MedScan Assist is an artificial intelligence research prototype developed strictly for educational and academic exploration. It is **NOT** a certified, approved, or clinically validated diagnostic medical device. It must **NEVER** be used as a standalone diagnostic tool or substitute for professional clinical judgment. All medical decisions, radiographic reviews, and diagnoses require evaluation by licensed physicians and qualified healthcare professionals.

---

## 1. Project Overview

**MedScan Assist** is an end-to-end medical deep-learning application designed to analyze frontal chest radiographs (CXR) and evaluate the likelihood of **Normal** lung anatomy versus **Pneumonia**.

Key components include:
- **Deep Learning Transfer Learning Backbone:** EfficientNet-B0 fine-tuned with custom binary classification head.
- **Strict Preprocessing & Guardrails:** Consistent deterministic evaluation pipelines with ImageNet normalization, aspect-ratio preserved scaling, and RGB standardization.
- **Image Quality & Technical Auditing:** Automated detection of unreadable files, dimension bounds, magic byte mismatch, severe underexposure/overexposure, and blurriness using Laplacian variance.
- **Prediction Uncertainty & Abstention:** Clinical assurance thresholding where predictions with softmax confidence below threshold (default 70%) trigger explicit `"Needs human review"` triage alerts.
- **Explainable AI (Grad-CAM):** Dynamic Gradient-weighted Class Activation Mapping computed directly on the final convolutional feature layer (`model.features[-1]`), producing side-by-side interpretability heatmaps.
- **Quantitative Evaluation:** Real metrics tracking accuracy, sensitivity, specificity, precision, F1-score, ROC-AUC, confusion matrices, and training curves.
- **Self-Contained Reporting:** Automated downloadable HTML screening reports embedding Base64 diagnostic images and technical audit logs with zero Protected Health Information (PHI) exposure.

---

## 2. System Architecture & Technology Stack

```
MedScan Assist/
├── app.py                      # Flask backend API & static server
├── requirements.txt            # Project dependencies
├── .env.example                # Environment variables configuration template
├── .gitignore                  # Git ignore rules
│
├── config/
│   └── config.py               # Central configuration, hardware detection & constants
│
├── data/
│   ├── README.md               # Dataset source instructions and pediatric limitations
│   ├── chest_xray/             # Kaggle Kermany dataset root (train/val/test)
│   └── processed/              # Stratified split partitions and data manifest
│
├── models/
│   ├── model_architecture.py   # EfficientNet-B0 PyTorch architecture & weights saving
│   ├── model_loader.py         # Thread-safe cached singleton model loading service
│   └── checkpoints/            # Saved model weights (.pth)
│
├── training/
│   ├── inspect_dataset.py      # Dataset validation, class imbalance & duplicate checker
│   ├── prepare_dataset.py      # Stratified 15% validation re-split & DataLoader factory
│   ├── train.py                # 2-phase transfer learning pipeline with early stopping
│   ├── evaluate.py             # Strict held-out test evaluation & curve generator
│   └── inference.py            # Standalone batch & single CLI inference script
│
├── services/
│   ├── image_validator.py      # Technical validation (magic bytes, size, blur, brightness)
│   ├── preprocessing.py        # Clinically defensible transforms & tensor pipelines
│   ├── prediction_service.py   # Full orchestration (validation -> inference -> abstention -> Grad-CAM)
│   ├── gradcam_service.py      # PyTorch Grad-CAM computation & side-by-side visualization
│   └── report_service.py       # Base64-embedded self-contained HTML clinical report generator
│
├── reports/
│   ├── evaluation/             # Test metrics JSON, confusion matrix & ROC curve plots
│   ├── generated/              # Saved downloadable screening reports
│   └── gradcam/                # Generated Grad-CAM saliency maps
│
├── static/
│   ├── css/style.css           # Custom medical dashboard stylesheet (Navy/Teal theme)
│   ├── js/app.js               # Reactive frontend client handling API calls & UI states
│   ├── images/samples/         # Demonstration X-ray radiographs
│   └── evaluation/             # Cached static plots for frontend metrics dashboard
│
├── templates/
│   ├── index.html              # Landing presentation page with clinical disclaimer
│   ├── dashboard.html          # Interactive screening dashboard & metrics viewer
│   └── about.html              # Research scope, methodology & limitation disclosures
│
└── tests/
    └── test_all.py             # Pytest automated test suite (11 unit/integration tests)
```

---

## 3. Dataset Setup (Kaggle Chest X-Ray Images)

The primary dataset used for training is the **Chest X-Ray Images (Pneumonia)** dataset by Kermany et al.:
- **Kaggle Source:** [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Dataset Size:** ~5,863 pediatric chest radiographs (JPEG).

### Dataset Setup Instructions:
1. Download the archive from Kaggle (`chest-xray-pneumonia.zip`).
2. Extract the folder into `data/chest_xray/`:
   ```
   data/chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── val/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── test/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

> **Dataset Limitations:**
> 1. **Pediatric Cohort:** All images originate from pediatric patients (ages 1–5). Model performance cannot be assumed to generalize to adult clinical populations.
> 2. **Single Center:** Sourced from Guangzhou Women and Children's Medical Center. Inter-scanner variability across other hospital systems was not evaluated.
> 3. **Validation Re-splitting:** The standard Kaggle `val` folder contains only 16 images. Our `training/prepare_dataset.py` automatically generates a stratified 15% validation split from `train/` to ensure statistically valid early stopping without touching the test set.

---

## 4. Windows Setup & Run Instructions (PowerShell)

Follow these exact steps from the project root (`MedScan Assist`):

### Step 1: Verify Python Environment
Ensure Python 3.10+ (or 3.11/3.14) is installed:
```powershell
python --version
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install PyTorch and Dependencies
For standard CPU execution:
```powershell
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
*(For NVIDIA CUDA GPU acceleration, install PyTorch with CUDA matching your driver, e.g., `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`)*.

### Step 4: Inspect Dataset (Optional / When Dataset is Downloaded)
Inspect class distribution, detect corrupted images, and check duplicate checksums:
```powershell
python training/inspect_dataset.py
```

### Step 5: Prepare Stratified Splits
Perform stratified validation re-splitting (preserving test data):
```powershell
python training/prepare_dataset.py
```

### Step 6: Train the Model
Train the 2-phase transfer learning model with class weights and early stopping:
```powershell
python training/train.py --epochs 15 --batch-size 32 --lr 0.001
```

### Step 7: Evaluate on Held-out Test Set
Evaluate the trained checkpoint and generate confusion matrix and ROC curves:
```powershell
python training/evaluate.py
```

### Step 8: Run Automated Tests
Verify all 11 unit and integration tests pass:
```powershell
pytest tests/test_all.py -v
```

### Step 9: Launch the Web Application
Start the Flask server:
```powershell
python app.py
```

### Step 10: Open in Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- Visit `/` for the research prototype landing page.
- Visit `/dashboard` to analyze chest X-rays, inspect Grad-CAM overlays, review metrics, and download reports.

---

## 5. API Endpoints Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /api/health` | GET | System status, model loading state, and compute device (CPU/CUDA). |
| `POST /api/analyze` | POST | Upload an image file (`multipart/form-data`) for technical audit, classification, abstention rule check, and Grad-CAM generation. |
| `GET /api/metrics` | GET | Quantitative test performance metrics, confusion matrix, ROC-AUC, and training curves. |
| `GET /api/sample-images` | GET | Demonstration radiographs available for testing without manual file uploads. |
| `POST /api/report` | POST | Generate and download a self-contained HTML research screening report with Base64 embedded saliency maps. |
| `GET /uploads/<file>` | GET | Securely serve processed input radiographs and Grad-CAM visualizations. |

---

## 6. Uncertainty, Abstention & Technical Quality Policies

- **Abstention Policy:** Softmax outputs are uncalibrated network confidence scores and do not directly denote absolute clinical probability of disease. If the top-class confidence is below `70%` (`CONFIDENCE_THRESHOLD`), the system raises a mandatory `"Needs Human Review"` banner.
- **Technical Quality Checks:**
  - File extension & MIME magic-byte authentication.
  - Image decoding & resolution boundaries (64px to 4096px).
  - Brightness mean evaluation (flags underexposed `<20` or overexposed `>235` radiographs).
  - Blur estimation using Laplacian variance (`<80` variance flags potential motion blur).
- **Grad-CAM Limitations:** Grad-CAM visualizes spatial gradients indicating which anatomical lung zones influenced the model's feature representations. It **does not** prove clinical pathology or replace radiologist review.
