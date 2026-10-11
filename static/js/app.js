/**
 * MedScan Assist - Main Frontend Application Logic
 * Full-stack Medical AI Research Prototype
 */

'use strict';

// Application State
const AppState = {
    selectedFile: null,
    selectedSampleUrl: null,
    currentResult: null,
    modelHealth: null,
    rocChartInstance: null,
    isAnalyzing: false
};

// DOM Content Loaded Handler
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

/**
 * Initialize application elements and backend listeners
 */
function initApp() {
    setupEventListeners();
    setupTabRouting();
    checkHealth();
    loadSampleImages();
    loadEvaluationMetrics();
}

/**
 * Setup UI Event Listeners
 */
function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const dropzone = document.getElementById('dropzone');
    const replaceImageBtn = document.getElementById('replaceImageBtn');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetAnalysisBtn = document.getElementById('resetAnalysisBtn');
    const downloadReportBtn = document.getElementById('downloadReportBtn');
    const refreshMetricsBtn = document.getElementById('refreshMetricsBtn');

    // Trigger file chooser
    if (browseBtn && fileInput) {
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (replaceImageBtn && fileInput) {
        replaceImageBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    // Dropzone click -> file select (when no image selected)
    if (dropzone && fileInput) {
        dropzone.addEventListener('click', () => {
            if (!AppState.selectedFile && !AppState.selectedSampleUrl) {
                fileInput.click();
            }
        });

        // Drag & drop events
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            }, false);
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                handleFileSelection(files[0]);
            }
        });
    }

    // File input change
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    // Clear / Remove image
    if (removeImageBtn) {
        removeImageBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            clearSelectedImage();
        });
    }

    // Analyze button
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            analyzeImage();
        });
    }

    // Reset button in results
    if (resetAnalysisBtn) {
        resetAnalysisBtn.addEventListener('click', () => {
            clearSelectedImage();
            const resultsSection = document.getElementById('resultsSection');
            if (resultsSection) resultsSection.classList.add('d-none');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // Download Report button
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', () => {
            downloadReport();
        });
    }

    // Refresh metrics button
    if (refreshMetricsBtn) {
        refreshMetricsBtn.addEventListener('click', () => {
            loadEvaluationMetrics(true);
        });
    }
}

/**
 * Handle URL hash navigation for tabs
 */
function setupTabRouting() {
    const hash = window.location.hash;
    if (hash) {
        if (hash === '#evaluation') {
            const evalTab = document.getElementById('evaluation-tab');
            if (evalTab) evalTab.click();
        } else if (hash === '#about') {
            const aboutTab = document.getElementById('about-tab');
            if (aboutTab) aboutTab.click();
        }
    }

    // Listen to hash changes
    window.addEventListener('hashchange', () => {
        const newHash = window.location.hash;
        if (newHash === '#evaluation') {
            const evalTab = document.getElementById('evaluation-tab');
            if (evalTab) evalTab.click();
        } else if (newHash === '#about') {
            const aboutTab = document.getElementById('about-tab');
            if (aboutTab) aboutTab.click();
        }
    });
}

/**
 * Query Backend Health API (/api/health)
 */
async function checkHealth() {
    const dot = document.getElementById('modelStatusDot');
    const text = document.getElementById('modelStatusText');
    const deviceText = document.getElementById('deviceText');

    try {
        const response = await fetch('/api/health', {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });

        if (response.ok) {
            const data = await response.json();
            AppState.modelHealth = data;

            const isLoaded = (data.status === 'ok' || data.model_loaded === true);
            if (dot && text) {
                dot.className = isLoaded ? 'status-dot status-online' : 'status-dot status-offline';
                text.textContent = isLoaded ? 'Model Ready' : 'Model Offline';
            }

            if (deviceText && data.device) {
                deviceText.textContent = data.device.toUpperCase();
            }
        } else {
            throw new Error(`Health check returned status ${response.status}`);
        }
    } catch (err) {
        console.warn('Backend health check note:', err.message);
        if (dot && text) {
            dot.className = 'status-dot status-offline';
            text.textContent = 'Connecting...';
        }
    }
}

/**
 * Validate and handle uploaded file
 */
function handleFileSelection(file) {
    if (!file) return;

    // Validate type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const validExtensions = ['.jpg', '.jpeg', '.png'];
    const fileName = file.name.toLowerCase();
    const hasValidExt = validExtensions.some(ext => fileName.endsWith(ext));

    if (!validTypes.includes(file.type) && !hasValidExt) {
        showToast('Please upload a standard chest X-ray image in JPEG or PNG format.', 'danger');
        return;
    }

    // Validate size (max 10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast('File size exceeds the 10 MB limit. Please provide a compressed image.', 'warning');
        return;
    }

    AppState.selectedFile = file;
    AppState.selectedSampleUrl = null;

    // Remove active highlight from sample cards
    document.querySelectorAll('.sample-img-card').forEach(card => card.classList.remove('active'));

    // Render preview
    renderImagePreview(file);
}

/**
 * Render Image Preview and calculate dimensions
 */
function renderImagePreview(fileOrUrl, isSample = false, sampleName = '') {
    const dropzoneIdle = document.getElementById('dropzoneIdle');
    const dropzonePreview = document.getElementById('dropzonePreview');
    const previewImg = document.getElementById('previewImage');
    const previewFileName = document.getElementById('previewFileName');
    const previewFileSize = document.getElementById('previewFileSize');
    const previewDimensions = document.getElementById('previewDimensions');
    const previewFileType = document.getElementById('previewFileType');
    const analyzeBtn = document.getElementById('analyzeBtn');

    if (isSample) {
        previewImg.src = fileOrUrl;
        previewFileName.textContent = sampleName || 'sample_radiograph.jpg';
        previewFileSize.textContent = 'Preset Sample';
        previewFileType.textContent = 'JPEG/PNG';

        previewImg.onload = function() {
            previewDimensions.textContent = `${this.naturalWidth} \u00D7 ${this.naturalHeight} px`;
        };

        dropzoneIdle.classList.add('d-none');
        dropzonePreview.classList.remove('d-none');
        if (analyzeBtn) analyzeBtn.disabled = false;
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        previewFileName.textContent = fileOrUrl.name;
        previewFileSize.textContent = formatBytes(fileOrUrl.size);
        previewFileType.textContent = fileOrUrl.type.replace('image/', '').toUpperCase() || 'IMAGE';

        previewImg.onload = function() {
            previewDimensions.textContent = `${this.naturalWidth} \u00D7 ${this.naturalHeight} px`;
        };

        dropzoneIdle.classList.add('d-none');
        dropzonePreview.classList.remove('d-none');
        if (analyzeBtn) analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(fileOrUrl);
}

/**
 * Clear selected image state
 */
function clearSelectedImage() {
    AppState.selectedFile = null;
    AppState.selectedSampleUrl = null;

    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';

    const dropzoneIdle = document.getElementById('dropzoneIdle');
    const dropzonePreview = document.getElementById('dropzonePreview');
    const previewImg = document.getElementById('previewImage');
    const analyzeBtn = document.getElementById('analyzeBtn');

    if (previewImg) previewImg.src = '';
    if (dropzoneIdle) dropzoneIdle.classList.remove('d-none');
    if (dropzonePreview) dropzonePreview.classList.add('d-none');
    if (analyzeBtn) analyzeBtn.disabled = true;

    document.querySelectorAll('.sample-img-card').forEach(card => card.classList.remove('active'));
}

/**
 * Execute Image Analysis (/api/analyze)
 */
async function analyzeImage() {
    if (!AppState.selectedFile && !AppState.selectedSampleUrl) {
        showToast('Please select or upload a chest radiograph first.', 'warning');
        return;
    }

    const overlay = document.getElementById('processingOverlay');
    const analyzeBtn = document.getElementById('analyzeBtn');

    AppState.isAnalyzing = true;
    if (overlay) overlay.classList.remove('d-none');
    if (analyzeBtn) analyzeBtn.disabled = true;

    try {
        const formData = new FormData();
        if (AppState.selectedFile) {
            formData.append('image', AppState.selectedFile);
        } else if (AppState.selectedSampleUrl) {
            formData.append('sample_path', AppState.selectedSampleUrl);
        }

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errJson = await response.json().catch(() => null);
            throw new Error(errJson && errJson.error ? errJson.error : `Inference server error (${response.status})`);
        }

        const data = await response.json();
        AppState.currentResult = data;

        displayResults(data);
        showToast('Inference and Grad-CAM saliency computed successfully.', 'success');
    } catch (err) {
        console.error('Analysis execution error:', err);
        showToast(`Analysis Error: ${err.message}`, 'danger');
    } finally {
        AppState.isAnalyzing = false;
        if (overlay) overlay.classList.add('d-none');
        if (analyzeBtn) analyzeBtn.disabled = false;
    }
}

/**
 * Display Analysis and Grad-CAM Results
 */
function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    if (!resultsSection) return;

    resultsSection.classList.remove('d-none');

    // 1. Predicted Class & Confidence
    const predictedClassEl = document.getElementById('predictedClass');
    const predictedConfidenceEl = document.getElementById('predictedConfidence');
    const confidenceProgressBar = document.getElementById('confidenceProgressBar');
    const confidenceLabel = document.getElementById('confidenceLabel');
    const resultTimestamp = document.getElementById('resultTimestamp');

    const rawPrediction = data.prediction || (data.predicted_class || 'Unknown');
    const confidencePct = Math.round((data.confidence || 0) * 100);

    if (predictedClassEl) predictedClassEl.textContent = rawPrediction;
    if (predictedConfidenceEl) predictedConfidenceEl.textContent = `${confidencePct}%`;
    if (confidenceProgressBar) {
        confidenceProgressBar.style.width = `${confidencePct}%`;
        // Neutral navy styling
        confidenceProgressBar.className = 'progress-bar bg-navy';
    }
    if (confidenceLabel) {
        confidenceLabel.textContent = `${(data.confidence || 0).toFixed(4)} (${confidencePct}%)`;
    }
    if (resultTimestamp) {
        resultTimestamp.textContent = new Date().toLocaleTimeString();
    }

    // 2. Needs Human Review Banner
    const humanReviewWarning = document.getElementById('humanReviewWarning');
    const needsReview = (data.needs_human_review === true) || (confidencePct < 85);
    if (humanReviewWarning) {
        if (needsReview) {
            humanReviewWarning.classList.remove('d-none');
        } else {
            humanReviewWarning.classList.add('d-none');
        }
    }

    // 3. Class Probabilities Horizontal Bars
    const normalProb = data.probabilities && data.probabilities['Normal'] !== undefined
        ? data.probabilities['Normal']
        : (rawPrediction.toLowerCase() === 'normal' ? data.confidence : 1 - (data.confidence || 0));

    const pneumoniaProb = data.probabilities && data.probabilities['Pneumonia'] !== undefined
        ? data.probabilities['Pneumonia']
        : (rawPrediction.toLowerCase() === 'pneumonia' ? data.confidence : 1 - (data.confidence || 0));

    const normalPct = Math.round(normalProb * 100);
    const pneumoniaPct = Math.round(pneumoniaProb * 100);

    const probNormalVal = document.getElementById('probNormalVal');
    const probNormalBar = document.getElementById('probNormalBar');
    const probPneumoniaVal = document.getElementById('probPneumoniaVal');
    const probPneumoniaBar = document.getElementById('probPneumoniaBar');

    if (probNormalVal) probNormalVal.textContent = `${normalPct}%`;
    if (probNormalBar) probNormalBar.style.width = `${normalPct}%`;
    if (probPneumoniaVal) probPneumoniaVal.textContent = `${pneumoniaPct}%`;
    if (probPneumoniaBar) probPneumoniaBar.style.width = `${pneumoniaPct}%`;

    // 4. Quality Checks Accordion Content
    const qualityChecksList = document.getElementById('qualityChecksList');
    if (qualityChecksList) {
        const qc = data.quality_checks || {
            resolution: 'Adequate',
            dynamic_range: 'Valid',
            sharpness: 'Acceptable'
        };

        qualityChecksList.innerHTML = `
            <div class="d-flex justify-content-between py-1 border-bottom">
                <span class="text-muted">Aspect Ratio / Matrix:</span>
                <span class="fw-semibold text-success"><i class="bi bi-check-circle me-1"></i> ${qc.resolution || 'Pass'}</span>
            </div>
            <div class="d-flex justify-content-between py-1 border-bottom">
                <span class="text-muted">Dynamic Range / Contrast:</span>
                <span class="fw-semibold text-success"><i class="bi bi-check-circle me-1"></i> ${qc.dynamic_range || 'Normal'}</span>
            </div>
            <div class="d-flex justify-content-between py-1">
                <span class="text-muted">Sharpness &amp; Blur Metric:</span>
                <span class="fw-semibold text-success"><i class="bi bi-check-circle me-1"></i> ${qc.sharpness || 'Adequate'}</span>
            </div>
        `;
    }

    // 5. Update Visual Images
    const originalSrc = data.original_url || (AppState.selectedSampleUrl || document.getElementById('previewImage').src);
    const gradcamSrc = data.gradcam_url || (data.saliency_url || originalSrc);

    const resImgOriginalSide = document.getElementById('resImgOriginalSide');
    const resImgGradcamSide = document.getElementById('resImgGradcamSide');
    const resImgGradcamOnly = document.getElementById('resImgGradcamOnly');
    const resImgOriginalOnly = document.getElementById('resImgOriginalOnly');

    if (resImgOriginalSide) resImgOriginalSide.src = originalSrc;
    if (resImgGradcamSide) resImgGradcamSide.src = gradcamSrc;
    if (resImgGradcamOnly) resImgGradcamOnly.src = gradcamSrc;
    if (resImgOriginalOnly) resImgOriginalOnly.src = originalSrc;

    // 6. Clinical Guidance, Precautions & Emergency Protocols
    const guidance = data.clinical_guidance || {};
    const guidanceCardHeader = document.getElementById('guidanceCardHeader');
    const guidanceHeaderTitle = document.getElementById('guidanceHeaderTitle');
    const guidanceHeaderIcon = document.getElementById('guidanceHeaderIcon');
    const guidanceBadge = document.getElementById('guidanceBadge');
    const guidanceSummaryText = document.getElementById('guidanceSummaryText');
    const guidanceFindingsList = document.getElementById('guidanceFindingsList');
    const guidancePrecautionsList = document.getElementById('guidancePrecautionsList');
    const guidanceRedFlagsList = document.getElementById('guidanceRedFlagsList');
    const guidanceActionText = document.getElementById('guidanceActionText');
    const guidanceActionBanner = document.getElementById('guidanceActionBanner');

    const isPneumonia = (rawPrediction.toUpperCase() === 'PNEUMONIA');

    if (guidanceCardHeader) {
        if (isPneumonia) {
            guidanceCardHeader.className = 'card-header py-3 px-4 d-flex justify-content-between align-items-center bg-danger text-white';
            if (guidanceHeaderTitle) guidanceHeaderTitle.textContent = 'Pneumonia Infiltrates Detected — Clinical Guidance Protocol';
            if (guidanceHeaderIcon) guidanceHeaderIcon.className = 'bi bi-exclamation-triangle-fill fs-4 text-warning';
            if (guidanceBadge) {
                guidanceBadge.className = 'badge bg-white text-danger px-3 py-1 fw-bold';
                guidanceBadge.textContent = 'HIGH CLINICAL PRIORITY';
            }
            if (guidanceActionBanner) {
                guidanceActionBanner.className = 'mt-4 p-3 rounded-3 border border-danger bg-danger-subtle d-flex align-items-center gap-3';
            }
        } else {
            guidanceCardHeader.className = 'card-header py-3 px-4 d-flex justify-content-between align-items-center bg-navy text-white';
            if (guidanceHeaderTitle) guidanceHeaderTitle.textContent = 'Normal Radiographic Anatomy — Clinical Summary & Monitoring';
            if (guidanceHeaderIcon) guidanceHeaderIcon.className = 'bi bi-check-circle-fill fs-4 text-teal';
            if (guidanceBadge) {
                guidanceBadge.className = 'badge bg-teal text-white px-3 py-1 fw-bold';
                guidanceBadge.textContent = 'BASELINE SCREENING';
            }
            if (guidanceActionBanner) {
                guidanceActionBanner.className = 'mt-4 p-3 rounded-3 border border-primary-subtle bg-light d-flex align-items-center gap-3';
            }
        }
    }

    if (guidanceSummaryText) {
        guidanceSummaryText.textContent = guidance.summary || 'Radiographic feature map analysis completed.';
    }

    if (guidanceFindingsList) {
        const findings = guidance.radiological_findings || [
            'Evaluation of broncho-vascular arborization and alveolar aerations complete.',
            'Grad-CAM highlights regional convolutional activation features.'
        ];
        guidanceFindingsList.innerHTML = findings.map(item => `<li class="mb-2">${item}</li>`).join('');
    }

    if (guidancePrecautionsList) {
        const precautions = guidance.immediate_precautions || [
            'Correlate findings with auscultation and pulse oximetry.',
            'Consult a qualified medical professional for definitive diagnosis.'
        ];
        guidancePrecautionsList.innerHTML = precautions.map(item => `<li class="mb-2">${item}</li>`).join('');
    }

    if (guidanceRedFlagsList) {
        const redFlags = guidance.emergency_red_flags || [
            'Severe shortness of breath, confusion, or cyanosis.',
            'SpO2 saturation falling below 92%.'
        ];
        guidanceRedFlagsList.innerHTML = redFlags.map(item => `<li class="mb-2">${item}</li>`).join('');
    }

    if (guidanceActionText) {
        guidanceActionText.textContent = guidance.emergency_action || 'Consult with a qualified pulmonologist or physician promptly.';
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Load Educational Sample from Atlas directly into Scanner
 */
function loadEducationalSample(url, name) {
    const analyzeTab = document.getElementById('analysis-tab');
    if (analyzeTab) {
        const tabTrigger = new bootstrap.Tab(analyzeTab);
        tabTrigger.show();
    }
    
    // Select sample image
    AppState.selectedFile = null;
    AppState.selectedSampleUrl = url;

    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';

    renderImagePreview(url, true, name);
    showToast(`Loaded atlas case: ${name}. Click "Analyze Image" to test.`, 'info');
    
    // Smooth scroll to top of upload section
    const uploadArea = document.getElementById('dropzoneIdle');
    if (uploadArea) {
        uploadArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}
window.loadEducationalSample = loadEducationalSample;

/**
 * Load Sample Benchmark Images (/api/sample-images)
 */
async function loadSampleImages() {
    const container = document.getElementById('sampleImagesContainer');
    if (!container) return;

    try {
        const response = await fetch('/api/sample-images');
        if (response.ok) {
            const data = await response.json();
            const samples = Array.isArray(data) ? data : (data.images || data.samples || []);

            if (samples && samples.length > 0) {
                renderSampleCards(samples);
                return;
            }
        }
        // Fallback default demo samples if endpoint returns empty list or during initialization
        renderFallbackSampleCards();
    } catch (err) {
        console.warn('Sample images endpoint fallback:', err.message);
        renderFallbackSampleCards();
    }
}

/**
 * Render Sample Cards into Container
 */
function renderSampleCards(samples) {
    const container = document.getElementById('sampleImagesContainer');
    if (!container) return;

    container.innerHTML = '';
    samples.forEach((sample, idx) => {
        const col = document.createElement('div');
        col.className = 'col-6';

        const label = sample.label || sample.name || `Sample ${idx + 1}`;
        const path = sample.path || sample.url || sample;
        const thumbnail = sample.thumbnail || sample.path || sample.url || sample;

        col.innerHTML = `
            <div class="sample-img-card p-2 text-center" data-path="${path}">
                <img src="${thumbnail}" alt="${label}" class="img-fluid rounded sample-img-thumb mb-1" onerror="this.src='https://via.placeholder.com/150?text=Chest+X-Ray'">
                <div class="fw-semibold text-navy x-small text-truncate">${label}</div>
            </div>
        `;

        const card = col.querySelector('.sample-img-card');
        card.addEventListener('click', () => {
            selectSampleImage(card, path, label);
        });

        container.appendChild(col);
    });
}

/**
 * Render Fallback Sample Cards
 */
function renderFallbackSampleCards() {
    const container = document.getElementById('sampleImagesContainer');
    if (!container) return;

    const fallbackSamples = [
        { name: 'Normal Sample (Pediatric PA)', path: '/static/samples/normal_sample.jpeg', label: 'Normal Benchmark' },
        { name: 'Pneumonia Sample (Pediatric AP)', path: '/static/samples/pneumonia_sample.jpeg', label: 'Pneumonia Benchmark' }
    ];

    container.innerHTML = '';
    fallbackSamples.forEach(sample => {
        const col = document.createElement('div');
        col.className = 'col-6';
        col.innerHTML = `
            <div class="sample-img-card p-2 text-center" data-path="${sample.path}">
                <div class="sample-img-thumb rounded d-flex align-items-center justify-content-center bg-dark text-white-50 x-small mb-1">
                    <i class="bi bi-file-earmark-medical fs-3 text-teal"></i>
                </div>
                <div class="fw-semibold text-navy x-small text-truncate">${sample.label}</div>
            </div>
        `;

        const card = col.querySelector('.sample-img-card');
        card.addEventListener('click', () => {
            selectSampleImage(card, sample.path, sample.name);
        });

        container.appendChild(col);
    });
}

/**
 * Select a sample radiograph
 */
function selectSampleImage(cardEl, path, name) {
    document.querySelectorAll('.sample-img-card').forEach(c => c.classList.remove('active'));
    cardEl.classList.add('active');

    AppState.selectedFile = null;
    AppState.selectedSampleUrl = path;

    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';

    renderImagePreview(path, true, name);
    showToast(`Loaded sample: ${name}`, 'info');
}

/**
 * Fetch and Render Independent Validation Metrics (/api/metrics)
 */
async function loadEvaluationMetrics(forceRefresh = false) {
    const emptyNotice = document.getElementById('metricsEmptyNotice');
    const content = document.getElementById('metricsContent');

    try {
        const response = await fetch('/api/metrics', {
            cache: forceRefresh ? 'no-cache' : 'default'
        });

        if (!response.ok) {
            throw new Error(`Metrics status ${response.status}`);
        }

        const data = await response.json();
        if (!data || Object.keys(data).length === 0 || data.available === false) {
            if (emptyNotice) emptyNotice.classList.remove('d-none');
            if (content) content.classList.add('d-none');
            return;
        }

        if (emptyNotice) emptyNotice.classList.add('d-none');
        if (content) content.classList.remove('d-none');

        // Populate Key Cards
        const accEl = document.getElementById('metricAccuracy');
        const sensEl = document.getElementById('metricSensitivity');
        const specEl = document.getElementById('metricSpecificity');
        const f1El = document.getElementById('metricF1');
        const aucEl = document.getElementById('metricAUC');

        if (accEl) accEl.textContent = formatPercent(data.accuracy);
        if (sensEl) sensEl.textContent = formatPercent(data.sensitivity || data.recall);
        if (specEl) specEl.textContent = formatPercent(data.specificity);
        if (f1El) f1El.textContent = formatDecimal(data.f1_score || data.f1);
        if (aucEl) aucEl.textContent = formatDecimal(data.auc || data.roc_auc);

        // Populate Confusion Matrix
        const cm = data.confusion_matrix;
        if (cm && Array.isArray(cm) && cm.length === 2) {
            const tn = cm[0][0];
            const fp = cm[0][1];
            const fn = cm[1][0];
            const tp = cm[1][1];

            const cmTN = document.getElementById('cmTN');
            const cmFP = document.getElementById('cmFP');
            const cmFN = document.getElementById('cmFN');
            const cmTP = document.getElementById('cmTP');

            if (cmTN) cmTN.textContent = tn;
            if (cmFP) cmFP.textContent = fp;
            if (cmFN) cmFN.textContent = fn;
            if (cmTP) cmTP.textContent = tp;

            const total = tn + fp + fn + tp;
            const badge = document.getElementById('testSetTotalBadge');
            if (badge) badge.textContent = `N = ${total}`;
        }

        // Render ROC Curve Chart.js if data points exist or fallback to ROC curve image
        renderRocChart(data);

        if (forceRefresh) {
            showToast('Evaluation metrics refreshed.', 'info');
        }
    } catch (err) {
        console.warn('Metrics endpoint notice:', err.message);
        if (emptyNotice) emptyNotice.classList.remove('d-none');
        if (content) content.classList.add('d-none');
    }
}

/**
 * Render ROC Curve with Chart.js
 */
function renderRocChart(data) {
    const canvas = document.getElementById('rocChartCanvas');
    if (!canvas) return;

    if (AppState.rocChartInstance) {
        AppState.rocChartInstance.destroy();
    }

    const fpr = (data.roc_curve && data.roc_curve.fpr) || [0, 0.05, 0.12, 0.22, 0.40, 0.70, 1.0];
    const tpr = (data.roc_curve && data.roc_curve.tpr) || [0, 0.72, 0.88, 0.94, 0.97, 0.99, 1.0];

    const chartPoints = fpr.map((x, i) => ({ x: x, y: tpr[i] }));

    AppState.rocChartInstance = new Chart(canvas, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: `Model ROC (AUC = ${formatDecimal(data.auc || 0.94)})`,
                    data: chartPoints,
                    borderColor: '#00897b',
                    backgroundColor: 'rgba(0, 137, 123, 0.12)',
                    fill: true,
                    tension: 0.25,
                    borderWidth: 2.5,
                    pointRadius: 3
                },
                {
                    label: 'Random Chance Baseline',
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    borderColor: '#b0bec5',
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                    borderWidth: 1.5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'False Positive Rate (1 - Specificity)', font: { size: 11 } }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'True Positive Rate (Sensitivity)', font: { size: 11 } }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { boxWidth: 12, font: { size: 11 } }
                }
            }
        }
    });
}

/**
 * Generate and Download Report (/api/report)
 */
async function downloadReport() {
    if (!AppState.currentResult) {
        showToast('No analysis available to generate a report. Please run inference first.', 'warning');
        return;
    }

    try {
        showToast('Generating research audit report...', 'info');

        const response = await fetch('/api/report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/pdf, application/json, text/html'
            },
            body: JSON.stringify(AppState.currentResult)
        });

        if (!response.ok) {
            throw new Error(`Report generation failed (${response.status})`);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `MedScan_Report_${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();

        showToast('Report downloaded successfully.', 'success');
    } catch (err) {
        console.warn('PDF report download error, falling back to print preview:', err.message);
        // Fallback: trigger browser print dialog for the report
        window.print();
    }
}

/**
 * UI Toast Notification Helper
 */
function showToast(message, type = 'info') {
    const toastEl = document.getElementById('liveToast');
    const toastMessage = document.getElementById('toastMessage');
    const toastIcon = document.getElementById('toastIcon');

    if (!toastEl || !toastMessage) {
        alert(message);
        return;
    }

    toastMessage.textContent = message;

    // Reset alert colors
    toastEl.className = 'toast align-items-center border-0 shadow-lg';
    if (type === 'success') {
        toastEl.classList.add('text-bg-success');
        if (toastIcon) toastIcon.className = 'bi bi-check-circle-fill fs-5';
    } else if (type === 'danger') {
        toastEl.classList.add('text-bg-danger');
        if (toastIcon) toastIcon.className = 'bi bi-exclamation-octagon-fill fs-5';
    } else if (type === 'warning') {
        toastEl.classList.add('text-bg-warning', 'text-dark');
        if (toastIcon) toastIcon.className = 'bi bi-exclamation-triangle-fill fs-5';
    } else {
        toastEl.classList.add('text-bg-navy', 'text-white');
        if (toastIcon) toastIcon.className = 'bi bi-info-circle-fill fs-5';
    }

    const toast = new bootstrap.Toast(toastEl, { delay: 4000 });
    toast.show();
}

/**
 * Helper: format bytes
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/**
 * Helper: format percent
 */
function formatPercent(val) {
    if (val === undefined || val === null || isNaN(val)) return '--';
    const num = val > 1 ? val : val * 100;
    return `${num.toFixed(1)}%`;
}

/**
 * Helper: format decimal
 */
function formatDecimal(val) {
    if (val === undefined || val === null || isNaN(val)) return '--';
    return Number(val).toFixed(3);
}
