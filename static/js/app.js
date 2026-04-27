/**
 * CertVerify — Frontend Application
 * ==================================
 * Handles multi-file upload, batch pipeline visualisation, 
 * and results rendering for the Certificate Authenticity Verification System.
 */

/* ── State ──────────────────────────────────────────────────────────────── */
let selectedFiles = [];
let batchResults = [];
let activeResultIndex = 0;

/* ── DOM refs ───────────────────────────────────────────────────────────── */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const uploadZone       = $('#uploadZone');
const fileInput        = $('#fileInput');
const filePreviewList  = $('#filePreviewList');
const uploadActions    = $('#uploadActions');
const addMoreBtn       = $('#addMoreBtn');
const verifyBtn        = $('#verifyBtn');

const uploadSection    = $('#uploadSection');
const pipelineSection  = $('#pipelineSection');
const resultsSection   = $('#resultsSection');

const pipelineEl       = $('#pipeline');
const pipelineBar      = $('#pipelineBar');
const pipelineDesc     = $('#pipelineDesc');

const resultNavigator  = $('#resultNavigator');
const resultList       = $('#resultList');
const resultCount      = $('#resultCount');

/* ── Pipeline step definitions ──────────────────────────────────────────── */
const PIPELINE_STEPS = [
    { id: 'validate',   label: 'File Validation',           icon: '🔍' },
    { id: 'metadata',   label: 'Metadata Extraction',       icon: '📋' },
    { id: 'render',     label: 'PDF Rendering',             icon: '🖼️' },
    { id: 'preprocess', label: 'Image Preprocessing',       icon: '⚙️' },
    { id: 'ocr',        label: 'OCR Text Extraction',       icon: '📝' },
    { id: 'layout',     label: 'Layout Validation',         icon: '📐' },
    { id: 'signature',  label: 'Signature & Seal',          icon: '✍️' },
    { id: 'forensic',   label: 'Forensic Analysis',         icon: '🔬' },
    { id: 'classify',   label: 'ML Classification',         icon: '🤖' },
    { id: 'report',     label: 'Report Generation',         icon: '📊' },
];

/* ── Initialisation ─────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', init);

function init() {
    setupUploadZone();
    setupButtons();
    setupTabs();
}

/* ── Upload Zone ────────────────────────────────────────────────────────── */
function setupUploadZone() {
    uploadZone.addEventListener('click', () => fileInput.click());

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('upload-zone--dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('upload-zone--dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('upload-zone--dragover');
        handleFileSelect(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', () => {
        handleFileSelect(fileInput.files);
    });
}

function handleFileSelect(files) {
    const allowed = ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp'];
    
    Array.from(files).forEach(file => {
        const ext = file.name.split('.').pop().toLowerCase();
        if (!allowed.includes(ext)) {
            alert(`"${file.name}" is an unsupported format.`);
            return;
        }
        if (file.size > 50 * 1024 * 1024) {
            alert(`"${file.name}" is too large (max 50MB).`);
            return;
        }
        
        // Add to state
        if (!selectedFiles.find(f => f.name === file.name && f.size === file.size)) {
            selectedFiles.push(file);
        }
    });

    renderFilePreview();
}

function renderFilePreview() {
    if (selectedFiles.length === 0) {
        uploadZone.style.display = '';
        filePreviewList.style.display = 'none';
        uploadActions.style.display = 'none';
        return;
    }

    uploadZone.style.display = 'none';
    filePreviewList.style.display = 'flex';
    uploadActions.style.display = 'flex';

    filePreviewList.innerHTML = selectedFiles.map((file, index) => `
        <div class="file-preview-item">
            <div class="file-preview-item__info">
                <div class="file-preview-item__icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14,2 14,8 20,8"/></svg>
                </div>
                <div>
                    <p class="file-preview-item__name">${file.name}</p>
                    <p class="file-preview-item__size">${formatFileSize(file.size)}</p>
                </div>
            </div>
            <button class="btn btn--outline btn--sm" onclick="removeFile(${index})">Remove</button>
        </div>
    `).join('');

    verifyBtn.innerHTML = `
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM16 16l-3.5-3.5"/></svg>
        Verify ${selectedFiles.length} ${selectedFiles.length === 1 ? 'File' : 'Files'}
    `;
}

window.removeFile = function(index) {
    selectedFiles.splice(index, 1);
    renderFilePreview();
};

/* ── Buttons ────────────────────────────────────────────────────────────── */
function setupButtons() {
    addMoreBtn.addEventListener('click', () => fileInput.click());
    verifyBtn.addEventListener('click', startVerification);
    $('#newVerificationBtn').addEventListener('click', resetToUpload);
    $('#downloadReportBtn').addEventListener('click', downloadCurrentReport);
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
function setupTabs() {
    $$('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.tab').forEach(t => t.classList.remove('tab--active'));
            $$('.tab-panel').forEach(p => p.classList.remove('tab-panel--active'));
            tab.classList.add('tab--active');
            $(`#panel-${tab.dataset.tab}`).classList.add('tab-panel--active');
        });
    });
}

/* ── Verification Pipeline ──────────────────────────────────────────────── */
let pipelineAnimInterval = null;
let pipelineTimerInterval = null;
let pipelineStartTime = null;

const STATUS_MESSAGES = [
    'Initialising verification engine…',
    'Loading documents into memory…',
    'Extracting raw image data…',
    'Running image enhancement filters…',
    'Performing deep OCR analysis…',
    'Scanning layout structures…',
    'Detecting signature regions…',
    'Running forensic pixel analysis…',
    'Training ML classifier on features…',
    'Compiling final reports…',
    'Cross-referencing extracted data…',
    'Validating document integrity…',
    'Batch processing active — please wait…',
];

async function startVerification() {
    if (selectedFiles.length === 0) return;

    // Disable buttons
    verifyBtn.disabled = true;
    addMoreBtn.disabled = true;
    verifyBtn.innerHTML = '<span class="spinner"></span> Verifying…';

    // Show pipeline
    showSection('pipeline');
    renderPipeline();
    pipelineDesc.textContent = `Analysing ${selectedFiles.length} document${selectedFiles.length > 1 ? 's' : ''} through multiple verification modules…`;

    // Start the animation
    animatePipelineStart();

    try {
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        const response = await fetch('/api/verify-batch', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok || !data.success) {
            throw new Error(data.detail || `Server error (${response.status})`);
        }

        batchResults = data.results;

        // Stop the animation loops
        stopPipelineAnimation();

        // Complete the pipeline animation (using the first result's steps as a proxy for the UI)
        completePipeline(batchResults[0].pipeline_steps);

        // Short delay, then show results
        await sleep(800);
        showBatchResults();

    } catch (err) {
        console.error('Verification failed:', err);
        stopPipelineAnimation();
        alert(`Verification failed: ${err.message}`);
        resetToUpload();
    } finally {
        verifyBtn.disabled = false;
        addMoreBtn.disabled = false;
        renderFilePreview(); // Reset button text
    }
}

/* ── Pipeline rendering ─────────────────────────────────────────────────── */
function renderPipeline() {
    pipelineEl.innerHTML = PIPELINE_STEPS.map((step, i) => `
        <div class="pipeline-step" id="step-${step.id}" style="animation-delay: ${i * 0.05}s">
            <div class="pipeline-step__icon">${step.icon}</div>
            <span class="pipeline-step__label">${step.label}</span>
            <span class="pipeline-step__time" id="time-${step.id}"></span>
        </div>
    `).join('');

    let statusBar = $('#pipelineStatusBar');
    if (!statusBar) {
        statusBar = document.createElement('div');
        statusBar.id = 'pipelineStatusBar';
        statusBar.className = 'pipeline-status-bar';
        statusBar.innerHTML = `
            <div class="pipeline-status-bar__left">
                <span class="pipeline-status-bar__spinner"></span>
                <span class="pipeline-status-bar__msg" id="pipelineStatusMsg">Initialising batch process…</span>
            </div>
            <span class="pipeline-status-bar__timer" id="pipelineTimer">0s elapsed</span>
        `;
        pipelineSection.appendChild(statusBar);
    }
}

function animatePipelineStart() {
    let i = 0;
    pipelineStartTime = Date.now();

    pipelineAnimInterval = setInterval(() => {
        const step = PIPELINE_STEPS[i];
        const el = $(`#step-${step.id}`);
        if (el) {
            $$('.pipeline-step--active').forEach(s => s.classList.remove('pipeline-step--active'));
            el.classList.add('pipeline-step--active');
            const progress = Math.min(((i + 1) / PIPELINE_STEPS.length) * 100, 100);
            pipelineBar.style.width = `${progress}%`;
        }
        i = (i + 1) % PIPELINE_STEPS.length;
    }, 1200);

    const statusMsgEl = $('#pipelineStatusMsg');
    if (statusMsgEl) {
        let prevMsg = '';
        pipelineAnimInterval._msgInterval = setInterval(() => {
            let newMsg;
            do {
                newMsg = STATUS_MESSAGES[Math.floor(Math.random() * STATUS_MESSAGES.length)];
            } while (newMsg === prevMsg);
            prevMsg = newMsg;
            statusMsgEl.style.opacity = '0';
            setTimeout(() => {
                statusMsgEl.textContent = newMsg;
                statusMsgEl.style.opacity = '1';
            }, 300);
        }, 3500);
    }

    const timerEl = $('#pipelineTimer');
    pipelineTimerInterval = setInterval(() => {
        if (timerEl && pipelineStartTime) {
            const elapsed = Math.round((Date.now() - pipelineStartTime) / 1000);
            timerEl.textContent = `${elapsed}s elapsed`;
        }
    }, 1000);
}

function stopPipelineAnimation() {
    if (pipelineAnimInterval) {
        clearInterval(pipelineAnimInterval);
        if (pipelineAnimInterval._msgInterval) clearInterval(pipelineAnimInterval._msgInterval);
        pipelineAnimInterval = null;
    }
    if (pipelineTimerInterval) {
        clearInterval(pipelineTimerInterval);
        pipelineTimerInterval = null;
    }
}

function completePipeline(steps) {
    $$('.pipeline-step').forEach(el => {
        el.classList.remove('pipeline-step--active');
        el.classList.add('pipeline-step--done');
    });

    if (steps) {
        steps.forEach((s, i) => {
            if (i < PIPELINE_STEPS.length) {
                const timeEl = $(`#time-${PIPELINE_STEPS[i].id}`);
                if (timeEl) timeEl.textContent = `${s.time}s`;
            }
        });
    }

    pipelineBar.style.width = '100%';

    const statusMsg = $('#pipelineStatusMsg');
    const statusSpinner = $('.pipeline-status-bar__spinner');
    const timerEl = $('#pipelineTimer');
    if (statusMsg) statusMsg.textContent = '✓ Batch verification complete!';
    if (statusSpinner) {
        statusSpinner.className = 'pipeline-status-bar__check';
        statusSpinner.textContent = '✓';
    }
    if (timerEl && pipelineStartTime) {
        const total = ((Date.now() - pipelineStartTime) / 1000).toFixed(1);
        timerEl.textContent = `${total}s total`;
    }
}

/* ── Results Dashboard ──────────────────────────────────────────────────── */
function showBatchResults() {
    showSection('results');
    activeResultIndex = 0;
    renderResultNavigator();
    displayResult(activeResultIndex);
}

function renderResultNavigator() {
    if (batchResults.length <= 1) {
        resultNavigator.style.display = 'none';
        return;
    }

    resultNavigator.style.display = 'block';
    resultCount.textContent = `${activeResultIndex + 1} / ${batchResults.length}`;

    resultList.innerHTML = batchResults.map((res, i) => {
        const risk = res.report ? res.report.risk_level : 'Error';
        const statusClass = risk ? `status--${risk.toLowerCase()}` : 'status--error';
        const activeClass = i === activeResultIndex ? 'result-item--active' : '';
        
        return `
            <div class="result-item ${activeClass}" onclick="switchResult(${i})">
                <span class="result-item__name">${res.filename}</span>
                <span class="result-item__status ${statusClass}">${risk || 'Failed'}</span>
            </div>
        `;
    }).join('');
}

window.switchResult = function(index) {
    if (index === activeResultIndex) return;
    activeResultIndex = index;
    renderResultNavigator();
    displayResult(index);
};

function displayResult(index) {
    const data = batchResults[index];
    
    // Check if verification for this file was successful
    if (!data.success) {
        showRejection(data);
        return;
    }

    // Clear any existing rejection card
    const existing = $('#rejectionCard');
    if (existing) existing.remove();
    
    // Show the results sub-sections (they might have been hidden by a rejection)
    $('#verdictCard').style.display = 'flex';
    $('#scoresGrid').style.display = 'grid';
    $('#resultTabs').style.display = 'flex';
    $('.tab-panel--active').style.display = 'block';

    const report = data.report;
    const score = report.authenticity_score || 0;
    const risk = report.risk_level || 'Unknown';

    // Gauge animation
    animateGauge(score);

    // Verdict badge
    const badge = $('#verdictBadge');
    badge.textContent = risk;
    badge.className = 'verdict-badge';
    if (risk === 'Genuine') badge.classList.add('verdict-badge--genuine');
    else if (risk === 'Suspicious') badge.classList.add('verdict-badge--suspicious');
    else badge.classList.add('verdict-badge--forged');

    // Update gauge gradient colour
    const gaugeGrad = $('#gaugeGrad');
    if (gaugeGrad) {
        const stops = gaugeGrad.querySelectorAll('stop');
        if (risk === 'Genuine') {
            stops[0].setAttribute('stop-color', '#10b981');
            stops[1].setAttribute('stop-color', '#34d399');
        } else if (risk === 'Suspicious') {
            stops[0].setAttribute('stop-color', '#f59e0b');
            stops[1].setAttribute('stop-color', '#fbbf24');
        } else {
            stops[0].setAttribute('stop-color', '#f43f5e');
            stops[1].setAttribute('stop-color', '#fb7185');
        }
    }

    $('#verdictSummary').textContent = report.summary || '';
    $('#verdictConfidence').textContent = `Confidence: ${((report.confidence || 0) * 100).toFixed(1)}%`;
    $('#verdictTime').textContent = `Time: ${data.processing_time}s`;
    $('#verdictMethod').textContent = `Method: ${report.method || 'Heuristic'}`;

    renderScores(report.module_scores || {});
    renderFindings(report.findings || []);
    renderExtractedData(report.extracted_data || {});
    renderMetadata(report.metadata || {});
    renderPipelineDetails(data.pipeline_steps || []);
}

function animateGauge(score) {
    const arc = $('#gaugeArc');
    const numberEl = $('#gaugeNumber');
    const circumference = 2 * Math.PI * 85;
    const offset = circumference - (score / 100) * circumference;

    arc.style.transition = 'stroke-dashoffset 1.5s ease-out';
    setTimeout(() => {
        arc.setAttribute('stroke-dashoffset', offset);
    }, 100);

    animateValue(numberEl, 0, Math.round(score), 1500);
}

function animateValue(el, start, end, duration) {
    const range = end - start;
    const startTime = performance.now();

    function step(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + range * eased);
        el.textContent = current;
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

function renderScores(moduleScores) {
    const grid = $('#scoresGrid');
    grid.innerHTML = '';

    const scoreConfig = {
        'OCR Quality':           { colour: '#00d4ff' },
        'Layout Consistency':    { colour: '#7c3aed' },
        'Metadata Integrity':    { colour: '#e040fb' },
        'Signature Authenticity':{ colour: '#10b981' },
        'Seal Verification':     { colour: '#f59e0b' },
        'Forensic Analysis':     { colour: '#f43f5e' },
    };

    Object.entries(moduleScores).forEach(([label, value], i) => {
        const config = scoreConfig[label] || { colour: '#00d4ff' };
        const card = document.createElement('div');
        card.className = 'score-card';
        card.style.animationDelay = `${i * 0.1}s`;
        card.innerHTML = `
            <div class="score-card__label">${label}</div>
            <div class="score-card__value" style="color: ${config.colour}">${Math.round(value)}</div>
            <div class="score-card__bar">
                <div class="score-card__fill" id="fill-${i}" style="background: ${config.colour}"></div>
            </div>
        `;
        grid.appendChild(card);
        setTimeout(() => {
            const fill = document.getElementById(`fill-${i}`);
            if (fill) fill.style.width = `${value}%`;
        }, 300 + i * 100);
    });
}

function renderFindings(findings) {
    const list = $('#findingsList');
    list.innerHTML = '';
    if (!findings.length) {
        list.innerHTML = '<p style="color: var(--text-secondary); text-align:center; padding: 2rem;">No findings to display.</p>';
        return;
    }
    findings.forEach(f => {
        const item = document.createElement('div');
        item.className = 'finding-item';
        item.innerHTML = `
            <div class="finding-dot finding-dot--${f.severity || 'info'}"></div>
            <div>
                <div class="finding-module">${f.module || 'General'}</div>
                <div class="finding-desc">${f.description || ''}</div>
            </div>
        `;
        list.appendChild(item);
    });
}

function renderExtractedData(data) {
    const container = $('#extractedData');
    const fields = data.text_fields || {};
    const avgConf = data.average_ocr_confidence || 0;
    let rows = '';
    Object.entries(fields).forEach(([key, value]) => {
        const display = value || '<span class="data-value--missing">Not detected</span>';
        rows += `<tr><td>${capitalise(key.replace(/_/g, ' '))}</td><td>${display}</td></tr>`;
    });
    rows += `<tr><td>Average OCR Confidence</td><td>${avgConf}%</td></tr>`;
    container.innerHTML = `
        <table class="data-table">
            <thead><tr><th>Field</th><th>Value</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

function renderMetadata(meta) {
    const container = $('#metadataInfo');
    const entries = [
        ['Format', meta.format || '—'],
        ['Pages', meta.page_count || '—'],
        ['Creator', meta.creator || '—'],
        ['Producer', meta.producer || '—'],
    ];
    let rows = entries.map(([k, v]) => `<tr><td>${k}</td><td>${v || '—'}</td></tr>`).join('');
    const flags = meta.suspicious_flags || [];
    if (flags.length) {
        rows += `<tr><td>Suspicious Flags</td><td style="color: var(--accent-amber)">${flags.join('<br>')}</td></tr>`;
    } else {
        rows += `<tr><td>Suspicious Flags</td><td style="color: var(--accent-green)">None</td></tr>`;
    }
    container.innerHTML = `
        <table class="data-table">
            <thead><tr><th>Property</th><th>Value</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

function renderPipelineDetails(steps) {
    const container = $('#pipelineDetails');
    if (!steps.length) {
        container.innerHTML = '<p style="color: var(--text-secondary); text-align:center; padding:2rem;">No pipeline data.</p>';
        return;
    }
    container.innerHTML = steps.map(s => `
        <div class="pipeline-detail-item">
            <span class="pipeline-detail-item__name">${s.step}</span>
            <span class="pipeline-detail-item__time">${s.time}s</span>
            <span class="pipeline-detail-item__status">✓ ${s.status}</span>
        </div>
    `).join('');
}

/* ── Rejection Display ──────────────────────────────────────────────────── */
function showRejection(data) {
    // Hide standard results
    $('#verdictCard').style.display = 'none';
    $('#scoresGrid').style.display = 'none';
    $('#resultTabs').style.display = 'none';
    const activePanel = $('.tab-panel--active');
    if (activePanel) activePanel.style.display = 'none';

    const reason  = data.rejection_reason || 'invalid_document';
    const message = data.detail           || 'This document could not be verified.';
    const docType = data.detected_type    || null;

    const iconMap = {
        domestic_id:      '🪪',
        not_certificate:  '📄',
        invalid_document: '❌',
    };
    const titleMap = {
        domestic_id:      'Domestic ID Document Detected',
        not_certificate:  'Not a Certificate',
        invalid_document: 'Invalid Document',
    };

    const icon  = iconMap[reason]  || '❌';
    const title = titleMap[reason] || 'Verification Rejected';

    let card = $('#rejectionCard');
    if (!card) {
        card = document.createElement('div');
        card.id = 'rejectionCard';
        card.className = 'rejection-card';
        resultsSection.insertBefore(card, $('.results-actions'));
    }
    
    card.innerHTML = `
        <div class="rejection-card__icon">${icon}</div>
        <h2 class="rejection-card__title">${title}</h2>
        <p class="rejection-card__message">${message}</p>
        ${docType ? `<p class="rejection-card__type">Detected type: <strong>${docType}</strong></p>` : ''}
    `;
}

/* ── Section management ─────────────────────────────────────────────────── */
function showSection(section) {
    uploadSection.style.display   = section === 'upload' ? '' : 'none';
    pipelineSection.style.display = section === 'pipeline' ? '' : 'none';
    resultsSection.style.display  = section === 'results' ? '' : 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function resetToUpload() {
    selectedFiles = [];
    batchResults = [];
    activeResultIndex = 0;
    
    fileInput.value = '';
    renderFilePreview();
    stopPipelineAnimation();
    
    const statusBar = $('#pipelineStatusBar');
    if (statusBar) statusBar.remove();
    
    const rejCard = $('#rejectionCard');
    if (rejCard) rejCard.remove();
    
    showSection('upload');
    
    const arc = $('#gaugeArc');
    if (arc) {
        arc.style.transition = 'none';
        arc.setAttribute('stroke-dashoffset', '534');
    }
}

/* ── Report download ────────────────────────────────────────────────────── */
function downloadCurrentReport() {
    const data = batchResults[activeResultIndex];
    if (!data || !data.report) return;
    
    const blob = new Blob(
        [JSON.stringify(data.report, null, 2)],
        { type: 'application/json' }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `verification_report_${data.filename}_${data.report.report_id || 'unknown'}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

/* ── Utilities ──────────────────────────────────────────────────────────── */
function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function capitalise(str) {
    return str.replace(/\b\w/g, c => c.toUpperCase());
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
