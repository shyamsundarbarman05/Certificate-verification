# CertVerify — Complete Project Reference & Documentation

**Version:** 1.0.0  
**Last Updated:** May 6, 2026  
**System:** Advanced Certificate Authenticity Verification System

---

## TABLE OF CONTENTS

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Directory Structure](#directory-structure)
4. [Files & Modules](#files--modules)
5. [API Endpoints](#api-endpoints)
6. [Database Schema](#database-schema)
7. [Configuration](#configuration)
8. [Deployment & Setup](#deployment--setup)
9. [Module Documentation](#module-documentation)
10. [Verification Pipeline](#verification-pipeline)
11. [Error Handling](#error-handling)
12. [Security](#security)

---

## PROJECT OVERVIEW

### Purpose

Advanced authenticity verification system for pre-issued certificates using document forensics, optical character recognition (OCR), and machine learning ensemble methods.

### Key Features

- Multi-file batch processing
- Real-time pipeline visualization
- Forensic tampering detection
- Digital signature & seal verification
- OCR text extraction with confidence scoring
- Weighted ML ensemble classification
- JSON report generation
- Responsive web interface

### System Architecture

```
Frontend (HTML5/CSS3/JS) → FastAPI Backend → Python Analysis Modules → JSON Reports
```

---

## TECH STACK

### Backend

| Technology           | Purpose                       | Version                |
| -------------------- | ----------------------------- | ---------------------- |
| **FastAPI**          | Modern async web framework    | Latest                 |
| **Uvicorn**          | ASGI server                   | [standard]             |
| **PyMuPDF (fitz)**   | PDF processing & rendering    | Latest                 |
| **OpenCV**           | Image processing (headless)   | opencv-python-headless |
| **NumPy**            | Numerical computing           | Latest                 |
| **Pillow**           | Image I/O & manipulation      | Latest                 |
| **EasyOCR**          | Optical character recognition | Latest                 |
| **scikit-learn**     | ML utilities & preprocessing  | Latest                 |
| **scikit-image**     | Advanced image analysis       | Latest                 |
| **XGBoost**          | Gradient boosting classifier  | Latest                 |
| **Jinja2**           | Report templating             | Latest                 |
| **aiofiles**         | Async file operations         | Latest                 |
| **python-multipart** | Form data parsing             | Latest                 |

### Frontend

| Technology            | Purpose                                    |
| --------------------- | ------------------------------------------ |
| **HTML5**             | Semantic markup                            |
| **CSS3**              | Styling, animations, responsiveness        |
| **JavaScript (ES6+)** | Client-side logic (vanilla, no frameworks) |
| **Fetch API**         | HTTP communication                         |
| **SVG**               | Charts & visualization                     |

### Storage

| Type      | Location               | Format                         |
| --------- | ---------------------- | ------------------------------ |
| Uploads   | `uploads/`             | PDF, JPG, PNG, TIFF, BMP, WebP |
| Reports   | `reports/`             | JSON                           |
| Templates | `reference_templates/` | Various formats                |
| Static    | `static/`              | HTML, CSS, JS                  |

---

## DIRECTORY STRUCTURE

```
Certificate Verification/
│
├── app/                                 # Main FastAPI application
│   ├── __init__.py                     # Package initialization
│   ├── main.py                         # FastAPI orchestrator & routes
│   │
│   └── modules/                        # Verification pipeline modules
│       ├── __init__.py
│       ├── certificate_detector.py     # Pre-gate: Is document a certificate?
│       ├── document_processor.py       # PDF/image I/O, metadata extraction
│       ├── image_preprocessor.py       # Image enhancement & normalization
│       ├── ocr_engine.py               # EasyOCR wrapper for text extraction
│       ├── layout_analyzer.py          # Text layout & positioning analysis
│       ├── signature_verifier.py       # Signature & seal detection
│       ├── forensic_analyzer.py        # Tampering & artifact detection
│       ├── ml_classifier.py            # Ensemble scoring & classification
│       ├── report_generator.py         # JSON report synthesis
│       └── __pycache__/                # Python cache
│
├── static/                              # Frontend web assets
│   ├── index.html                      # Main HTML template
│   ├── css/
│   │   └── style.css                   # All styling & animations
│   └── js/
│       └── app.js                      # Frontend JavaScript logic
│
├── uploads/                            # Temporary uploaded files (auto-created)
├── reports/                            # Generated verification reports (auto-created)
├── reference_templates/                # Certificate template references (optional)
│
├── requirements.txt                    # Python dependencies
├── start.bat                           # Windows startup script
└── PROJECT_COMPLETE_REFERENCE.md       # This file

```

---

## FILES & MODULES

### Core Application Files

#### [app/main.py](app/main.py)

**Purpose:** FastAPI application orchestrator  
**Size:** ~400 lines  
**Key Components:**

- FastAPI app initialization
- CORS middleware configuration
- Static file serving (`/static`)
- Module instantiation (singleton pattern)
- Route handlers

**Routes:**

```python
GET  /                          # Serve frontend
POST /api/verify                # Single file verification
POST /api/verify-batch          # Batch file verification
```

**Key Functions:**

- `root()` — Serve index.html
- `_process_verification(file, start_time)` — Internal pipeline orchestrator
- `verify_certificate(file)` — Single file endpoint
- `verify_batch(files)` — Batch endpoint

---

#### [app/modules/document_processor.py](app/modules/document_processor.py)

**Purpose:** Document I/O, validation, metadata extraction  
**Dependencies:** `fitz (PyMuPDF)`, `PIL`, `numpy`

**Class:** `DocumentProcessor`

**Methods:**
| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `__init__(upload_dir)` | path | self | Initialize with upload directory |
| `validate_file(file_path)` | path | dict | Check format, size, existence |
| `extract_metadata(file_path)` | path | dict | Extract PDF metadata & flags |
| `render_to_images(file_path, dpi)` | path, int | list[PIL.Image] | Convert PDF pages to images |
| `get_quality_metrics(image)` | PIL.Image | dict | Calculate image quality scores |

**Supported Formats:** `.pdf`, `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp`, `.webp`

**Max File Size:** 50 MB

**Suspicious Software Detection:**

```python
photoshop, gimp, paint.net, illustrator, inkscape, canva, pixlr, affinity, corel, paintshop
```

**Trusted Producers:**

```python
microsoft, adobe acrobat, libreoffice, openoffice, google docs, latex, tex, quartz
```

---

#### [app/modules/image_preprocessor.py](app/modules/image_preprocessor.py)

**Purpose:** Image enhancement and normalization  
**Dependencies:** `opencv-python-headless`, `PIL`, `numpy`, `scikit-image`

**Class:** `ImagePreprocessor`

**Methods:**
| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `preprocess(image)` | PIL.Image | dict | Full preprocessing pipeline |
| `normalize_contrast(image)` | np.array | np.array | CLAHE contrast enhancement |
| `denoise(image)` | np.array | np.array | Bilateral filtering |
| `binarize(image)` | np.array | np.array | Otsu thresholding |

**Pipeline:**

1. Convert to grayscale
2. CLAHE contrast enhancement
3. Bilateral denoising
4. Otsu thresholding
5. Morphological operations

---

#### [app/modules/ocr_engine.py](app/modules/ocr_engine.py)

**Purpose:** Text extraction from images  
**Dependencies:** `easyocr`

**Class:** `OCREngine`

**Methods:**
| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `extract_text(image)` | PIL.Image | dict | Extract text with confidence |
| `compute_ocr_score(results)` | dict | float | Calculate OCR quality score |

**Output Fields:**

```python
{
    "raw_text": str,              # Full extracted text
    "lines": list[dict],          # Line-by-line results with confidence
    "ocr_score": float,           # Quality score (0-100)
    "average_confidence": float   # Mean confidence across all lines
}
```

---

#### [app/modules/certificate_detector.py](app/modules/certificate_detector.py)

**Purpose:** Pre-gate validation — verify document is a certificate  
**Dependencies:** `numpy`, `PIL`

**Class:** `CertificateDetector`

**Methods:**
| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `detect(image, ocr_text, ocr_lines)` | PIL.Image, str, list | dict | Certificate classification |

**Rejection Types:**

- `is_certificate: False` → Not a certificate
- `is_domestic_id: True` → Detected as domestic ID
- Other document types → Not a certificate

**Output Fields:**

```python
{
    "is_certificate": bool,
    "is_domestic_id": bool,
    "confidence": float,
    "rejection_reason": str,
    "reasons": list[str],
    "checks": dict  # Individual check results
}
```

**Checks Performed:**

- Certificate keywords detection
- Layout pattern matching
- Text density analysis
- Domestic ID keyword detection
- Confidence thresholding

---

#### [app/modules/layout_analyzer.py](app/modules/layout_analyzer.py)

**Purpose:** Text layout and positioning validation  
**Dependencies:** `opencv-python-headless`, `scikit-image`, `numpy`

**Class:** `LayoutAnalyzer`

**Methods:**
| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `analyze(image, binary)` | PIL.Image, np.array | dict | Full layout analysis |
| `detect_text_blocks(binary)` | np.array | list[tuple] | Find text regions |
| `check_alignment(image, binary)` | PIL.Image, np.array | dict | Verify text alignment |
| `check_margins(binary)` | np.array | dict | Validate margins |
| `compute_layout_score(checks)` | dict | float | Aggregate score |

**Findings:**

- Text block centering
- Margin consistency
- Line spacing uniformity
- Professional formatting

---

#### [app/modules/signature_verifier.py](app/modules/signature_verifier.py)

**Purpose:** Digital signature and seal detection  
**Dependencies:** `opencv-python-headless`, `numpy`, `PIL`

**Class:** `SignatureVerifier`

**Methods:**
| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `verify(image)` | PIL.Image | dict | Detect signatures & seals |
| `detect_signatures(image)` | PIL.Image | dict | Find signature patterns |
| `detect_seals(image)` | PIL.Image | dict | Find seal stamps |
| `compute_scores(sig_result, seal_result)` | dict, dict | dict | Generate scores |

**Output Fields:**

```python
{
    "signature_score": float,     # Signature authenticity (0-100)
    "seal_score": float,          # Seal verification (0-100)
    "signatures_found": int,
    "seals_found": int,
    "findings": list[str]
}
```

---

#### [app/modules/forensic_analyzer.py](app/modules/forensic_analyzer.py)

**Purpose:** Tampering and artifact detection  
**Dependencies:** `opencv-python-headless`, `numpy`, `PIL`, `scikit-image`

**Class:** `ForensicAnalyzer`

**Methods:**
| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `analyze(image)` | PIL.Image | dict | Full forensic analysis |
| `detect_copy_paste_artifacts(image)` | PIL.Image | dict | Find duplicated regions |
| `detect_image_manipulation(image)` | PIL.Image | dict | Detect editing signs |
| `compute_forensic_score(findings)` | dict | float | Aggregate score |

**Forensic Checks:**

- Copy-paste artifact detection (template matching)
- Image noise patterns
- Compression artifacts
- Color channel anomalies
- Edge inconsistencies

**Output Fields:**

```python
{
    "forensic_score": float,
    "copy_paste_regions": int,    # Number of suspicious matches
    "manipulation_indicators": list[str],
    "risk_level": str              # "low" | "medium" | "high"
}
```

---

#### [app/modules/ml_classifier.py](app/modules/ml_classifier.py)

**Purpose:** Ensemble scoring and authenticity classification  
**Dependencies:** `numpy`, `scikit-learn`, `xgboost` (optional)

**Class:** `MLClassifier`

**Mode 1: Heuristic (Default)**

```
Weighted ensemble of 6 module scores
```

**Mode 2: XGBoost (When model available)**

```
Trained ML model path: models/xgb_classifier.json
```

**Methods:**
| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `classify(scores_dict)` | dict | dict | Classify authenticity |
| `_load_model()` | N/A | None | Load XGBoost if available |
| `_heuristic_classify(feature_vector)` | list | dict | Fallback heuristic |

**Score Weights (Heuristic Mode):**

```python
{
    "ocr_score":       0.15,    # 15%
    "layout_score":    0.15,    # 15%
    "metadata_score":  0.15,    # 15%
    "signature_score": 0.15,    # 15%
    "seal_score":      0.10,    # 10%
    "forensic_score":  0.30     # 30% (highest weight)
}
```

**Risk Thresholds:**

```python
score >= 75      → "Genuine"         (authentic)
45 <= score < 75 → "Suspicious"      (requires review)
score < 45       → "Likely Forged"   (reject)
```

**Output Fields:**

```python
{
    "authenticity_score": float,
    "risk_level": str,
    "classification": str,
    "confidence": float,
    "method": str,              # "heuristic_ensemble" or "xgboost"
    "feature_vector": list,
    "module_contributions": dict
}
```

---

#### [app/modules/report_generator.py](app/modules/report_generator.py)

**Purpose:** Comprehensive JSON report synthesis  
**Dependencies:** `jinja2`, `json`

**Class:** `ReportGenerator`

**Methods:**
| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `__init__(reports_dir)` | str | self | Initialize with report directory |
| `generate(pipeline_results)` | dict | dict | Synthesize complete report |
| `_aggregate_findings(results)` | dict | list | Consolidate all findings |
| `_save_report(report_dict)` | dict | str | Save to disk, return report_id |

**Report Structure:**

```json
{
  "report_id": "1941f4c1-e97",
  "timestamp": "2026-05-06T16:34:53.106847+00:00",
  "authenticity_score": 87.5,
  "risk_level": "Genuine",
  "confidence": 0.843,
  "method": "heuristic_ensemble",
  "summary": "Certificate authenticity assessment complete...",
  "module_scores": {
    "OCR Quality": 85.0,
    "Layout Consistency": 80.0,
    "Metadata Integrity": 100,
    "Signature Authenticity": 95.0,
    "Seal Verification": 95.0,
    "Forensic Analysis": 80.0
  },
  "findings": [
    {
      "module": "OCR Analysis",
      "severity": "warning|critical",
      "description": "...",
      "score_impact": "Negative|Positive"
    }
  ],
  "extracted_data": {
    "text_fields": {
      "name": null,
      "institution": "...",
      "certificate_id": "...",
      "issue_date": "04/22/2026",
      "course": null,
      "grade": null
    },
    "average_ocr_confidence": 89.0
  },
  "metadata": {
    "format": "PDF",
    "page_count": 1,
    "creator": "...",
    "producer": "...",
    "suspicious_flags": []
  },
  "classification_details": {
    "authenticity_score": 87.5,
    "risk_level": "Genuine",
    "classification": "Genuine",
    "confidence": 0.843,
    "method": "heuristic_ensemble",
    "feature_vector": [85.0, 80.0, 100.0, 95.0, 95.0, 80.0],
    "module_contributions": {
      "ocr_score": { "raw": 85.0, "weight": 0.15, "weighted": 12.8 },
      "layout_score": { "raw": 80.0, "weight": 0.15, "weighted": 12.0 },
      "metadata_score": { "raw": 100.0, "weight": 0.15, "weighted": 15.0 },
      "signature_score": { "raw": 95.0, "weight": 0.15, "weighted": 14.2 },
      "seal_score": { "raw": 95.0, "weight": 0.1, "weighted": 9.5 },
      "forensic_score": { "raw": 80.0, "weight": 0.3, "weighted": 24.0 }
    }
  }
}
```

---

### Frontend Files

#### [static/index.html](static/index.html)

**Purpose:** Main HTML template  
**Size:** ~200 lines (with inline structure)

**Sections:**

```html
<header>              <!-- Logo, status indicator -->
<main>
  <section#uploadSection>      <!-- File upload UI -->
  <section#pipelineSection>    <!-- Real-time pipeline viz -->
  <section#resultsSection>     <!-- Results dashboard -->
</main>
<footer>              <!-- Attribution -->
<script>              <!-- Load app.js -->
```

**Key Elements:**

- Drag-drop upload zone
- File preview list
- Action buttons (Add More, Verify)
- Pipeline step visualizer
- Result navigator (batch mode)
- Verdict card with gauge
- Module scores grid
- Tabbed results (Findings, Extracted Data, Metadata, Pipeline)

---

#### [static/css/style.css](static/css/style.css)

**Purpose:** All styling and animations

**Features:**

- Responsive design (mobile/tablet/desktop)
- CSS Grid for layouts
- Flexbox for components
- Animated background orbs
- Gradient backgrounds
- Smooth transitions
- Dark/light theme support
- Gauge chart styles
- Progress bar animations

**Key Classes:**

```
.header                -- Top navigation bar
.upload-zone           -- Drag-drop upload area
.file-preview-list     -- Selected files display
.pipeline              -- Step-by-step visualizer
.pipeline__bar         -- Progress bar
.verdict-card          -- Main result display
.scores-grid           -- Module scores grid
.tab-panel             -- Tabbed content
.btn                   -- Button styles
.section               -- Content sections
```

---

#### [static/js/app.js](static/js/app.js)

**Purpose:** Frontend JavaScript logic  
**Size:** ~400+ lines

**Key Features:**

- File upload handling (drag-drop & click)
- Form validation
- Batch file management
- Fetch API calls
- Real-time pipeline visualization
- Results rendering
- Tab switching
- Gauge chart drawing (SVG)
- Error handling & user feedback

**Global State:**

```javascript
selectedFiles = []; // Array of File objects
batchResults = []; // Array of result objects
activeResultIndex = 0; // Currently displayed result
```

**Pipeline Steps (Frontend):**

```javascript
PIPELINE_STEPS = [
  { id: "validate", label: "File Validation" },
  { id: "metadata", label: "Metadata Extraction" },
  { id: "render", label: "PDF Rendering" },
  { id: "preprocess", label: "Image Preprocessing" },
  { id: "ocr", label: "OCR Text Extraction" },
  { id: "layout", label: "Layout Validation" },
  { id: "signature", label: "Signature & Seal" },
  { id: "forensic", label: "Forensic Analysis" },
  { id: "classify", label: "ML Classification" },
  { id: "report", label: "Report Generation" },
];
```

**Key Functions:**

- `init()` — DOMContentLoaded handler
- `setupUploadZone()` — Configure drag-drop
- `handleFileSelect(files)` — Process selected files
- `renderFilePreview()` — Show selected files
- `submitVerification()` — POST to backend
- `updatePipelineVisualization(result)` — Animate progress
- `renderResults(report)` — Display verdict & details
- `drawGaugeChart(score)` — SVG gauge visualization
- `switchTab(tabName)` — Navigate result tabs

---

### Configuration Files

#### [requirements.txt](requirements.txt)

**Purpose:** Python dependencies

**Dependencies:**

```
fastapi              # Web framework
uvicorn[standard]    # ASGI server with extensions
python-multipart     # Form data parsing
pymupdf              # PDF processing (as 'fitz')
opencv-python-headless  # Image processing (no display)
numpy                # Numerical computing
Pillow               # Image I/O
easyocr              # OCR engine
scikit-learn         # ML utilities
scikit-image         # Advanced image analysis
xgboost              # Gradient boosting
jinja2               # Templating
aiofiles             # Async file I/O
```

---

#### [start.bat](start.bat)

**Purpose:** Windows startup script

**Operations:**

1. Display header
2. Create directories (`uploads/`, `reports/`, `reference_templates/`)
3. Install dependencies
4. Start server on `localhost:8003`

**Usage:**

```bash
start.bat
```

---

## API ENDPOINTS

### Base URL

```
http://localhost:8003
```

### Routes

#### GET `/`

**Purpose:** Serve frontend  
**Response:** HTML (index.html)

---

#### POST `/api/verify`

**Purpose:** Verify single certificate file

**Request:**

```
Content-Type: multipart/form-data
Body: file (single file)
```

**Accepted Formats:** `.pdf`, `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp`, `.webp`  
**Max Size:** 50 MB

**Response (Success - 200):**

```json
{
  "success": true,
  "file_id": "a1b2c3d4e5f6",
  "filename": "certificate.pdf",
  "processing_time": 12.45,
  "pipeline_steps": [
    {"step": "File Validation", "time": 0.05, "status": "done"},
    {"step": "Metadata Extraction", "time": 0.1, "status": "done"},
    ...
  ],
  "quality": {...},
  "report": {...}
}
```

**Response (Structural Rejection - 400):**

```json
{
  "success": false,
  "file_id": "a1b2c3d4e5f6",
  "filename": "document.jpg",
  "rejection_type": "not_certificate",
  "rejection_reason": "Detected as domestic ID, not a certificate",
  "detection_confidence": 0.92,
  "detection_reasons": [...]
}
```

**Response (Server Error - 500):**

```json
{
  "detail": "Verification pipeline failed: {error message}"
}
```

---

#### POST `/api/verify-batch`

**Purpose:** Verify multiple certificate files

**Request:**

```
Content-Type: multipart/form-data
Body: files (multiple files)
```

**Response (Success - 200):**

```json
{
  "success": true,
  "results": [
    {...result 1...},
    {...result 2...},
    {...result 3...}
  ]
}
```

---

## DATABASE SCHEMA

### Report Storage (JSON)

**Location:** `reports/report_<report_id>.json`

**Schema:**

```json
{
  "report_id": "uuid",
  "timestamp": "ISO8601",
  "authenticity_score": 0-100,
  "risk_level": "Genuine|Suspicious|Likely Forged",
  "confidence": 0-1,
  "method": "heuristic_ensemble|xgboost",
  "summary": "string",
  "module_scores": {
    "OCR Quality": 0-100,
    "Layout Consistency": 0-100,
    "Metadata Integrity": 0-100,
    "Signature Authenticity": 0-100,
    "Seal Verification": 0-100,
    "Forensic Analysis": 0-100
  },
  "findings": [
    {
      "module": "string",
      "severity": "warning|critical",
      "description": "string",
      "score_impact": "Negative|Positive"
    }
  ],
  "extracted_data": {
    "text_fields": {
      "name": "string|null",
      "institution": "string|null",
      "certificate_id": "string|null",
      "issue_date": "string|null",
      "course": "string|null",
      "grade": "string|null"
    },
    "average_ocr_confidence": 0-100
  },
  "metadata": {
    "format": "PDF|JPG|PNG|...",
    "page_count": "int",
    "creator": "string",
    "producer": "string",
    "suspicious_flags": [string]
  },
  "classification_details": {
    "authenticity_score": 0-100,
    "risk_level": "Genuine|Suspicious|Likely Forged",
    "classification": "Genuine|Suspicious|Forged",
    "confidence": 0-1,
    "method": "string",
    "feature_vector": [number, number, number, number, number, number],
    "module_contributions": {
      "ocr_score": {"raw": number, "weight": number, "weighted": number},
      "layout_score": {...},
      "metadata_score": {...},
      "signature_score": {...},
      "seal_score": {...},
      "forensic_score": {...}
    }
  }
}
```

---

## CONFIGURATION

### Environment Variables

None currently required. All configuration is hardcoded in `main.py`.

### Directories (Auto-created)

```
uploads/              # Temporary uploaded files
reports/              # Generated JSON reports
reference_templates/  # Certificate templates (optional)
```

### Ports

```
Frontend: http://localhost:8003
API: http://localhost:8003/api
```

### CORS Policy

```
allow_origins: ["*"]       # All origins allowed
allow_methods: ["*"]       # All HTTP methods
allow_headers: ["*"]       # All headers
```

---

## DEPLOYMENT & SETUP

### Requirements

- Windows (for `start.bat`)
- Python 3.8+
- pip package manager
- ~500MB disk space (models, dependencies)
- ~2GB RAM minimum

### Installation Steps

1. **Clone/Download project**

   ```
   c:\Users\shyam\Desktop\Certificate verification
   ```

2. **Run startup script**

   ```bash
   start.bat
   ```

   This will:
   - Create required directories
   - Install all Python dependencies
   - Start the server

3. **Access web interface**
   ```
   http://localhost:8003
   ```

### Manual Startup (Alternative)

```bash
# Navigate to project
cd "c:\Users\shyam\Desktop\Certificate verification"

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir uploads reports reference_templates

# Start server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8003
```

### Logs

Console output shows:

```
INFO:     Uvicorn running on http://0.0.0.0:8003
INFO:     Application startup complete
```

### Stopping the Server

Press `Ctrl+C` in the terminal

---

## MODULE DOCUMENTATION

### DocumentProcessor

**Validates & processes uploaded documents**

```python
from app.modules.document_processor import DocumentProcessor

proc = DocumentProcessor(upload_dir="uploads")

# Validate
validation = proc.validate_file("document.pdf")
# → {"valid": True, "file_type": "pdf", ...}

# Extract metadata
metadata = proc.extract_metadata("document.pdf")
# → {"creator": "Microsoft Word", "producer": "...", ...}

# Render to images
images = proc.render_to_images("document.pdf", dpi=300)
# → [PIL.Image, PIL.Image, ...]

# Quality metrics
quality = proc.get_quality_metrics(images[0])
# → {"brightness": 150, "contrast": 80, ...}
```

---

### ImagePreprocessor

**Enhances image quality for analysis**

```python
from app.modules.image_preprocessor import ImagePreprocessor

preprocessor = ImagePreprocessor()

result = preprocessor.preprocess(image)
# → {"preprocessed": PIL.Image, "binary": np.array, ...}
```

---

### OCREngine

**Extracts text using EasyOCR**

```python
from app.modules.ocr_engine import OCREngine

ocr = OCREngine()

result = ocr.extract_text(image)
# → {
#     "raw_text": "Certificate of Completion...",
#     "lines": [
#       {"text": "Line 1", "confidence": 0.95},
#       ...
#     ],
#     "ocr_score": 85.0,
#     "average_confidence": 0.89
#   }
```

---

### LayoutAnalyzer

**Validates text layout & positioning**

```python
from app.modules.layout_analyzer import LayoutAnalyzer

analyzer = LayoutAnalyzer()

result = analyzer.analyze(image, binary)
# → {
#     "layout_score": 80.0,
#     "text_blocks": [...],
#     "alignment_issues": [...],
#     "margin_issues": [...]
#   }
```

---

### SignatureVerifier

**Detects signatures & seals**

```python
from app.modules.signature_verifier import SignatureVerifier

verifier = SignatureVerifier()

result = verifier.verify(image)
# → {
#     "signature_score": 95.0,
#     "seal_score": 95.0,
#     "signatures_found": 1,
#     "seals_found": 1
#   }
```

---

### ForensicAnalyzer

**Detects tampering & artifacts**

```python
from app.modules.forensic_analyzer import ForensicAnalyzer

forensic = ForensicAnalyzer()

result = forensic.analyze(image)
# → {
#     "forensic_score": 80.0,
#     "copy_paste_regions": 0,
#     "manipulation_indicators": [],
#     "risk_level": "low"
#   }
```

---

### MLClassifier

**Classifies authenticity**

```python
from app.modules.ml_classifier import MLClassifier

classifier = MLClassifier()

scores = {
    "ocr_score": 85.0,
    "layout_score": 80.0,
    "metadata_score": 100,
    "signature_score": 95.0,
    "seal_score": 95.0,
    "forensic_score": 80.0,
}

result = classifier.classify(scores)
# → {
#     "authenticity_score": 87.5,
#     "risk_level": "Genuine",
#     "classification": "Genuine",
#     "confidence": 0.843,
#     "method": "heuristic_ensemble",
#     "feature_vector": [85, 80, 100, 95, 95, 80],
#     "module_contributions": {...}
#   }
```

---

### ReportGenerator

**Synthesizes JSON reports**

```python
from app.modules.report_generator import ReportGenerator

gen = ReportGenerator(reports_dir="reports")

report = gen.generate(pipeline_results_dict)
# → Full report JSON (saved to disk)
# → Returns report dict with report_id
```

---

## VERIFICATION PIPELINE

### Sequential Processing Order

```
1. File Validation          [DocumentProcessor.validate_file()]
2. Metadata Extraction      [DocumentProcessor.extract_metadata()]
3. PDF Rendering            [DocumentProcessor.render_to_images()]
4. Image Preprocessing      [ImagePreprocessor.preprocess()]
5. Quick OCR Scan           [OCREngine.extract_text()]
6. Certificate Detection    [CertificateDetector.detect()]  ← PRE-GATE
7. Layout Analysis          [LayoutAnalyzer.analyze()]
8. Signature & Seal Verify  [SignatureVerifier.verify()]
9. Forensic Analysis        [ForensicAnalyzer.analyze()]
10. ML Classification       [MLClassifier.classify()]
11. Report Generation       [ReportGenerator.generate()]
```

### Timing Example

```
Stage                       Time (seconds)
─────────────────────────   ─────────────
File Validation             0.05
Metadata Extraction         0.1
PDF Rendering               0.8
Image Preprocessing         0.3
Quick OCR Scan              2.5
Certificate Detection       0.2
Layout Analysis             0.4
Signature Verification      0.8
Forensic Analysis           1.2
ML Classification           0.05
Report Generation           0.1
─────────────────────────   ─────────────
TOTAL                       ~6.5 seconds
```

---

## ERROR HANDLING

### Pipeline Errors

| Error Type          | Status | Response                                  |
| ------------------- | ------ | ----------------------------------------- |
| Invalid file format | 400    | `{success: false, detail: "..."}`         |
| File too large      | 400    | `{success: false, detail: "..."}`         |
| File corruption     | 500    | `{detail: "..."}`                         |
| Not a certificate   | 400    | `{success: false, rejection_type: "..."}` |
| PDF rendering fails | 500    | `{detail: "..."}`                         |
| OCR failure         | 500    | `{detail: "..."}`                         |
| Pipeline exception  | 500    | `{detail: "..."}`                         |

### Cleanup

Uploaded files are automatically deleted after processing (success or failure).

---

## SECURITY

### Validation Gates

1. **File Type Gate** — Only supported formats allowed
2. **File Size Gate** — Max 50 MB per file
3. **Integrity Gate** — File must be readable
4. **Certificate Detection Gate** — Must be actual certificate (rejects fake documents)
5. **Metadata Gate** — Checks for suspicious software signatures
6. **Forensic Gate** — Detects tampering/manipulation

### Suspicious Software Detection

```
photoshop, gimp, paint.net, illustrator, inkscape,
canva, pixlr, affinity, corel, paintshop
```

When detected in PDF metadata → Flag for review

### CORS

All origins allowed (frontend can be served from any domain).

### Input Sanitization

All user inputs passed through validation before processing. Results sanitized for JSON serialization (numpy types converted to Python natives).

---

## TROUBLESHOOTING

### Server Won't Start

```
Error: "Port 8003 already in use"
Solution: Change port in start.bat or kill process on 8003
```

### Missing Dependencies

```
Error: "No module named 'easyocr'"
Solution: pip install -r requirements.txt
```

### File Upload Fails

```
Error: "File size too large"
Solution: Reduce file size (<50MB) or compress document
```

### OCR Low Confidence

```
Solution: Use high-DPI scans (300+ DPI recommended)
```

### Pipeline Slow

```
Optimize: Reduce image resolution, use faster hardware
```

---

## PERFORMANCE METRICS

### Typical Processing Times

- **Small PDF** (1 page, 500KB): 4-8 seconds
- **Medium PDF** (5 pages, 5MB): 10-15 seconds
- **High-res image** (2MB JPG): 6-10 seconds

### Resource Usage

- **CPU**: Moderate (multi-threaded EasyOCR)
- **Memory**: 200-500MB per process
- **Disk**: ~100MB per analysis (temporary)

### Concurrency

Currently processes files sequentially. For batch operations, consider:

- Async task queue (Celery)
- Multi-worker deployment
- Load balancing

---

## VERSION HISTORY

| Version | Date       | Changes         |
| ------- | ---------- | --------------- |
| 1.0.0   | 2026-05-06 | Initial release |

---

## CONTACT & SUPPORT

**System:** CertVerify v1.0.0  
**Last Updated:** May 6, 2026  
**Maintained by:** Development Team

For issues or feature requests, contact the development team.

---

**END OF DOCUMENT**
