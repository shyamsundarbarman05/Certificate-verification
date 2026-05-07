"""
Advanced Certificate Authenticity Verification System
=====================================================
FastAPI application that orchestrates the full verification pipeline:
  Document Upload → PDF Rendering → Image Preprocessing → OCR →
  Layout Validation → Signature/Seal Verification → Forensic Analysis →
  ML Classification → Authenticity Score → Verification Report
"""

import os
import uuid
import time
import shutil
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Directories (ensure creation) -----------------------------------------------
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")
STATIC_DIR = "static"

# Create directories at startup
for directory in [UPLOAD_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- Module imports ----------------------------------------------------------
from app.modules.document_processor import DocumentProcessor
from app.modules.image_preprocessor import ImagePreprocessor
from app.modules.ocr_engine import OCREngine
from app.modules.layout_analyzer import LayoutAnalyzer
from app.modules.signature_verifier import SignatureVerifier
from app.modules.forensic_analyzer import ForensicAnalyzer
from app.modules.ml_classifier import MLClassifier
from app.modules.report_generator import ReportGenerator
from app.modules.certificate_detector import CertificateDetector

# --- FastAPI app -------------------------------------------------------------
app = FastAPI(
    title="Certificate Authenticity Verification System",
    description="Advanced authenticity verification for pre-issued certificates",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Instantiate modules (singletons) ---------------------------------------
doc_processor = DocumentProcessor(upload_dir=UPLOAD_DIR)
img_preprocessor = ImagePreprocessor()
ocr_engine = OCREngine()
layout_analyzer = LayoutAnalyzer()
sig_verifier = SignatureVerifier()
forensic_analyzer = ForensicAnalyzer()
ml_classifier = MLClassifier()
report_generator = ReportGenerator(reports_dir=REPORTS_DIR)
cert_detector = CertificateDetector()


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=FileResponse)
async def root():
    """Serve the frontend."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def _process_verification(file: UploadFile, start_time: float) -> dict:
    """
    Internal helper to process a single file through the verification pipeline.
    """
    file_id = str(uuid.uuid4())[:12]
    file_ext = os.path.splitext(file.filename or "upload")[1].lower() or ".pdf"
    saved_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")

    # --- 1. Save uploaded file -----------------------------------------------
    try:
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        return {
            "success": False,
            "filename": file.filename,
            "detail": f"Failed to save upload: {e}"
        }

    pipeline_steps = []

    try:
        # --- 2. File validation ----------------------------------------------
        t0 = time.time()
        validation = doc_processor.validate_file(saved_path)
        if not validation["valid"]:
            return {
                "success": False,
                "file_id": file_id,
                "filename": file.filename,
                "detail": f"Invalid file: {'; '.join(validation['issues'])}",
                "rejection_type": "invalid_document"
            }
        pipeline_steps.append({"step": "File Validation", "time": round(time.time() - t0, 2), "status": "done"})

        # --- 3. Metadata extraction ------------------------------------------
        t0 = time.time()
        metadata_result = doc_processor.extract_metadata(saved_path)
        pipeline_steps.append({"step": "Metadata Extraction", "time": round(time.time() - t0, 2), "status": "done"})

        # --- 4. Render to images ---------------------------------------------
        t0 = time.time()
        images = doc_processor.render_to_images(saved_path, dpi=300)
        if not images:
            return {
                "success": False,
                "file_id": file_id,
                "filename": file.filename,
                "detail": "Could not render document to images",
                "rejection_type": "invalid_document"
            }
        pipeline_steps.append({"step": "PDF Rendering", "time": round(time.time() - t0, 2), "status": "done"})

        # Work with the first page (primary certificate page)
        primary_image = images[0]
        quality = doc_processor.get_quality_metrics(primary_image)

        # --- 4a. Quick OCR for certificate detection gate -------------------
        t0 = time.time()
        quick_ocr = ocr_engine.extract_text(primary_image)
        quick_ocr_text = quick_ocr.get("raw_text", "")
        quick_ocr_lines = quick_ocr.get("lines", [])
        pipeline_steps.append({"step": "Quick OCR Scan", "time": round(time.time() - t0, 2), "status": "done"})

        # --- 4b. Certificate format detection (PRE-GATE) --------------------
        t0 = time.time()
        detection = cert_detector.detect(
            image=primary_image,
            ocr_text=quick_ocr_text,
            ocr_lines=quick_ocr_lines,
        )
        pipeline_steps.append({"step": "Certificate Detection", "time": round(time.time() - t0, 2), "status": "done"})

        if not detection["is_certificate"]:
            total_time = round(time.time() - start_time, 2)
            rejection_type = "domestic_id" if detection.get("is_domestic_id") else "not_certificate"
            return {
                "success": False,
                "file_id": file_id,
                "filename": file.filename,
                "processing_time": total_time,
                "pipeline_steps": pipeline_steps,
                "rejection_type": rejection_type,
                "rejection_reason": detection["rejection_reason"],
                "detection_confidence": detection["confidence"],
                "detection_reasons": detection["reasons"],
                "detection_checks": detection["checks"],
            }

        # --- 5. Image preprocessing -----------------------------------------
        t0 = time.time()
        preprocess_result = img_preprocessor.preprocess(primary_image)
        pipeline_steps.append({"step": "Image Preprocessing", "time": round(time.time() - t0, 2), "status": "done"})

        # --- 6. OCR result (reuse from detection gate) -----------------------
        ocr_result = quick_ocr
        pipeline_steps.append({"step": "OCR Text Extraction", "time": 0.0, "status": "done (reused)"})

        # --- 7. Layout analysis ----------------------------------------------
        t0 = time.time()
        layout_result = layout_analyzer.analyze(
            primary_image,
            binary=preprocess_result.get("binary"),
        )
        pipeline_steps.append({"step": "Layout Validation", "time": round(time.time() - t0, 2), "status": "done"})

        # --- 8. Signature & seal verification --------------------------------
        t0 = time.time()
        sig_result = sig_verifier.verify(primary_image)
        pipeline_steps.append({"step": "Signature & Seal Verification", "time": round(time.time() - t0, 2), "status": "done"})

        # --- 9. Forensic analysis --------------------------------------------
        t0 = time.time()
        forensic_result = forensic_analyzer.analyze(primary_image)
        pipeline_steps.append({"step": "Forensic Analysis", "time": round(time.time() - t0, 2), "status": "done"})

        # --- 10. ML classification -------------------------------------------
        t0 = time.time()
        scores = {
            "ocr_score": ocr_result.get("ocr_score", 50),
            "layout_score": layout_result.get("layout_score", 50),
            "metadata_score": metadata_result.get("metadata_score", 50),
            "signature_score": sig_result.get("signature_score", 50),
            "seal_score": sig_result.get("seal_score", 50),
            "forensic_score": forensic_result.get("forensic_score", 50),
        }
        classification = ml_classifier.classify(scores)
        pipeline_steps.append({"step": "ML Classification", "time": round(time.time() - t0, 2), "status": "done"})

        # --- 11. Report generation -------------------------------------------
        t0 = time.time()
        report = report_generator.generate({
            "metadata": metadata_result,
            "ocr": ocr_result,
            "layout": layout_result,
            "signature": sig_result,
            "forensic": forensic_result,
            "classification": classification,
        })
        pipeline_steps.append({"step": "Report Generation", "time": round(time.time() - t0, 2), "status": "done"})

        total_time = round(time.time() - start_time, 2)

        # --- Sanitise numpy/non-JSON types before returning ---
        def sanitise(obj):
            """Recursively convert numpy types to Python natives."""
            import numpy as _np
            if isinstance(obj, dict):
                return {k: sanitise(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [sanitise(v) for v in obj]
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, (_np.bool_,)):
                return bool(obj)
            return obj

        return sanitise({
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "processing_time": total_time,
            "pipeline_steps": pipeline_steps,
            "quality": quality,
            "report": report,
        })

    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "filename": file.filename,
            "detail": f"Verification pipeline failed: {str(e)}"
        }
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(saved_path):
                os.remove(saved_path)
        except Exception:
            pass


@app.post("/api/verify")
async def verify_certificate(file: UploadFile = File(...)):
    """
    Full verification pipeline for a single file.
    """
    start_time = time.time()
    result = await _process_verification(file, start_time)
    
    if not result["success"]:
        # If it's a structural rejection (from detector), return 400 with details
        if "rejection_type" in result or "rejection_reason" in result:
             return JSONResponse(status_code=400, content=result)
        # Otherwise it's a server error
        raise HTTPException(status_code=500, detail=result.get("detail", "Unknown error"))
    
    return JSONResponse(content=result)


@app.post("/api/verify-batch")
async def verify_batch(files: list[UploadFile] = File(...)):
    """
    Batch verification pipeline for multiple files.
    """
    results = []
    for file in files:
        start_time = time.time()
        result = await _process_verification(file, start_time)
        results.append(result)
    
    return JSONResponse(content={
        "success": True,
        "results": results
    })



@app.get("/api/report/{report_id}")
async def get_report(report_id: str):
    """Retrieve a previously generated report by ID."""
    file_path = os.path.join(REPORTS_DIR, f"report_{report_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(file_path, media_type="application/json")


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "Certificate Authenticity Verification System"}
