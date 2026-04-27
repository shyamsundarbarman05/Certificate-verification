"""
OCR Engine Module
==================
Extracts text from certificate images using EasyOCR, then parses the raw
text into structured fields (name, institution, certificate ID, dates,
course details).  Also computes per-field confidence scores and detects
missing or anomalous fields.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple


class OCREngine:
    """OCR-based text extraction and structured field parsing."""

    def __init__(self):
        self._reader = None
        self._initialized = False

    def _init_reader(self):
        """Lazy-load EasyOCR to avoid slow startup."""
        if not self._initialized:
            try:
                import easyocr
                self._reader = easyocr.Reader(
                    ["en"], gpu=False, verbose=False
                )
                self._initialized = True
            except Exception as e:
                print(f"[OCR] EasyOCR init failed: {e}")
                self._initialized = True  # prevent retry loop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(self, image: np.ndarray) -> Dict:
        """
        Run OCR on the image and return structured results.

        Returns dict with:
            raw_text        – full concatenated text
            lines           – list of {text, confidence, bbox}
            structured      – parsed fields dict
            ocr_score       – overall OCR quality score (0-100)
            field_analysis  – per-field confidence and issues
        """
        self._init_reader()

        raw_results = self._run_ocr(image)
        raw_text = " ".join(r["text"] for r in raw_results)
        avg_conf = (
            sum(r["confidence"] for r in raw_results) / len(raw_results)
            if raw_results else 0
        )

        structured = self._parse_fields(raw_text, raw_results)
        field_analysis = self._analyse_fields(structured, raw_results)

        ocr_score = self._compute_ocr_score(avg_conf, structured, field_analysis)

        return {
            "raw_text": raw_text,
            "lines": raw_results,
            "structured": structured,
            "ocr_score": round(ocr_score, 1),
            "average_confidence": round(avg_conf * 100, 1),
            "field_analysis": field_analysis,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_ocr(self, image: np.ndarray) -> List[Dict]:
        """Execute OCR and normalise output."""
        results: List[Dict] = []
        if self._reader is None:
            return results

        try:
            detections = self._reader.readtext(image)
            for bbox, text, conf in detections:
                # bbox is list of 4 corner points
                flat = [coord for pt in bbox for coord in pt]
                results.append({
                    "text": text.strip(),
                    "confidence": float(conf),
                    "bbox": [float(c) for c in flat],
                })
        except Exception as e:
            print(f"[OCR] Extraction error: {e}")

        return results

    # ------------------------------------------------------------------
    # Field parsing
    # ------------------------------------------------------------------

    # Common date patterns
    _DATE_PATTERNS = [
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{2,4})\b",
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}[\s,]+\d{2,4})\b",
        r"\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b",
    ]
    _CERT_ID_PATTERNS = [
        r"(?:certificate|cert|credential|registration|ref|id|no|number|serial)[.:\s#]*([A-Z0-9][A-Z0-9\-\/]{3,20})",
        r"\b([A-Z]{2,5}[\-\/]\d{4,12})\b",
        r"\b(\d{6,15})\b",
    ]

    def _parse_fields(self, raw_text: str, ocr_lines: List[Dict]) -> Dict:
        """Parse raw OCR text into structured certificate fields."""
        fields: Dict = {
            "name": None,
            "institution": None,
            "certificate_id": None,
            "issue_date": None,
            "course": None,
            "grade": None,
        }

        text_upper = raw_text.upper()
        lines_text = [l["text"] for l in ocr_lines]

        # --- Name extraction ---
        fields["name"] = self._extract_name(raw_text, lines_text)

        # --- Institution ---
        fields["institution"] = self._extract_institution(raw_text, lines_text)

        # --- Certificate ID ---
        fields["certificate_id"] = self._extract_cert_id(raw_text)

        # --- Dates ---
        fields["issue_date"] = self._extract_date(raw_text)

        # --- Course ---
        fields["course"] = self._extract_course(raw_text, lines_text)

        # --- Grade ---
        fields["grade"] = self._extract_grade(raw_text)

        return fields

    def _extract_name(self, raw: str, lines: List[str]) -> Optional[str]:
        """Try to find the candidate / recipient name."""
        # Look for "awarded to NAME" or "certify that NAME" patterns
        patterns = [
            r"(?:awarded?\s+to|presented?\s+to|certif(?:y|ied)\s+that|granted?\s+to|conferred?\s+(?:upon|to)|this\s+is\s+to\s+certify\s+that)\s+([A-Z][a-zA-Z\s\.]{2,40})",
            r"(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-zA-Z\s\.]{2,40})",
        ]
        for pat in patterns:
            m = re.search(pat, raw, re.IGNORECASE)
            if m:
                name = m.group(1).strip().rstrip(".,;:")
                # Basic validation – at least two words
                if len(name.split()) >= 2:
                    return name
                return name

        # Heuristic: the largest-font line near the top third is often the name
        if len(lines) >= 3:
            # Pick the line after a "certify" keyword or the 2nd-3rd line
            for i, line in enumerate(lines):
                if any(kw in line.lower() for kw in ["certify", "awarded", "presented"]):
                    if i + 1 < len(lines):
                        candidate = lines[i + 1].strip()
                        if 2 <= len(candidate) <= 50 and candidate[0].isupper():
                            return candidate

        return None

    def _extract_institution(self, raw: str, lines: List[str]) -> Optional[str]:
        patterns = [
            r"(?:university|institute|college|school|academy|board|council|organisation|organization)\s+(?:of\s+)?([A-Za-z\s&,]+)",
            r"([A-Z][A-Za-z\s&,]+(?:University|Institute|College|School|Academy|Board|Council))",
        ]
        for pat in patterns:
            m = re.search(pat, raw, re.IGNORECASE)
            if m:
                return m.group(0).strip().rstrip(".,;:")
        # Fallback: first line is often the institution header
        if lines:
            first = lines[0].strip()
            if len(first) > 5:
                return first
        return None

    def _extract_cert_id(self, raw: str) -> Optional[str]:
        for pat in self._CERT_ID_PATTERNS:
            m = re.search(pat, raw, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    def _extract_date(self, raw: str) -> Optional[str]:
        for pat in self._DATE_PATTERNS:
            m = re.search(pat, raw, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    def _extract_course(self, raw: str, lines: List[str]) -> Optional[str]:
        patterns = [
            r"(?:course|program|programme|subject|module|training|certification)\s*(?:in|on|of|:)\s*([A-Za-z\s&,\-]+)",
            r"(?:completed?|passed?|finished?)\s+(?:the\s+)?(?:course\s+)?(?:in|on)\s+([A-Za-z\s&,\-]+)",
        ]
        for pat in patterns:
            m = re.search(pat, raw, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,;:")
        return None

    def _extract_grade(self, raw: str) -> Optional[str]:
        patterns = [
            r"(?:grade|score|result|percentage|marks|gpa|cgpa)\s*[:\-]?\s*([A-Fa-f0-9\.%\+\s\/]+)",
            r"\b(distinction|merit|pass|first\s+class|second\s+class)\b",
        ]
        for pat in patterns:
            m = re.search(pat, raw, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,;:")
        return None

    # ------------------------------------------------------------------
    # Field analysis
    # ------------------------------------------------------------------

    def _analyse_fields(self, fields: Dict, ocr_lines: List[Dict]) -> Dict:
        """Check each field for presence and confidence anomalies."""
        analysis: Dict = {}
        required = ["name", "institution", "issue_date"]
        optional = ["certificate_id", "course", "grade"]

        for field in required:
            val = fields.get(field)
            if val:
                analysis[field] = {"status": "found", "value": val, "required": True}
            else:
                analysis[field] = {
                    "status": "missing",
                    "value": None,
                    "required": True,
                    "issue": f"Required field '{field}' not found",
                }

        for field in optional:
            val = fields.get(field)
            analysis[field] = {
                "status": "found" if val else "missing",
                "value": val,
                "required": False,
            }

        # Confidence anomaly: flag any line with very low confidence
        low_conf_lines = [l for l in ocr_lines if l["confidence"] < 0.4]
        if low_conf_lines:
            analysis["_low_confidence_regions"] = {
                "count": len(low_conf_lines),
                "issue": "Some text regions have very low OCR confidence – possible tampering or low quality",
            }

        return analysis

    def _compute_ocr_score(
        self, avg_conf: float, fields: Dict, analysis: Dict
    ) -> float:
        """Compute an overall OCR quality score (0-100)."""
        score = avg_conf * 100  # base from OCR confidence

        # Penalise missing required fields
        for key in ["name", "institution", "issue_date"]:
            if not fields.get(key):
                score -= 10

        # Bonus for complete data
        filled = sum(1 for v in fields.values() if v)
        score += filled * 2

        # Low-confidence penalty
        if "_low_confidence_regions" in analysis:
            score -= analysis["_low_confidence_regions"]["count"] * 3

        return max(0, min(100, score))
