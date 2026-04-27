"""
Certificate Format Detector Module
====================================
Determines whether an uploaded document is actually a certificate (or at
least a formal document in proper pro-forma) before the full verification
pipeline runs.

Checks performed:
  0. **Domestic ID rejection** — immediately discards Aadhaar, Voter ID,
     PAN Card, Driving License, Passport, Ration Card, etc.
  1. Certificate keyword matching — looks for terms like "certificate",
     "awarded", "completion", "hereby certify", etc.
  2. Layout structure — checks for borders, centered content, headings
  3. Handwriting detection — high edge density + irregular line spacing
     suggests handwritten notes, not a printed certificate
  4. Text density — certificates have moderate text; handwritten pages
     may have very different density patterns
  5. Aspect ratio & formatting — certificates typically have standard
     document aspect ratios

Returns a classification with confidence and rejection reason.
"""

import cv2
import numpy as np
import re
from typing import Dict, List, Tuple


class CertificateDetector:
    """Detect whether a document is a certificate in proper pro-forma."""

    # ── Domestic / Government Document keywords (REJECT immediately) ─────
    # Each entry is (regex pattern, human-readable label)
    DOMESTIC_ID_KEYWORDS = [
        # ── Identity Cards ──────────────────────────────────────────────
        # Aadhaar
        (r"\baa+dh?a+r\b", "Aadhaar Card"),
        (r"\buidai\b", "Aadhaar Card"),
        (r"\bunique\s+identification\b", "Aadhaar Card"),
        (r"\b\d{4}\s+\d{4}\s+\d{4}\b", "Aadhaar Card"),
        (r"\benrolment\s*no\b", "Aadhaar Card"),
        # Voter ID / EPIC
        (r"\bvoter\s*(?:id|card|i\.?d)\b", "Voter ID"),
        (r"\belection\s+commission\b", "Voter ID"),
        (r"\belectoral\s+(?:roll|photo)\b", "Voter ID"),
        (r"\bepic\s*(?:no|number)?\b", "Voter ID"),
        (r"\belectors?\s+photo\b", "Voter ID"),
        # PAN Card
        (r"\bpermanent\s+account\s+number\b", "PAN Card"),
        (r"\bpan\s*(?:card|no|number)\b", "PAN Card"),
        (r"\bincome\s*tax\s*department\b", "PAN Card"),
        # Driving License
        (r"\bdriving\s+licen[cs]e\b", "Driving License"),
        (r"\bdriver'?s?\s+licen[cs]e\b", "Driving License"),
        (r"\bmotor\s+vehicle\b", "Driving License"),
        (r"\btransport\s+(?:authority|department|commissioner)\b", "Driving License"),
        (r"\bdl\s*(?:no|number)\b", "Driving License"),
        (r"\brto\b", "Driving License"),
        # Passport  (must be more specific — "passport" alone appears in CVs)
        (r"\bpassport\s*(?:no|number|issuing|authority)\b", "Passport"),
        (r"\bplace\s+of\s+issue\b", "Passport"),
        (r"\bimmigration\s+authority\b", "Passport"),
        # Ration Card
        (r"\bration\s+card\b", "Ration Card"),
        (r"\bfair\s+price\s+shop\b", "Ration Card"),
        (r"\bpublic\s+distribution\b", "Ration Card"),
        (r"\bbpl\b", "Ration Card"),
        (r"\bapl\b", "Ration Card"),
        # ── Government Certificates ─────────────────────────────────────
        # Birth Certificate
        (r"\bbirth\s+certificate\b", "Birth Certificate"),
        (r"\bregistrar\s+of\s+births?\b", "Birth Certificate"),
        (r"\bcertificate\s+of\s+birth\b", "Birth Certificate"),
        (r"\bplace\s+of\s+birth\b", "Birth Certificate"),
        # Death Certificate
        (r"\bdeath\s+certificate\b", "Death Certificate"),
        (r"\bregistrar\s+of\s+deaths?\b", "Death Certificate"),
        (r"\bcertificate\s+of\s+death\b", "Death Certificate"),
        (r"\bcause\s+of\s+death\b", "Death Certificate"),
        # Caste / Category Certificate
        (r"\bcaste\s+certificate\b", "Caste Certificate"),
        (r"\bscheduled\s+caste\b", "Caste Certificate"),
        (r"\bscheduled\s+tribe\b", "Caste Certificate"),
        (r"\bother\s+backward\s+class(?:es)?\b", "OBC Certificate"),
        (r"\bobc\s+certificate\b", "OBC Certificate"),
        (r"\bsc\s*/?\s*st\b", "SC/ST Certificate"),
        (r"\bcategory\s+certificate\b", "Caste Certificate"),
        (r"\bcommunity\s+certificate\b", "Community Certificate"),
        # Income Certificate
        (r"\bincome\s+certificate\b", "Income Certificate"),
        (r"\bannual\s+income\b", "Income Certificate"),
        (r"\bfamily\s+income\b", "Income Certificate"),
        (r"\btahsildar\b", "Income Certificate"),
        (r"\bblock\s+development\s+officer\b", "Income Certificate"),
        # Domicile / Residence Certificate
        (r"\bdomicile\s+certificate\b", "Domicile Certificate"),
        (r"\bresidence\s+certificate\b", "Residence Certificate"),
        (r"\bpermanent\s+resident\b", "Domicile Certificate"),
        (r"\bstate\s+domicile\b", "Domicile Certificate"),
        # Marriage Certificate  (avoid "wife" / "husband" — too common in unrelated docs)
        (r"\bmarriage\s+certificate\b", "Marriage Certificate"),
        (r"\bcertificate\s+of\s+marriage\b", "Marriage Certificate"),
        (r"\bregistrar\s+of\s+marriages?\b", "Marriage Certificate"),
        (r"\bspecial\s+marriage\s+act\b", "Marriage Certificate"),
        # Land / Property Documents
        (r"\bkhata\b", "Land Record"),
        (r"\bkhasra\b", "Land Record"),
        (r"\bkhatauni\b", "Land Record"),
        (r"\bpatta\b", "Land Record"),
        (r"\bsurvey\s+no\b", "Land Record"),
        (r"\bproperty\s+(?:document|deed|record)\b", "Property Document"),
        (r"\bsale\s+deed\b", "Property Document"),
        (r"\bencumbrance\s+certificate\b", "Property Document"),
        (r"\bregistration\s+deed\b", "Property Document"),
        (r"\bsub\s*[-\s]?registrar\b", "Property Document"),
        # Police / Character Certificate
        (r"\bpolice\s+clearance\b", "Police Clearance Certificate"),
        (r"\bcharacter\s+certificate\b", "Character Certificate"),
        (r"\bno\s+criminal\s+record\b", "Police Clearance Certificate"),
        (r"\bsuperintendent\s+of\s+police\b", "Police Certificate"),
        (r"\bstation\s+house\s+officer\b", "Police Certificate"),
        # GST / Tax Certificates
        (r"\bgst\s*(?:certificate|registration|no|number)\b", "GST Certificate"),
        (r"\bgoods\s+and\s+services\s+tax\b", "GST Certificate"),
        (r"\bgstin\b", "GST Certificate"),
        (r"\btax\s+(?:clearance|compliance)\s+certificate\b", "Tax Certificate"),
        # FSSAI / Food License
        (r"\bfssai\b", "FSSAI License"),
        (r"\bfood\s+(?:safety|license|licence)\b", "FSSAI License"),
        (r"\bfood\s+and\s+drug\b", "FSSAI License"),
        # Shop & Establishment / Trade License
        (r"\bshop\s+(?:and|&)\s+establishment\b", "Shop & Establishment Certificate"),
        (r"\btrade\s+licen[cs]e\b", "Trade License"),
        (r"\bmunicipal\s+(?:corporation|council|committee)\b", "Municipal Certificate"),
        # Import / Export
        (r"\bimport\s+export\s+code\b", "IEC Certificate"),
        (r"\biec\s*(?:no|number|certificate)\b", "IEC Certificate"),
        (r"\bdgft\b", "IEC Certificate"),
        # MSME / Udyam
        (r"\bmsme\b", "MSME Certificate"),
        (r"\budyam\b", "MSME Certificate"),
        (r"\budyog\s+aadhar\b", "MSME Certificate"),
        (r"\bsmall\s+(?:scale|medium)\s+(?:industry|enterprise)\b", "MSME Certificate"),
        # Medical / Disability
        (r"\bmedical\s+certificate\b", "Medical Certificate"),
        (r"\bdisability\s+certificate\b", "Disability Certificate"),
        (r"\bpwbd\b", "Disability Certificate"),
        (r"\bphysically\s+handicapped\b", "Disability Certificate"),
        (r"\bchief\s+medical\s+officer\b", "Medical Certificate"),
        # Migration / Transfer / Life
        (r"\bmigration\s+certificate\b", "Migration Certificate"),
        (r"\btransfer\s+certificate\b", "Transfer Certificate"),
        (r"\blife\s+certificate\b", "Life Certificate"),
        (r"\bliving\s+certificate\b", "Life Certificate"),
        # Employment / Experience / Service
        (r"\bexperience\s+certificate\b", "Experience Certificate"),
        (r"\bemployment\s+certificate\b", "Employment Certificate"),
        (r"\bservice\s+certificate\b", "Service Certificate"),
        (r"\brelieving\s+letter\b", "Relieving Letter"),
        # Solvency / Net Worth
        (r"\bsolvency\s+certificate\b", "Solvency Certificate"),
        (r"\bnet\s+worth\s+certificate\b", "Net Worth Certificate"),
        # No Objection Certificate  (\bnoc\b is too short — skip standalone)
        (r"\bno\s+objection\s+certificate\b", "NOC"),
        # Affidavit / Notary
        (r"\baffidavit\b", "Affidavit"),
        (r"\bnotary\s+public\b", "Notarized Document"),
        (r"\bsworn\s+before\b", "Affidavit"),
        # Ration / EWS
        (r"\bews\s+certificate\b", "EWS Certificate"),
        (r"\beconomically\s+weaker\s+section\b", "EWS Certificate"),
        # Insurance
        (r"\binsurance\s+(?:certificate|policy)\b", "Insurance Certificate"),
        (r"\bpolicy\s+(?:no|number)\b", "Insurance Certificate"),
        # Generic Government document markers
        (r"\bgovernment\s+of\s+india\b", "Government Document"),
        (r"\brepublic\s+of\s+india\b", "Government Document"),
        (r"\bministry\s+of\b", "Government Document"),
        (r"\bstate\s+government\b", "Government Document"),
        (r"\bcollector\b", "Government Document"),
        (r"\bdistrict\s+magistrate\b", "Government Document"),
        (r"\btehsildar\b", "Government Document"),
        (r"\bnaib\s+tehsildar\b", "Government Document"),
        (r"\bidentity\s+card\b", "Identity Card"),
        (r"\bphoto\s+id\s+(?:card|proof)\b", "Photo ID"),
        # NOTE: "date of birth" / "DOB" alone are NOT reliable ID indicators
        # (they appear in CVs, resumes, application forms, etc.) — removed.
    ]

    # ── Certificate vocabulary ──────────────────────────────────────────
    STRONG_KEYWORDS = [
        r"\bcertificate\b",
        r"\bcertification\b",
        r"\bhereby\s+certif(?:y|ied)\b",
        r"\bawarded?\s+to\b",
        r"\bpresented?\s+to\b",
        r"\bconferred?\s+(?:upon|to)\b",
        r"\bthis\s+is\s+to\s+certify\b",
        r"\bsuccessfully\s+completed?\b",
        r"\bdiploma\b",
        r"\bdegree\b",
        r"\baccreditation\b",
        r"\bletter\s+of\s+(?:completion|achievement|recognition)\b",
    ]

    MODERATE_KEYWORDS = [
        r"\bcompletion\b",
        r"\bachievement\b",
        r"\bexcellence\b",
        r"\brecognition\b",
        r"\bparticipation\b",
        r"\bauthorized?\s+signator(?:y|ies)\b",
        r"\bregistrar\b",
        r"\bdean\b",
        r"\bprincipal\b",
        r"\bdirector\b",
        r"\bsignature\b",
        r"\bseal\b",
        r"\buniversity\b",
        r"\binstitut(?:e|ion)\b",
        r"\bacademy\b",
        r"\bcollege\b",
        r"\bboard\s+of\b",
        r"\bcourse\b",
        r"\btraining\b",
        r"\bgrade\b",
        r"\bmerit\b",
        r"\bdate\s+of\s+issue\b",
        r"\bvalid(?:ity)?\b",
        r"\bregistration\s+(?:no|number|id)\b",
        r"\bcredential\b",
    ]

    ANTI_KEYWORDS = [
        r"\bdear\s+(?:sir|madam|diary)\b",
        r"\bchapter\s+\d+\b",
        r"\bpage\s+\d+\b",
        r"\bhomework\b",
        r"\bassignment\b",
        r"\bto\-?do\s+list\b",
        r"\bgrocery\b",
        r"\bshopping\s+list\b",
        r"\brecipe\b",
        r"\bmeeting\s+(?:notes|minutes)\b",
        r"\blecture\s+notes?\b",
        r"\bnotes?\s*[:]\b",
        # CV / Resume — clearly not a certificate
        r"\bcurriculum\s+vitae\b",
        r"\bresume\b",
        r"\bwork\s+experience\b",
        r"\bprofessional\s+summary\b",
        r"\bcareer\s+objective\b",
        r"\bskills?\s*[:]\b",
        r"\bhobbies?\s*[:]\b",
        r"\breferences?\s+available\b",
        r"\blinkedin\.com\b",
        r"\bgithub\.com\b",
        r"\bportfolio\b",
    ]

    # ── Thresholds ──────────────────────────────────────────────────────
    CERTIFICATE_THRESHOLD = 40
    HIGH_CONFIDENCE_THRESHOLD = 65

    def detect(
        self,
        image: np.ndarray,
        ocr_text: str = "",
        ocr_lines: list = None,
    ) -> Dict:
        """
        Determine whether the document is a certificate.

        Parameters
        ----------
        image : np.ndarray
            The rendered document image (RGB).
        ocr_text : str
            Full OCR-extracted text.
        ocr_lines : list
            List of OCR line dicts with text, confidence, bbox.

        Returns
        -------
        dict with:
            is_certificate  – bool
            confidence      – 0-100
            reasons         – list of human-readable reasons
            checks          – detailed check results
        """
        checks = {}
        score = 0.0
        reasons = []

        # ── 0. Domestic ID rejection (runs FIRST) ───────────────────────
        id_result = self._check_domestic_id(ocr_text)
        checks["domestic_id"] = id_result

        if id_result["is_domestic_id"]:
            # Immediately reject — no further checks needed
            return {
                "is_certificate": False,
                "confidence": 0.0,
                "reasons": id_result["reasons"],
                "checks": checks,
                "rejection_reason": self._build_domestic_id_rejection(
                    id_result
                ),
                "is_domestic_id": True,
            }

        # ── 1. Keyword analysis ─────────────────────────────────────────
        kw_result = self._check_keywords(ocr_text)
        checks["keywords"] = kw_result
        score += kw_result["score"]

        if kw_result["strong_matches"]:
            reasons.append(
                f"Found certificate keywords: {', '.join(kw_result['strong_matches'][:3])}"
            )
        if kw_result["anti_matches"]:
            reasons.append(
                f"Found non-certificate keywords: {', '.join(kw_result['anti_matches'][:3])}"
            )

        # ── 2. Layout structure ─────────────────────────────────────────
        layout_result = self._check_layout_structure(image)
        checks["layout"] = layout_result
        score += layout_result["score"]

        if layout_result["has_border"]:
            reasons.append("Document has a decorative border (common in certificates)")
        if not layout_result["is_centered"]:
            reasons.append("Content is not well-centered (unusual for certificates)")

        # ── 3. Handwriting detection ────────────────────────────────────
        hw_result = self._detect_handwriting(image, ocr_lines)
        checks["handwriting"] = hw_result
        score += hw_result["score"]

        if hw_result["is_handwritten"]:
            reasons.append(
                "Document appears to be handwritten — certificates are typically printed"
            )

        # ── 4. Text density & structure ─────────────────────────────────
        density_result = self._check_text_density(ocr_text, ocr_lines, image)
        checks["text_density"] = density_result
        score += density_result["score"]

        if density_result.get("too_dense"):
            reasons.append("Text is too dense — resembles notes/essay, not a certificate")
        if density_result.get("too_sparse"):
            reasons.append("Very little text detected")

        # ── 5. Aspect ratio ─────────────────────────────────────────────
        ar_result = self._check_aspect_ratio(image)
        checks["aspect_ratio"] = ar_result
        score += ar_result["score"]

        # ── Final verdict ───────────────────────────────────────────────
        score = max(0, min(100, score))
        is_certificate = score >= self.CERTIFICATE_THRESHOLD

        if not is_certificate:
            if not reasons:
                reasons.append(
                    "Document does not match the expected format of a certificate"
                )

        return {
            "is_certificate": is_certificate,
            "confidence": round(score, 1),
            "reasons": reasons,
            "checks": checks,
            "rejection_reason": (
                None if is_certificate
                else self._build_rejection_message(score, checks, reasons)
            ),
            "is_domestic_id": False,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Check implementations
    # ──────────────────────────────────────────────────────────────────────

    def _check_domestic_id(self, text: str) -> Dict:
        """
        Check if the document is a domestic/government ID card.
        This runs FIRST and causes immediate rejection.
        """
        if not text:
            return {
                "is_domestic_id": False,
                "matched_types": [],
                "match_count": 0,
                "reasons": [],
            }

        text_lower = text.lower()
        text_upper = text.upper()
        matched_types: Dict[str, int] = {}

        for pattern, label in self.DOMESTIC_ID_KEYWORDS:
            # Try case-insensitive search
            if re.search(pattern, text_lower, re.IGNORECASE):
                matched_types[label] = matched_types.get(label, 0) + 1

        # PAN format needs uppercase check (strict pattern)
        pan_match = re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", text_upper)
        if pan_match:
            matched_types["PAN Card"] = matched_types.get("PAN Card", 0) + 1

        # Determine if it's actually a domestic ID
        # Need at least 2 matches from the same type, or matches from 2+ types
        is_domestic_id = False
        reasons = []

        # Single strong match types (very specific keywords)
        strong_types = {"Aadhaar Card", "Voter ID", "PAN Card", "Driving License", "Ration Card"}
        for doc_type, count in matched_types.items():
            if doc_type in strong_types and count >= 2:
                is_domestic_id = True
                reasons.append(
                    f"Document appears to be a {doc_type} (matched {count} indicators)"
                )

        # If we have matches from multiple different ID types, also flag
        if len(matched_types) >= 2 and not is_domestic_id:
            # Check if it's not just generic markers
            specific_types = {k for k in matched_types if k not in {"Government ID", "Identity Document", "Photo ID", "Identity Card"}}
            if specific_types:
                is_domestic_id = True
                type_names = ", ".join(sorted(specific_types))
                reasons.append(
                    f"Document matches multiple ID document types: {type_names}"
                )

        # Special case: Aadhaar 12-digit number pattern is very strong alone
        if "Aadhaar Card" in matched_types and re.search(r"\b\d{4}\s+\d{4}\s+\d{4}\b", text):
            is_domestic_id = True
            if not reasons:
                reasons.append("Document contains Aadhaar card number pattern")

        if not reasons and is_domestic_id:
            reasons.append("Document appears to be a government-issued identity document")

        return {
            "is_domestic_id": is_domestic_id,
            "matched_types": list(matched_types.keys()),
            "match_details": matched_types,
            "match_count": sum(matched_types.values()),
            "reasons": reasons,
        }

    def _check_keywords(self, text: str) -> Dict:
        """Score based on certificate-related keywords in OCR text."""
        text_lower = text.lower()
        score = 0.0

        strong_matches = []
        for pat in self.STRONG_KEYWORDS:
            m = re.search(pat, text_lower)
            if m:
                strong_matches.append(m.group(0))
                score += 12  # strong signal

        moderate_matches = []
        for pat in self.MODERATE_KEYWORDS:
            m = re.search(pat, text_lower)
            if m:
                moderate_matches.append(m.group(0))
                score += 3  # moderate signal

        anti_matches = []
        for pat in self.ANTI_KEYWORDS:
            m = re.search(pat, text_lower)
            if m:
                anti_matches.append(m.group(0))
                score -= 10  # negative signal

        # Cap keyword contribution
        score = max(-30, min(50, score))

        return {
            "score": score,
            "strong_matches": strong_matches,
            "moderate_matches": moderate_matches[:5],
            "anti_matches": anti_matches,
            "strong_count": len(strong_matches),
            "moderate_count": len(moderate_matches),
            "anti_count": len(anti_matches),
        }

    def _check_layout_structure(self, image: np.ndarray) -> Dict:
        """Check for borders, centered content, formal layout."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        score = 0.0

        # ── Border detection ──
        edges = cv2.Canny(gray, 50, 150)
        bw = max(int(w * 0.05), 5)
        bh = max(int(h * 0.05), 5)

        top_strip = edges[:bh, :]
        bottom_strip = edges[h - bh:, :]
        left_strip = edges[:, :bw]
        right_strip = edges[:, w - bw:]

        densities = [
            np.count_nonzero(s) / max(s.size, 1)
            for s in [top_strip, bottom_strip, left_strip, right_strip]
        ]
        avg_border_density = float(np.mean(densities))
        has_border = avg_border_density > 0.08

        if has_border:
            score += 10

        # ── Symmetry ──
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 10,
        )
        left_half = binary[:, :w // 2]
        right_half = cv2.flip(binary[:, w // 2: w // 2 * 2], 1)
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]

        if left_half.size > 0:
            symmetry = float(np.sum(left_half == right_half) / left_half.size)
        else:
            symmetry = 0.5

        if symmetry > 0.85:
            score += 10
        elif symmetry > 0.75:
            score += 5

        # ── Content centering ──
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (h * w) * 0.001
        significant = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]

        centered_count = 0
        for (x, y, rw, rh) in significant:
            cx = (x + rw / 2) / w
            if 0.25 < cx < 0.75:
                centered_count += 1

        is_centered = (centered_count / max(len(significant), 1)) > 0.5 if significant else False
        if is_centered:
            score += 5

        return {
            "score": score,
            "has_border": has_border,
            "border_density": round(avg_border_density, 4),
            "symmetry": round(symmetry, 3),
            "is_centered": is_centered,
            "significant_regions": len(significant),
        }

    def _detect_handwriting(self, image: np.ndarray, ocr_lines: list = None) -> Dict:
        """
        Detect if the document is handwritten using:
        - Line spacing irregularity
        - Edge density patterns (handwriting has inconsistent stroke patterns)
        - OCR confidence patterns (handwriting typically has lower/variable confidence)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        score = 0.0
        is_handwritten = False
        signals = []

        # ── OCR confidence analysis ──
        if ocr_lines and len(ocr_lines) > 3:
            confidences = [l.get("confidence", 0) for l in ocr_lines]
            avg_conf = np.mean(confidences)
            conf_std = np.std(confidences)

            # Handwriting: low average confidence AND high variance
            if avg_conf < 0.5 and conf_std > 0.15:
                signals.append("low_variable_confidence")
                score -= 10
            elif avg_conf < 0.4:
                signals.append("very_low_confidence")
                score -= 8

        # ── Line spacing analysis ──
        if ocr_lines and len(ocr_lines) > 5:
            y_positions = sorted([
                l["bbox"][1] for l in ocr_lines if "bbox" in l and len(l["bbox"]) >= 2
            ])
            if len(y_positions) > 3:
                spacings = np.diff(y_positions)
                spacings = spacings[spacings > 5]  # filter noise
                if len(spacings) > 3:
                    spacing_cv = float(np.std(spacings) / max(np.mean(spacings), 1))
                    # High coefficient of variation = irregular spacing = handwriting
                    if spacing_cv > 0.6:
                        signals.append("irregular_line_spacing")
                        score -= 10

        # ── Stroke pattern: horizontal projection profile ──
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 10,
        )
        # Horizontal projection (sum of ink per row)
        h_proj = np.sum(binary > 0, axis=1).astype(float)
        # Normalize
        h_proj_norm = h_proj / max(w, 1)

        # For printed text: clear peaks (text lines) and valleys (gaps)
        # For handwriting: more noisy, less structured
        nonzero_rows = h_proj_norm > 0.01
        transitions = np.diff(nonzero_rows.astype(int))
        num_transitions = np.sum(np.abs(transitions))

        # Printed certificates: moderate transitions, handwriting: very many
        if num_transitions > h * 0.15:
            signals.append("high_row_transitions")
            score -= 5

        # ── Final handwriting verdict ──
        if len(signals) >= 2:
            is_handwritten = True
            score -= 10  # additional penalty
        elif len(signals) == 1:
            score -= 3  # minor penalty

        # If NOT handwritten, give a small bonus
        if not is_handwritten:
            score += 5

        return {
            "score": score,
            "is_handwritten": is_handwritten,
            "signals": signals,
            "signal_count": len(signals),
        }

    def _check_text_density(
        self, text: str, ocr_lines: list, image: np.ndarray
    ) -> Dict:
        """Check if text density matches certificate expectations."""
        score = 0.0
        word_count = len(text.split()) if text else 0
        line_count = len(ocr_lines) if ocr_lines else 0

        too_dense = False
        too_sparse = False

        # Certificates typically have 20-300 words
        if word_count < 5:
            score -= 15
            too_sparse = True
        elif word_count < 15:
            score -= 5
            too_sparse = True
        elif word_count > 400:
            score -= 15
            too_dense = True
        elif 20 <= word_count <= 300:
            score += 10  # ideal range

        # Line count: certificates typically have 10-40 distinct lines
        if line_count > 60:
            score -= 5
            too_dense = True
        elif 5 <= line_count <= 40:
            score += 5

        return {
            "score": score,
            "word_count": word_count,
            "line_count": line_count,
            "too_dense": too_dense,
            "too_sparse": too_sparse,
        }

    def _check_aspect_ratio(self, image: np.ndarray) -> Dict:
        """Check if aspect ratio matches common certificate formats."""
        h, w = image.shape[:2]
        ar = w / max(h, 1)
        score = 0.0

        # Common certificate aspect ratios:
        # A4 landscape: 1.414, Letter landscape: 1.294
        # A4 portrait: 0.707, Letter portrait: 0.773
        # Close to standard ratios is a positive signal
        standard_ratios = [1.414, 1.294, 0.707, 0.773, 1.0, 1.333]
        min_diff = min(abs(ar - r) for r in standard_ratios)

        if min_diff < 0.1:
            score += 5
        elif min_diff > 0.5:
            score -= 5  # very unusual aspect ratio

        return {
            "score": score,
            "aspect_ratio": round(ar, 3),
            "closest_standard_diff": round(min_diff, 3),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Rejection messages
    # ──────────────────────────────────────────────────────────────────────

    def _build_domestic_id_rejection(self, id_result: Dict) -> str:
        """Build rejection message for domestic ID documents."""
        doc_types = id_result.get("matched_types", [])
        if doc_types:
            type_str = ", ".join(doc_types[:3])
            return (
                f"The uploaded document appears to be a domestic identity document "
                f"({type_str}). This system is designed to verify academic and "
                f"professional certificates only — not government-issued ID cards. "
                f"Please upload a certificate, diploma, or credential document instead."
            )
        return (
            "The uploaded document appears to be a government-issued identity "
            "document. Please upload a certificate, diploma, or credential instead."
        )

    def _build_rejection_message(
        self, score: float, checks: Dict, reasons: List[str]
    ) -> str:
        """Build a user-friendly rejection message."""
        parts = [
            "The uploaded document does not appear to be a valid certificate."
        ]

        # Identify the primary reason
        kw = checks.get("keywords", {})
        hw = checks.get("handwriting", {})
        density = checks.get("text_density", {})

        if hw.get("is_handwritten"):
            parts.append(
                "The document appears to be handwritten notes or a hand-drawn page. "
                "This system verifies printed/digital certificates only."
            )
        elif kw.get("strong_count", 0) == 0 and kw.get("moderate_count", 0) < 2:
            parts.append(
                "No certificate-related content was detected in the document. "
                "Please upload a formal certificate, diploma, or credential document."
            )
        elif density.get("too_dense"):
            parts.append(
                "The document contains too much text for a certificate — "
                "it may be an essay, notes, or a multi-page document."
            )
        elif density.get("too_sparse"):
            parts.append(
                "The document contains very little readable text. "
                "Please upload a clear image of a certificate."
            )
        elif kw.get("anti_count", 0) > 0:
            parts.append(
                "The document content suggests it is personal notes or a letter, "
                "not a certificate."
            )
        else:
            parts.append(
                "The document layout and content do not match expected "
                "certificate formats. Please upload a proper certificate."
            )

        return " ".join(parts)
