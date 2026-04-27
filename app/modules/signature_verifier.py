"""
Signature & Seal Verification Module
======================================
Detects and analyses signature and seal/stamp regions in certificate images:
  • Contour-based signature region detection
  • Circular Hough transform for seal/stamp detection
  • Signature complexity analysis
  • Colour analysis for ink and seal authenticity
  • SSIM-based similarity comparison when a reference is available

Uses OpenCV as a practical stand-in for YOLOv8 object detection and
Siamese-network matching (which require trained weights).
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


class SignatureVerifier:
    """Detect and verify signatures and seals on certificate images."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        image: np.ndarray,
        reference_signature: Optional[np.ndarray] = None,
        reference_seal: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run the full signature and seal verification pipeline.

        Returns dict with:
            signature_score   – 0-100 authenticity estimate
            seal_score        – 0-100 authenticity estimate
            signature_regions – list of detected regions
            seal_regions      – list of detected regions
            anomalies         – list of textual findings
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if len(image.shape) == 3
            else image
        )
        bgr = (
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if len(image.shape) == 3 and image.shape[2] == 3
            else image
        )
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) if len(bgr.shape) == 3 else None

        h, w = gray.shape[:2]

        sig_regions = self._detect_signature_regions(gray, h, w)
        seal_regions = self._detect_seal_regions(gray, bgr, hsv, h, w)

        anomalies: List[str] = []

        # --- Signature analysis ---
        sig_score = 70.0  # neutral base
        if not sig_regions:
            anomalies.append("No signature region detected")
            sig_score = 30.0
        else:
            best = sig_regions[0]
            complexity = self._signature_complexity(gray, best)
            if complexity["stroke_density"] < 0.02:
                anomalies.append("Signature appears too simple / possibly printed text")
                sig_score -= 20
            if complexity["stroke_density"] > 0.5:
                anomalies.append("Signature region is overly dense – may be a filled block")
                sig_score -= 15
            sig_score += complexity["complexity_bonus"]

            # Colour check – real signatures are usually dark (blue / black)
            ink = self._check_ink_colour(bgr, best)
            if ink["is_valid_ink"]:
                sig_score += 10
            else:
                anomalies.append("Signature ink colour is unusual")
                sig_score -= 10

            # Reference comparison
            if reference_signature is not None:
                sim = self._compare_signature(gray, best, reference_signature)
                sig_score = sig_score * 0.5 + sim * 50
                if sim < 0.4:
                    anomalies.append("Signature does not match reference")

        # --- Seal analysis ---
        seal_score = 70.0
        if not seal_regions:
            # Seals are optional on many certificates – only mild penalty
            seal_score = 50.0
        else:
            best_seal = seal_regions[0]
            circ = best_seal.get("circularity", 0)
            if circ > 0.6:
                seal_score += 15
            elif circ < 0.3:
                anomalies.append("Detected seal region is not circular")
                seal_score -= 10

            seal_colour = self._check_seal_colour(hsv, best_seal)
            if seal_colour["is_valid"]:
                seal_score += 10
            else:
                anomalies.append("Seal colour is atypical")
                seal_score -= 10

            if reference_seal is not None:
                sim = self._compare_seal(gray, best_seal, reference_seal)
                seal_score = seal_score * 0.5 + sim * 50
                if sim < 0.4:
                    anomalies.append("Seal does not match reference")

        sig_score = max(0, min(100, sig_score))
        seal_score = max(0, min(100, seal_score))

        return {
            "signature_score": round(sig_score, 1),
            "seal_score": round(seal_score, 1),
            "signature_regions": sig_regions,
            "seal_regions": seal_regions,
            "anomalies": anomalies,
        }

    # ------------------------------------------------------------------
    # Signature detection
    # ------------------------------------------------------------------

    def _detect_signature_regions(
        self, gray: np.ndarray, h: int, w: int
    ) -> List[Dict]:
        """
        Detect potential signature regions.
        Heuristic: signatures are usually in the bottom 40 % of the
        certificate and have a distinctive stroke pattern.
        """
        # Focus on bottom portion
        bottom_start = int(h * 0.55)
        roi = gray[bottom_start:, :]

        # Adaptive threshold to find dark strokes
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10,
        )
        # Dilate to merge nearby strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        regions: List[Dict] = []
        roi_h, roi_w = roi.shape[:2]
        min_area = roi_h * roi_w * 0.005
        max_area = roi_h * roi_w * 0.4

        for cnt in contours:
            x, y, rw, rh = cv2.boundingRect(cnt)
            area = rw * rh
            aspect = rw / max(rh, 1)

            # Signature heuristic: wider than tall, moderate area
            if min_area < area < max_area and aspect > 1.5:
                regions.append({
                    "x": int(x),
                    "y": int(y + bottom_start),
                    "w": int(rw),
                    "h": int(rh),
                    "area": int(area),
                    "aspect_ratio": round(aspect, 2),
                })

        # Sort by area descending
        regions.sort(key=lambda r: r["area"], reverse=True)
        return regions[:5]

    # ------------------------------------------------------------------
    # Seal / stamp detection
    # ------------------------------------------------------------------

    def _detect_seal_regions(
        self,
        gray: np.ndarray,
        bgr: np.ndarray,
        hsv: Optional[np.ndarray],
        h: int,
        w: int,
    ) -> List[Dict]:
        """
        Detect circular seal / stamp regions using Hough circles and
        colour filtering.
        """
        regions: List[Dict] = []

        # --- Hough circle detection ---
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(min(h, w) * 0.1),
            param1=100,
            param2=40,
            minRadius=int(min(h, w) * 0.03),
            maxRadius=int(min(h, w) * 0.2),
        )

        if circles is not None:
            for circle in circles[0]:
                cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
                regions.append({
                    "x": max(0, cx - r),
                    "y": max(0, cy - r),
                    "w": int(2 * r),
                    "h": int(2 * r),
                    "center": [cx, cy],
                    "radius": int(r),
                    "circularity": 1.0,
                    "detection_method": "hough",
                })

        # --- Colour-based seal detection (red / blue seals) ---
        if hsv is not None and not regions:
            # Red seal mask
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])
            mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(
                hsv, lower_red2, upper_red2
            )

            # Blue seal mask
            lower_blue = np.array([100, 70, 50])
            upper_blue = np.array([130, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

            combined_mask = mask_red | mask_blue

            # Find colour blobs
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            min_area = h * w * 0.002
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                x, y, rw, rh = cv2.boundingRect(cnt)
                # Circularity check
                perimeter = cv2.arcLength(cnt, True)
                circularity = (
                    (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                )
                if circularity > 0.3:
                    regions.append({
                        "x": int(x),
                        "y": int(y),
                        "w": int(rw),
                        "h": int(rh),
                        "circularity": round(float(circularity), 3),
                        "detection_method": "colour",
                    })

        regions.sort(key=lambda r: r.get("circularity", 0), reverse=True)
        return regions[:5]

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _signature_complexity(self, gray: np.ndarray, region: Dict) -> Dict:
        """Analyse the stroke complexity of a signature region."""
        x, y, rw, rh = region["x"], region["y"], region["w"], region["h"]
        roi = gray[y : y + rh, x : x + rw]
        if roi.size == 0:
            return {"stroke_density": 0, "complexity_bonus": 0}

        edges = cv2.Canny(roi, 50, 150)
        density = float(np.count_nonzero(edges) / edges.size)

        # Count connected components as a proxy for complexity
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, _ = cv2.connectedComponents(binary)

        bonus = 0
        if 0.05 < density < 0.35:
            bonus += 10
        if 3 < num_labels < 50:
            bonus += 5

        return {
            "stroke_density": round(density, 4),
            "connected_components": int(num_labels),
            "complexity_bonus": bonus,
        }

    def _check_ink_colour(self, bgr: np.ndarray, region: Dict) -> Dict:
        """Check whether the ink colour in the signature region is plausible."""
        x, y, rw, rh = region["x"], region["y"], region["w"], region["h"]
        roi = bgr[y : y + rh, x : x + rw]
        if roi.size == 0:
            return {"is_valid_ink": False}

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Dark ink = low value channel
        v_channel = hsv_roi[:, :, 2]
        dark_ratio = float(np.sum(v_channel < 80) / v_channel.size)

        # Blue ink check
        h_channel = hsv_roi[:, :, 0]
        s_channel = hsv_roi[:, :, 1]
        blue_mask = (h_channel > 100) & (h_channel < 130) & (s_channel > 50)
        blue_ratio = float(np.sum(blue_mask) / h_channel.size)

        is_valid = dark_ratio > 0.05 or blue_ratio > 0.05
        return {
            "is_valid_ink": is_valid,
            "dark_ink_ratio": round(dark_ratio, 4),
            "blue_ink_ratio": round(blue_ratio, 4),
        }

    def _check_seal_colour(self, hsv: Optional[np.ndarray], region: Dict) -> Dict:
        """Check seal colour plausibility."""
        if hsv is None:
            return {"is_valid": True}  # can't check without colour

        x, y, rw, rh = region["x"], region["y"], region["w"], region["h"]
        roi = hsv[y : y + rh, x : x + rw]
        if roi.size == 0:
            return {"is_valid": False}

        h_ch = roi[:, :, 0]
        s_ch = roi[:, :, 1]

        # Red or blue dominant
        red_mask = ((h_ch < 10) | (h_ch > 170)) & (s_ch > 50)
        blue_mask = (h_ch > 100) & (h_ch < 130) & (s_ch > 50)
        colour_ratio = float((np.sum(red_mask) + np.sum(blue_mask)) / h_ch.size)

        return {"is_valid": colour_ratio > 0.05, "colour_ratio": round(colour_ratio, 4)}

    # ------------------------------------------------------------------
    # Reference comparison (SSIM-based)
    # ------------------------------------------------------------------

    def _compare_signature(
        self, gray: np.ndarray, region: Dict, reference: np.ndarray
    ) -> float:
        """Compare detected signature to a reference using SSIM."""
        return self._ssim_compare(gray, region, reference)

    def _compare_seal(
        self, gray: np.ndarray, region: Dict, reference: np.ndarray
    ) -> float:
        return self._ssim_compare(gray, region, reference)

    def _ssim_compare(
        self, gray: np.ndarray, region: Dict, reference: np.ndarray
    ) -> float:
        """Compute SSIM between a detected region and a reference image."""
        try:
            from skimage.metrics import structural_similarity as ssim

            x, y, rw, rh = region["x"], region["y"], region["w"], region["h"]
            roi = gray[y : y + rh, x : x + rw]
            if roi.size == 0:
                return 0.0

            ref_gray = (
                cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
                if len(reference.shape) == 3
                else reference
            )
            # Resize to match
            ref_resized = cv2.resize(ref_gray, (rw, rh))
            score = ssim(roi, ref_resized)
            return float(max(0, score))
        except Exception:
            return 0.5
