"""
Layout Analysis Module
=======================
Validates the spatial structure and visual consistency of a certificate:
  • Element detection (logo, heading, text blocks, borders)
  • Symmetry and alignment analysis
  • Margin and spacing consistency
  • Layout-mismatch scoring

Uses OpenCV contour analysis and heuristic spatial rules as a practical
stand-in for LayoutLMv3 (which requires fine-tuned weights).
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class LayoutAnalyzer:
    """Analyse the layout structure and consistency of certificate images."""

    def analyze(self, image: np.ndarray, binary: np.ndarray | None = None) -> Dict:
        """
        Run the full layout analysis pipeline.

        Parameters
        ----------
        image : np.ndarray
            Original colour image (RGB or BGR).
        binary : np.ndarray, optional
            Pre-computed binary image; if None, one will be generated.

        Returns
        -------
        dict with layout_score (0-100), detected elements, and anomalies.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if len(image.shape) == 3
            else image
        )

        if binary is None:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 10,
            )

        h, w = gray.shape[:2]

        elements = self._detect_elements(binary, gray, h, w)
        symmetry = self._check_symmetry(binary, h, w)
        alignment = self._check_alignment(elements, h, w)
        margins = self._check_margins(elements, h, w)
        border = self._detect_border(gray, h, w)

        anomalies: List[str] = []
        layout_score = 100.0

        # --- Symmetry penalties ---
        if symmetry["horizontal_symmetry"] < 0.6:
            anomalies.append("Poor horizontal symmetry")
            layout_score -= 15
        elif symmetry["horizontal_symmetry"] < 0.8:
            layout_score -= 5

        # --- Alignment penalties ---
        if alignment["center_alignment_ratio"] < 0.3:
            anomalies.append("Text blocks are poorly centred")
            layout_score -= 10

        # --- Margin penalties ---
        if not margins["consistent"]:
            anomalies.append("Inconsistent margins detected")
            layout_score -= 10

        # --- Element count ---
        if elements["total_regions"] < 3:
            anomalies.append("Very few layout regions detected")
            layout_score -= 10
        elif elements["total_regions"] > 50:
            anomalies.append("Unusually cluttered layout")
            layout_score -= 5

        # --- Border ---
        if border["has_border"]:
            layout_score += 5  # Bordered certificates are more formal
        
        layout_score = max(0, min(100, layout_score))

        return {
            "layout_score": round(layout_score, 1),
            "elements": elements,
            "symmetry": symmetry,
            "alignment": alignment,
            "margins": margins,
            "border": border,
            "anomalies": anomalies,
        }

    # ------------------------------------------------------------------
    # Element detection
    # ------------------------------------------------------------------

    def _detect_elements(
        self, binary: np.ndarray, gray: np.ndarray, h: int, w: int
    ) -> Dict:
        """Detect layout regions via contour analysis."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        regions: List[Dict] = []
        logo_candidates = []
        heading_candidates = []
        text_blocks = []

        min_area = (h * w) * 0.0005  # ignore very tiny blobs

        for cnt in contours:
            x, y, rw, rh = cv2.boundingRect(cnt)
            area = rw * rh
            if area < min_area:
                continue

            region = {
                "x": int(x), "y": int(y),
                "w": int(rw), "h": int(rh),
                "area": int(area),
                "type": "unknown",
            }

            rel_y = y / h
            rel_x_center = (x + rw / 2) / w
            aspect = rw / max(rh, 1)

            # Logo heuristic: small-ish, near top, roughly square
            if rel_y < 0.25 and 0.5 < aspect < 2.0 and area < (h * w) * 0.1:
                region["type"] = "logo"
                logo_candidates.append(region)
            # Heading: wide, near top, thin height
            elif rel_y < 0.35 and aspect > 3 and rw > w * 0.3:
                region["type"] = "heading"
                heading_candidates.append(region)
            # Text block
            elif aspect > 2:
                region["type"] = "text_block"
                text_blocks.append(region)
            else:
                region["type"] = "element"

            regions.append(region)

        return {
            "total_regions": len(regions),
            "logos": len(logo_candidates),
            "headings": len(heading_candidates),
            "text_blocks": len(text_blocks),
            "regions": regions[:30],  # cap for JSON size
        }

    # ------------------------------------------------------------------
    # Symmetry
    # ------------------------------------------------------------------

    def _check_symmetry(self, binary: np.ndarray, h: int, w: int) -> Dict:
        """Measure horizontal symmetry by comparing left/right halves."""
        left = binary[:, : w // 2]
        right = cv2.flip(binary[:, w // 2 : w // 2 * 2], 1)

        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]

        if left.size == 0:
            return {"horizontal_symmetry": 0.5}

        match = np.sum(left == right) / left.size
        return {"horizontal_symmetry": round(float(match), 3)}

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def _check_alignment(self, elements: Dict, h: int, w: int) -> Dict:
        """Check how well regions are centred horizontally."""
        regions = elements.get("regions", [])
        if not regions:
            return {"center_alignment_ratio": 0.5, "aligned_count": 0}

        center = w / 2
        threshold = w * 0.15  # within 15 % of centre
        aligned = 0
        for r in regions:
            rx_center = r["x"] + r["w"] / 2
            if abs(rx_center - center) < threshold:
                aligned += 1

        ratio = aligned / len(regions)
        return {
            "center_alignment_ratio": round(ratio, 3),
            "aligned_count": aligned,
            "total_checked": len(regions),
        }

    # ------------------------------------------------------------------
    # Margins
    # ------------------------------------------------------------------

    def _check_margins(self, elements: Dict, h: int, w: int) -> Dict:
        """Check margin consistency around the content area."""
        regions = elements.get("regions", [])
        if not regions:
            return {"consistent": True, "margins": {}}

        left_margins = [r["x"] for r in regions]
        right_margins = [w - (r["x"] + r["w"]) for r in regions]
        top_margin = min(r["y"] for r in regions)
        bottom_margin = h - max(r["y"] + r["h"] for r in regions)

        left_std = float(np.std(left_margins)) if left_margins else 0
        right_std = float(np.std(right_margins)) if right_margins else 0

        # Consistent if standard deviation of margins is moderate
        consistent = left_std < w * 0.15 and right_std < w * 0.15
        return {
            "consistent": consistent,
            "left_margin_std": round(left_std, 1),
            "right_margin_std": round(right_std, 1),
            "top_margin": int(top_margin),
            "bottom_margin": int(bottom_margin),
        }

    # ------------------------------------------------------------------
    # Border detection
    # ------------------------------------------------------------------

    def _detect_border(self, gray: np.ndarray, h: int, w: int) -> Dict:
        """Detect if the certificate has a decorative border."""
        edges = cv2.Canny(gray, 50, 150)

        # Sample edge density along the four borders (outer 5 %)
        bw = max(int(w * 0.05), 5)
        bh = max(int(h * 0.05), 5)

        top_strip = edges[:bh, :]
        bottom_strip = edges[h - bh :, :]
        left_strip = edges[:, :bw]
        right_strip = edges[:, w - bw :]

        densities = [
            np.count_nonzero(s) / max(s.size, 1)
            for s in [top_strip, bottom_strip, left_strip, right_strip]
        ]
        avg_density = float(np.mean(densities))

        has_border = avg_density > 0.08
        return {
            "has_border": has_border,
            "border_edge_density": round(avg_density, 4),
        }
