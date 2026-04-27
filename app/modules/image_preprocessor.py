"""
Image Preprocessing Module
===========================
Performs noise reduction, contrast enhancement, deskewing, morphological
cleaning, and other image-level transformations to improve downstream
OCR accuracy and forensic analysis quality.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


class ImagePreprocessor:
    """Applies a configurable preprocessing pipeline to certificate images."""

    def __init__(self):
        self.processing_log: list = []

    def preprocess(self, image: np.ndarray, full_pipeline: bool = True) -> Dict:
        """
        Run the full preprocessing pipeline.

        Returns a dict with:
            - original: original image
            - processed: final preprocessed image
            - grayscale: grayscale version
            - binary: binarised version
            - deskew_angle: detected skew angle
            - preprocessing_score: quality-of-preprocessing metric (0-100)
        """
        self.processing_log.clear()
        result: Dict = {
            "original": image.copy(),
            "preprocessing_score": 100,
            "deskew_angle": 0.0,
        }

        # 1. Convert to grayscale
        gray = self._to_grayscale(image)
        result["grayscale"] = gray

        if not full_pipeline:
            result["processed"] = gray
            result["binary"] = self._binarize(gray)
            return result

        # 2. Noise reduction
        denoised = self._denoise(gray)

        # 3. Contrast enhancement (CLAHE)
        enhanced = self._enhance_contrast(denoised)

        # 4. Deskew
        deskewed, angle = self._deskew(enhanced)
        result["deskew_angle"] = angle
        if abs(angle) > 5:
            result["preprocessing_score"] -= 10
            self.processing_log.append(f"Large skew detected: {angle:.1f}°")

        # 5. Morphological cleaning
        cleaned = self._morphological_clean(deskewed)

        # 6. Binarize
        binary = self._binarize(cleaned)

        result["processed"] = cleaned
        result["binary"] = binary
        result["log"] = list(self.processing_log)
        return result

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        """Apply bilateral filter (edge-preserving denoising)."""
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        self.processing_log.append("Applied bilateral denoising")
        return denoised

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        self.processing_log.append("Applied CLAHE contrast enhancement")
        return enhanced

    def _deskew(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """Correct skew using Hough-line-based angle detection."""
        angle = self._detect_skew_angle(gray)
        if abs(angle) < 0.5:
            return gray, 0.0

        h, w = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        self.processing_log.append(f"Deskewed by {angle:.2f}°")
        return rotated, angle

    def _detect_skew_angle(self, gray: np.ndarray) -> float:
        """Detect skew angle via probabilistic Hough transform."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=gray.shape[1] // 4, maxLineGap=10,
        )
        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return 0.0

        return float(np.median(angles))

    def _morphological_clean(self, gray: np.ndarray) -> np.ndarray:
        """Light morphological opening to remove small noise."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        self.processing_log.append("Applied morphological cleaning")
        return cleaned

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive thresholding for robust binarisation."""
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )
        return binary

    # ------------------------------------------------------------------
    # Utility helpers exposed to other modules
    # ------------------------------------------------------------------

    def resize_for_analysis(self, image: np.ndarray, max_dim: int = 2000) -> np.ndarray:
        """Resize image so that its longest dimension ≤ max_dim."""
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def extract_color_image(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is in BGR for OpenCV analysis."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def get_edge_density(self, gray: np.ndarray) -> float:
        """Return the ratio of edge pixels (Canny) to total pixels."""
        edges = cv2.Canny(gray, 50, 150)
        return float(np.count_nonzero(edges) / edges.size)
