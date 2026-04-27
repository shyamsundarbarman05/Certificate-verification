"""
Forensic Analysis Module
=========================
Performs document-level forensic checks to detect tampering:
  • Error Level Analysis (ELA) – JPEG recompression artefacts
  • Edge inconsistency detection
  • Noise-pattern analysis
  • Copy-paste / cloning detection (block-matching)
  • Compression artefact analysis
  • Overall forensic anomaly scoring
"""

import cv2
import io
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple


class ForensicAnalyzer:
    """Perform forensic analysis on certificate images to detect tampering."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, image: np.ndarray) -> Dict:
        """
        Run the full forensic analysis pipeline.

        Returns dict with:
            forensic_score – 0-100 (100 = no anomalies)
            ela             – error-level analysis results
            edge_analysis   – edge consistency results
            noise_analysis  – noise-pattern results
            copy_paste      – clone-detection results
            compression     – compression-artefact results
            anomalies       – list of textual findings
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if len(image.shape) == 3
            else image
        )

        anomalies: List[str] = []
        forensic_score = 100.0

        # 1. Error Level Analysis
        ela = self._error_level_analysis(image)
        if ela["suspicious"]:
            anomalies.append(
                f"ELA detected potential tampering (max deviation: {ela['max_deviation']:.1f})"
            )
            forensic_score -= ela["penalty"]

        # 2. Edge inconsistency
        edge = self._edge_consistency(gray)
        if edge["inconsistent"]:
            anomalies.append("Edge inconsistencies detected in document regions")
            forensic_score -= edge["penalty"]

        # 3. Noise analysis
        noise = self._noise_analysis(gray)
        if noise["suspicious"]:
            anomalies.append(
                f"Inconsistent noise patterns detected (variance ratio: {noise['variance_ratio']:.2f})"
            )
            forensic_score -= noise["penalty"]

        # 4. Copy-paste detection
        copy_paste = self._copy_paste_detection(gray)
        if copy_paste["suspicious"]:
            anomalies.append(
                f"Possible copy-paste artefacts detected ({copy_paste['match_count']} similar regions)"
            )
            forensic_score -= copy_paste["penalty"]

        # 5. Compression analysis
        compression = self._compression_analysis(image)
        if compression["suspicious"]:
            anomalies.append(
                f"Compression anomalies detected (quality estimate: {compression['estimated_quality']})"
            )
            forensic_score -= compression["penalty"]

        forensic_score = max(0, min(100, forensic_score))

        return {
            "forensic_score": round(forensic_score, 1),
            "ela": ela,
            "edge_analysis": edge,
            "noise_analysis": noise,
            "copy_paste": copy_paste,
            "compression": compression,
            "anomalies": anomalies,
        }

    # ------------------------------------------------------------------
    # Error Level Analysis (ELA)
    # ------------------------------------------------------------------

    def _error_level_analysis(self, image: np.ndarray, quality: int = 90) -> Dict:
        """
        Re-save the image at a known JPEG quality and compare with the
        original.  Tampered regions show different error levels.
        """
        try:
            pil_img = Image.fromarray(image)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            recompressed = np.array(Image.open(buf))

            # Ensure same size
            if recompressed.shape != image.shape:
                recompressed = cv2.resize(
                    recompressed, (image.shape[1], image.shape[0])
                )

            # Compute absolute difference
            diff = cv2.absdiff(image, recompressed).astype(np.float64)
            if len(diff.shape) == 3:
                diff = np.mean(diff, axis=2)

            max_dev = float(np.max(diff))
            mean_dev = float(np.mean(diff))
            std_dev = float(np.std(diff))

            # Divide image into grid and check per-block deviation
            block_devs = self._block_deviations(diff, blocks=8)
            dev_range = float(np.max(block_devs) - np.min(block_devs))

            suspicious = dev_range > 15 or std_dev > 12
            penalty = min(25, dev_range * 0.8) if suspicious else 0

            return {
                "max_deviation": max_dev,
                "mean_deviation": round(mean_dev, 2),
                "std_deviation": round(std_dev, 2),
                "block_deviation_range": round(dev_range, 2),
                "suspicious": suspicious,
                "penalty": round(penalty, 1),
            }
        except Exception as e:
            return {
                "max_deviation": 0,
                "mean_deviation": 0,
                "std_deviation": 0,
                "block_deviation_range": 0,
                "suspicious": False,
                "penalty": 0,
                "error": str(e),
            }

    def _block_deviations(self, diff: np.ndarray, blocks: int = 8) -> np.ndarray:
        """Compute mean deviation per grid block."""
        h, w = diff.shape[:2]
        bh, bw = h // blocks, w // blocks
        devs = []
        for i in range(blocks):
            for j in range(blocks):
                block = diff[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
                devs.append(float(np.mean(block)))
        return np.array(devs)

    # ------------------------------------------------------------------
    # Edge consistency
    # ------------------------------------------------------------------

    def _edge_consistency(self, gray: np.ndarray) -> Dict:
        """
        Detect edge-density inconsistencies across the document.
        Tampered regions may have sharper or blurrier edges than surrounding areas.
        """
        h, w = gray.shape[:2]
        edges = cv2.Canny(gray, 50, 150)

        blocks = 6
        bh, bw = h // blocks, w // blocks
        densities = []
        for i in range(blocks):
            for j in range(blocks):
                block = edges[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
                d = float(np.count_nonzero(block) / max(block.size, 1))
                densities.append(d)

        densities = np.array(densities)
        std = float(np.std(densities))
        mean_d = float(np.mean(densities))

        # High variance in edge density suggests manipulation
        inconsistent = std > 0.06
        penalty = min(15, std * 100) if inconsistent else 0

        return {
            "mean_edge_density": round(mean_d, 4),
            "edge_density_std": round(std, 4),
            "inconsistent": inconsistent,
            "penalty": round(penalty, 1),
        }

    # ------------------------------------------------------------------
    # Noise analysis
    # ------------------------------------------------------------------

    def _noise_analysis(self, gray: np.ndarray) -> Dict:
        """
        Analyse noise patterns.  Genuine scans have relatively uniform
        noise; spliced regions may have different noise characteristics.
        """
        h, w = gray.shape[:2]

        # Estimate noise via high-pass filter
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blurred).astype(np.float64)

        blocks = 6
        bh, bw = h // blocks, w // blocks
        variances = []
        for i in range(blocks):
            for j in range(blocks):
                block = noise[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
                variances.append(float(np.var(block)))

        variances = np.array(variances)
        if np.min(variances) == 0:
            ratio = 1.0
        else:
            ratio = float(np.max(variances) / max(np.min(variances), 1e-6))

        suspicious = ratio > 5.0
        penalty = min(15, (ratio - 5) * 3) if suspicious else 0

        return {
            "noise_variance_mean": round(float(np.mean(variances)), 2),
            "variance_ratio": round(ratio, 2),
            "suspicious": suspicious,
            "penalty": round(max(0, penalty), 1),
        }

    # ------------------------------------------------------------------
    # Copy-paste detection
    # ------------------------------------------------------------------

    def _copy_paste_detection(self, gray: np.ndarray) -> Dict:
        """
        Simplified block-matching to detect cloned regions.
        Divides the image into overlapping blocks and finds near-duplicates.
        """
        try:
            # Downsample for speed
            scale = 0.25
            small = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
            h, w = small.shape[:2]
            block_size = 16
            stride = 8

            blocks = []
            positions = []
            for y in range(0, h - block_size, stride):
                for x in range(0, w - block_size, stride):
                    block = small[y : y + block_size, x : x + block_size]
                    blocks.append(block.flatten().astype(np.float32))
                    positions.append((x, y))

            if len(blocks) < 10:
                return {"suspicious": False, "match_count": 0, "penalty": 0}

            blocks_arr = np.array(blocks)

            # Normalise and find near-duplicates via correlation
            norms = np.linalg.norm(blocks_arr, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalised = blocks_arr / norms

            match_count = 0
            checked = set()
            min_distance = block_size * 2 / scale

            # Sample comparisons to limit computation
            n = len(blocks)
            sample_size = min(n, 500)
            indices = np.random.choice(n, sample_size, replace=False)

            for idx in indices:
                if idx in checked:
                    continue
                # Compute similarity to all other blocks
                sims = normalised[idx] @ normalised.T
                high_sim_indices = np.where(sims > 0.98)[0]
                for other in high_sim_indices:
                    if other == idx:
                        continue
                    dist = np.sqrt(
                        (positions[idx][0] - positions[other][0]) ** 2
                        + (positions[idx][1] - positions[other][1]) ** 2
                    )
                    if dist > min_distance:
                        match_count += 1
                        checked.add(other)

            suspicious = match_count > 5
            penalty = min(20, match_count * 2) if suspicious else 0

            return {
                "match_count": match_count,
                "suspicious": suspicious,
                "penalty": round(penalty, 1),
            }
        except Exception as e:
            return {
                "match_count": 0,
                "suspicious": False,
                "penalty": 0,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Compression analysis
    # ------------------------------------------------------------------

    def _compression_analysis(self, image: np.ndarray) -> Dict:
        """
        Estimate JPEG quality and look for double-compression artefacts.
        """
        try:
            pil_img = Image.fromarray(image)

            # Estimate quality by comparing at different quality levels
            best_quality = 95
            best_diff = float("inf")
            for q in range(50, 100, 5):
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=q)
                buf.seek(0)
                recomp = np.array(Image.open(buf))
                if recomp.shape != image.shape:
                    recomp = cv2.resize(recomp, (image.shape[1], image.shape[0]))
                diff = float(np.mean(cv2.absdiff(image, recomp)))
                if diff < best_diff:
                    best_diff = diff
                    best_quality = q

            suspicious = best_quality < 70
            penalty = max(0, (70 - best_quality) * 0.3) if suspicious else 0

            return {
                "estimated_quality": best_quality,
                "min_diff": round(best_diff, 2),
                "suspicious": suspicious,
                "penalty": round(penalty, 1),
            }
        except Exception as e:
            return {
                "estimated_quality": 95,
                "min_diff": 0,
                "suspicious": False,
                "penalty": 0,
                "error": str(e),
            }
