"""
Document Processing Module
==========================
Handles PDF rendering, metadata extraction, file validation, and multi-page support.
Uses PyMuPDF (fitz) for PDF operations and Pillow for image handling.
"""

import fitz  # PyMuPDF
import os
import io
import numpy as np
from PIL import Image
from typing import Dict, List, Optional


class DocumentProcessor:
    """Processes uploaded certificate documents (PDF, JPG, PNG)."""

    SUPPORTED_FORMATS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}
    SUSPICIOUS_SOFTWARE = [
        "photoshop", "gimp", "paint.net", "illustrator", "inkscape",
        "canva", "pixlr", "affinity", "corel", "paintshop",
    ]
    TRUSTED_PRODUCERS = [
        "microsoft", "adobe acrobat", "libreoffice", "openoffice",
        "google docs", "latex", "tex", "quartz",
    ]

    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # File validation
    # ------------------------------------------------------------------
    def validate_file(self, file_path: str) -> Dict:
        """Validate the uploaded certificate file and return a status dict."""
        result = {
            "valid": False,
            "file_type": None,
            "file_size": 0,
            "issues": [],
        }

        if not os.path.exists(file_path):
            result["issues"].append("File does not exist")
            return result

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            result["issues"].append(f"Unsupported format: {ext}")
            return result

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            result["issues"].append("File is empty")
            return result
        if file_size > 50 * 1024 * 1024:
            result["issues"].append("File too large (max 50 MB)")
            return result

        result["valid"] = True
        result["file_type"] = ext
        result["file_size"] = file_size
        return result

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------
    def extract_metadata(self, file_path: str) -> Dict:
        """Extract and analyse document metadata."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._extract_pdf_metadata(file_path)
        return self._extract_image_metadata(file_path)

    def _extract_pdf_metadata(self, file_path: str) -> Dict:
        metadata = {
            "format": "PDF",
            "creator": None,
            "producer": None,
            "creation_date": None,
            "modification_date": None,
            "page_count": 0,
            "suspicious_flags": [],
            "metadata_score": 100,
            "raw_metadata": {},
        }
        try:
            doc = fitz.open(file_path)
            meta = doc.metadata or {}
            metadata["raw_metadata"] = {k: v for k, v in meta.items() if v}
            metadata["page_count"] = len(doc)
            metadata["creator"] = meta.get("creator", "")
            metadata["producer"] = meta.get("producer", "")
            metadata["creation_date"] = meta.get("creationDate", "")
            metadata["modification_date"] = meta.get("modDate", "")

            # --- suspicious software check ---
            creator_lower = (meta.get("creator", "") or "").lower()
            producer_lower = (meta.get("producer", "") or "").lower()
            combined = creator_lower + " " + producer_lower

            for software in self.SUSPICIOUS_SOFTWARE:
                if software in combined:
                    metadata["suspicious_flags"].append(
                        f"Created/modified with image-editing software: {software}"
                    )
                    metadata["metadata_score"] -= 25

            # --- modification check ---
            creation = meta.get("creationDate", "")
            modification = meta.get("modDate", "")
            if creation and modification and creation != modification:
                metadata["suspicious_flags"].append(
                    "Document has been modified after initial creation"
                )
                metadata["metadata_score"] -= 10

            # --- trusted producer bonus ---
            is_trusted = any(t in combined for t in self.TRUSTED_PRODUCERS)
            if is_trusted:
                metadata["metadata_score"] = min(100, metadata["metadata_score"] + 5)

            # --- missing metadata ---
            if not meta.get("creator") and not meta.get("producer"):
                metadata["suspicious_flags"].append("No creator/producer metadata found")
                metadata["metadata_score"] -= 5

            # --- excessive page count ---
            if len(doc) > 5:
                metadata["suspicious_flags"].append(
                    f"Unusually high page count for a certificate ({len(doc)} pages)"
                )
                metadata["metadata_score"] -= 10

            doc.close()
        except Exception as e:
            metadata["suspicious_flags"].append(f"Error reading PDF metadata: {e}")
            metadata["metadata_score"] -= 30

        metadata["metadata_score"] = max(0, min(100, metadata["metadata_score"]))
        return metadata

    def _extract_image_metadata(self, file_path: str) -> Dict:
        metadata = {
            "format": "Image",
            "creator": None,
            "producer": None,
            "creation_date": None,
            "modification_date": None,
            "page_count": 1,
            "suspicious_flags": [],
            "metadata_score": 100,
            "raw_metadata": {},
        }
        try:
            img = Image.open(file_path)
            metadata["raw_metadata"] = {
                "size": list(img.size),
                "mode": img.mode,
                "format": img.format,
            }
            exif = img.getexif() if hasattr(img, "getexif") else {}
            if exif:
                metadata["raw_metadata"]["exif_keys"] = list(
                    str(k) for k in exif.keys()
                )
                software = exif.get(305, "")
                if software:
                    metadata["creator"] = software
                    for sus in self.SUSPICIOUS_SOFTWARE:
                        if sus in software.lower():
                            metadata["suspicious_flags"].append(
                                f"Image edited with: {software}"
                            )
                            metadata["metadata_score"] -= 25
            img.close()
        except Exception as e:
            metadata["suspicious_flags"].append(f"Error reading image metadata: {e}")
            metadata["metadata_score"] -= 30

        metadata["metadata_score"] = max(0, min(100, metadata["metadata_score"]))
        return metadata

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_to_images(self, file_path: str, dpi: int = 300) -> List[np.ndarray]:
        """Convert document pages to high-resolution NumPy images (RGB)."""
        ext = os.path.splitext(file_path)[1].lower()
        images: List[np.ndarray] = []

        if ext == ".pdf":
            doc = fitz.open(file_path)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            for page in doc:
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(np.array(img.convert("RGB")))
            doc.close()
        else:
            img = Image.open(file_path)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            images.append(np.array(img))

        return images

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------
    def get_quality_metrics(self, image: np.ndarray) -> Dict:
        """Assess basic image quality metrics."""
        h, w = image.shape[:2]
        quality = {
            "resolution": f"{w}x{h}",
            "width": w,
            "height": h,
            "is_high_res": w >= 1000 and h >= 1000,
            "aspect_ratio": round(w / max(h, 1), 2),
            "quality_score": 100,
        }
        if w < 500 or h < 500:
            quality["quality_score"] -= 30
            quality["warning"] = "Low resolution – analysis may be less accurate"
        elif w < 1000 or h < 1000:
            quality["quality_score"] -= 10
        return quality
