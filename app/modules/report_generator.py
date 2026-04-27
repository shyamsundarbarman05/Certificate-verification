"""
Report Generator Module
========================
Compiles all analysis results into structured verification reports:
  • JSON report for API consumers
  • Summary for the frontend dashboard
  • Detailed finding descriptions
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any


class ReportGenerator:
    """Generate comprehensive verification reports from analysis results."""

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, analysis_results: Dict) -> Dict:
        """
        Compile all module outputs into a unified verification report.

        Parameters
        ----------
        analysis_results : dict
            Must contain keys: metadata, ocr, layout, signature, forensic, classification

        Returns
        -------
        dict with report_id, summary, detailed findings, and file path.
        """
        report_id = str(uuid.uuid4())[:12]
        timestamp = datetime.now(timezone.utc).isoformat()

        classification = analysis_results.get("classification", {})
        metadata_result = analysis_results.get("metadata", {})
        ocr_result = analysis_results.get("ocr", {})
        layout_result = analysis_results.get("layout", {})
        sig_result = analysis_results.get("signature", {})
        forensic_result = analysis_results.get("forensic", {})

        # --- Build summary ---
        authenticity_score = classification.get("authenticity_score", 0)
        risk_level = classification.get("risk_level", "Unknown")

        # Force 0 % display for forged certificates
        if risk_level == "Likely Forged":
            authenticity_score = 0

        summary = self._build_summary(
            authenticity_score, risk_level,
            metadata_result, ocr_result, layout_result,
            sig_result, forensic_result,
        )

        # --- Build detailed findings ---
        findings = self._build_findings(
            metadata_result, ocr_result, layout_result,
            sig_result, forensic_result,
        )

        # --- Module scores table ---
        module_scores = {
            "OCR Quality":            ocr_result.get("ocr_score", 0),
            "Layout Consistency":     layout_result.get("layout_score", 0),
            "Metadata Integrity":     metadata_result.get("metadata_score", 0),
            "Signature Authenticity": sig_result.get("signature_score", 0),
            "Seal Verification":      sig_result.get("seal_score", 0),
            "Forensic Analysis":      forensic_result.get("forensic_score", 0),
        }

        # For forged certificates show 0 across the board
        if risk_level == "Likely Forged":
            module_scores = {k: 0 for k in module_scores}

        # --- Assemble report ---
        report = {
            "report_id": report_id,
            "timestamp": timestamp,
            "authenticity_score": authenticity_score,
            "risk_level": risk_level,
            "confidence": classification.get("confidence", 0),
            "method": classification.get("method", "heuristic_ensemble"),
            "summary": summary,
            "module_scores": module_scores,
            "findings": findings,
            "extracted_data": {
                "text_fields": ocr_result.get("structured", {}),
                "average_ocr_confidence": ocr_result.get("average_confidence", 0),
            },
            "metadata": {
                "format": metadata_result.get("format", "Unknown"),
                "page_count": metadata_result.get("page_count", 0),
                "creator": metadata_result.get("creator", ""),
                "producer": metadata_result.get("producer", ""),
                "suspicious_flags": metadata_result.get("suspicious_flags", []),
            },
            "classification_details": classification,
        }

        # --- Save to disk ---
        file_path = os.path.join(self.reports_dir, f"report_{report_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            report["file_path"] = file_path
        except Exception as e:
            report["file_path"] = None
            report["save_error"] = str(e)

        return report

    # ------------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        score: float,
        risk: str,
        metadata: Dict,
        ocr: Dict,
        layout: Dict,
        sig: Dict,
        forensic: Dict,
    ) -> str:
        """Build a human-readable summary paragraph."""
        parts: List[str] = []

        parts.append(
            f"Certificate authenticity assessment complete. "
            f"Overall score: {score:.0f}/100 — Risk level: {risk}."
        )

        # Highlight key issues
        all_anomalies: List[str] = []
        all_anomalies.extend(metadata.get("suspicious_flags", []))
        all_anomalies.extend(layout.get("anomalies", []))
        all_anomalies.extend(sig.get("anomalies", []))
        all_anomalies.extend(forensic.get("anomalies", []))

        if all_anomalies:
            parts.append(
                f"{len(all_anomalies)} potential issue(s) were identified during analysis."
            )
        else:
            parts.append(
                "No significant anomalies were detected during the verification process."
            )

        # OCR completeness
        fields = ocr.get("structured", {})
        filled = sum(1 for v in fields.values() if v)
        total = len(fields)
        parts.append(
            f"OCR extraction identified {filled}/{total} expected certificate fields."
        )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Findings builder
    # ------------------------------------------------------------------

    def _build_findings(
        self,
        metadata: Dict,
        ocr: Dict,
        layout: Dict,
        sig: Dict,
        forensic: Dict,
    ) -> List[Dict]:
        """Build a list of structured finding objects."""
        findings: List[Dict] = []

        # Metadata findings
        for flag in metadata.get("suspicious_flags", []):
            findings.append({
                "module": "Metadata Analysis",
                "severity": self._flag_severity(flag),
                "description": flag,
                "score_impact": "Negative",
            })

        # OCR findings
        field_analysis = ocr.get("field_analysis", {})
        for field, info in field_analysis.items():
            if field.startswith("_"):
                findings.append({
                    "module": "OCR Analysis",
                    "severity": "warning",
                    "description": info.get("issue", "Low-confidence text regions"),
                    "score_impact": "Negative",
                })
            elif info.get("status") == "missing" and info.get("required"):
                findings.append({
                    "module": "OCR Analysis",
                    "severity": "warning",
                    "description": f"Required field '{field}' was not found",
                    "score_impact": "Negative",
                })

        # Layout findings
        for anomaly in layout.get("anomalies", []):
            findings.append({
                "module": "Layout Analysis",
                "severity": "warning",
                "description": anomaly,
                "score_impact": "Negative",
            })

        # Signature / seal findings
        for anomaly in sig.get("anomalies", []):
            sev = "critical" if "not match" in anomaly.lower() else "warning"
            findings.append({
                "module": "Signature & Seal Verification",
                "severity": sev,
                "description": anomaly,
                "score_impact": "Negative",
            })

        # Forensic findings
        for anomaly in forensic.get("anomalies", []):
            sev = "critical" if "tamper" in anomaly.lower() else "warning"
            findings.append({
                "module": "Forensic Analysis",
                "severity": sev,
                "description": anomaly,
                "score_impact": "Negative",
            })

        # If no findings, add a positive note
        if not findings:
            findings.append({
                "module": "Overall",
                "severity": "info",
                "description": "All checks passed — no anomalies detected.",
                "score_impact": "Positive",
            })

        return findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flag_severity(flag: str) -> str:
        flag_lower = flag.lower()
        if any(w in flag_lower for w in ["editing", "photoshop", "gimp"]):
            return "critical"
        if "modified" in flag_lower:
            return "warning"
        return "info"
