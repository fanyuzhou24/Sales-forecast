from __future__ import annotations

from pathlib import Path

import pandas as pd


class ReportExporter:
    """预测评估报告导出：Excel + PDF(简版文本)。"""

    def export_excel(self, report_df: pd.DataFrame, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_excel(output_path, index=False)
        return output_path

    def export_pdf(self, report_df: pd.DataFrame, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["Sales Forecast Report", "=" * 40]
        lines.extend(report_df.to_string(index=False).splitlines())
        # 轻量方案：输出 .pdf 后缀文本文件；生产环境可替换为 reportlab/WeasyPrint。
        output_path.write_text("\\n".join(lines), encoding="utf-8")
        return output_path

    def share(self, file_path: Path, roles: list[str]) -> dict:
        return {
            "file": str(file_path),
            "shared_roles": roles,
            "status": "shared",
        }

    def export_review_report(
        self,
        detail_df: pd.DataFrame,
        bias_breakdown: dict,
        recommendations: list[str],
        actions: list[dict],
        output_dir: Path,
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        detail_path = output_dir / "review_detail.xlsx"
        detail_df.to_excel(detail_path, index=False)

        analysis_path = output_dir / "review_analysis.pdf"
        lines = [
            "Forecast Review Analysis",
            "=" * 50,
            "Bias Breakdown:",
            str(bias_breakdown),
            "",
            "Recommendations:",
            *[f"- {x}" for x in recommendations],
            "",
            "Action Tracking:",
            str(actions),
        ]
        analysis_path.write_text("\\n".join(lines), encoding="utf-8")
        return {
            "detail_excel": str(detail_path),
            "analysis_pdf": str(analysis_path),
        }
