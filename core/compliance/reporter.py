#!/usr/bin/env python3

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import csv
import plotly.graph_objects as go
import plotly.express as px
from jinja2 import Environment, FileSystemLoader, select_autoescape

from core.compliance.analyzer import ComplianceResult, ComplianceLevel, ComplianceViolation
from config.settings import Settings

class ReportFormat(Enum):
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"

class ComplianceReporter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.template_dir = Path(__file__).parent / 'templates'
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml'])
        )

    def generate_report(
        self,
        results: Union[ComplianceResult, List[ComplianceResult]],
        format: ReportFormat,
        output_path: Path,
        template_name: Optional[str] = None
    ) -> Path:
        """Generate a compliance report in the specified format.

        Args:
            results: Single or multiple compliance results
            format: Output format (HTML, PDF, JSON, CSV)
            output_path: Where to save the report
            template_name: Optional custom template name

        Returns:
            Path to the generated report
        """
        if isinstance(results, ComplianceResult):
            results = [results]

        report_data = self._prepare_report_data(results)

        if format == ReportFormat.HTML:
            return self._generate_html_report(report_data, output_path, template_name)
        elif format == ReportFormat.PDF:
            return self._generate_pdf_report(report_data, output_path, template_name)
        elif format == ReportFormat.JSON:
            return self._generate_json_report(report_data, output_path)
        elif format == ReportFormat.CSV:
            return self._generate_csv_report(report_data, output_path)
        else:
            raise ValueError(f"Unsupported report format: {format}")

    def _prepare_report_data(self, results: List[ComplianceResult]) -> Dict:
        """Prepare data for report generation."""
        total_violations = sum(len(result.violations) for result in results)
        compliance_stats = self._calculate_compliance_stats(results)
        severity_stats = self._calculate_severity_stats(results)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_documents": len(results),
            "total_violations": total_violations,
            "compliance_stats": compliance_stats,
            "severity_stats": severity_stats,
            "results": [
                self._format_result(result) for result in results
            ]
        }

    def _calculate_compliance_stats(self, results: List[ComplianceResult]) -> Dict[str, int]:
        """Calculate statistics about compliance levels."""
        stats = {level.value: 0 for level in ComplianceLevel}
        for result in results:
            stats[result.compliance_level.value] += 1
        return stats

    def _calculate_severity_stats(self, results: List[ComplianceResult]) -> Dict[str, int]:
        """Calculate statistics about violation severities."""
        stats = {"High": 0, "Medium": 0, "Low": 0}
        for result in results:
            for violation in result.violations:
                stats[violation.severity] += 1
        return stats

    def _format_result(self, result: ComplianceResult) -> Dict:
        """Format a single compliance result for reporting."""
        return {
            "document_id": result.document_id,
            "timestamp": result.timestamp.isoformat(),
            "compliance_level": result.compliance_level.value,
            "violations": [
                self._format_violation(v) for v in result.violations
            ],
            "summary": result.summary,
            "metadata": result.metadata
        }

    def _format_violation(self, violation: ComplianceViolation) -> Dict:
        """Format a single violation for reporting."""
        return {
            "rule_id": violation.rule_id,
            "location": violation.location,
            "description": violation.description,
            "severity": violation.severity,
            "suggested_fixes": violation.suggested_fixes
        }

    def _generate_html_report(self, data: Dict, output_path: Path, template_name: Optional[str]) -> Path:
        """Generate an HTML report using templates."""
        template_name = template_name or 'compliance_report.html'
        template = self.jinja_env.get_template(template_name)

        # Generate charts
        compliance_chart = self._create_compliance_chart(data['compliance_stats'])
        severity_chart = self._create_severity_chart(data['severity_stats'])

        html_content = template.render(
            report_data=data,
            compliance_chart=compliance_chart.to_html(full_html=False),
            severity_chart=severity_chart.to_html(full_html=False)
        )

        output_path = output_path.with_suffix('.html')
        output_path.write_text(html_content)
        return output_path

    def _generate_pdf_report(self, data: Dict, output_path: Path, template_name: Optional[str]) -> Path:
        """Generate a PDF report."""
        # First generate HTML
        html_path = self._generate_html_report(data, output_path.with_suffix('.html'), template_name)

        # Convert HTML to PDF using weasyprint
        from weasyprint import HTML
        pdf_path = output_path.with_suffix('.pdf')
        HTML(str(html_path)).write_pdf(str(pdf_path))
        return pdf_path

    def _generate_json_report(self, data: Dict, output_path: Path) -> Path:
        """Generate a JSON report."""
        output_path = output_path.with_suffix('.json')
        with output_path.open('w') as f:
            json.dump(data, f, indent=2)
        return output_path

    def _generate_csv_report(self, data: Dict, output_path: Path) -> Path:
        """Generate a CSV report focusing on violations."""
        output_path = output_path.with_suffix('.csv')
        with output_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Document ID', 'Compliance Level', 'Rule ID',
                'Location', 'Severity', 'Description', 'Suggested Fixes'
            ])

            for result in data['results']:
                for violation in result['violations']:
                    writer.writerow([
                        result['document_id'],
                        result['compliance_level'],
                        violation['rule_id'],
                        violation['location'],
                        violation['severity'],
                        violation['description'],
                        '; '.join(violation['suggested_fixes'])
                    ])
        return output_path

    def _create_compliance_chart(self, compliance_stats: Dict[str, int]) -> go.Figure:
        """Create a pie chart showing compliance level distribution."""
        return px.pie(
            values=list(compliance_stats.values()),
            names=list(compliance_stats.keys()),
            title='Compliance Level Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )

    def _create_severity_chart(self, severity_stats: Dict[str, int]) -> go.Figure:
        """Create a bar chart showing violation severity distribution."""
        return px.bar(
            x=list(severity_stats.keys()),
            y=list(severity_stats.values()),
            title='Violation Severity Distribution',
            labels={'x': 'Severity', 'y': 'Number of Violations'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )