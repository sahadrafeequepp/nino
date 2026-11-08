"""Regression report generator for Nino fine-tuning evaluation results."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvaluationResult:
    prompt_id: str
    category: str
    prompt: str
    base_output: str
    tuned_output: str
    expected_rule: str
    passed: bool
    failure_reason: str = ""


class RegressionReportGenerator:
    """Generates markdown and JSON evaluation reports with side-by-side comparisons."""

    @staticmethod
    def generate_markdown(results: list[EvaluationResult], base_name: str = "Base", tuned_name: str = "Tuned") -> str:
        """Generate formatted markdown report."""
        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        failed_count = total - passed_count
        pass_rate = (passed_count / total * 100) if total > 0 else 0.0

        lines: list[str] = [
            "# Nino Fine-Tuning Regression & Quality Evaluation Report",
            "",
            f"- **Total Prompts Evaluated**: {total}",
            f"- **Passed Constraints**: {passed_count} ({pass_rate:.1f}%)",
            f"- **Regressions / Failures**: {failed_count}",
            "",
            "## Summary Table",
            "",
            "| Prompt ID | Category | Expected Rule | Status | Failure Reason |",
            "| :--- | :--- | :--- | :---: | :--- |",
        ]

        for r in results:
            status_icon = "✅ PASS" if r.passed else "❌ FAIL"
            reason = r.failure_reason if not r.passed else "Complies with constraints"
            lines.append(f"| `{r.prompt_id}` | {r.category} | `{r.expected_rule}` | {status_icon} | {reason} |")

        lines.extend([
            "",
            "---",
            "",
            "## Side-by-Side Output Comparisons",
            "",
        ])

        for r in results:
            status_text = "PASSED" if r.passed else f"FAILED ({r.failure_reason})"
            lines.extend([
                f"### Prompt: `{r.prompt_id}` ({r.category})",
                f"**User Prompt**: *\"{r.prompt}\"*",
                f"**Verification**: `{status_text}`",
                "",
                f"#### {base_name} Model Output:",
                "```text",
                r.base_output.strip(),
                "```",
                "",
                f"#### {tuned_name} Model Output:",
                "```text",
                r.tuned_output.strip(),
                "```",
                "",
                "---",
                "",
            ])

        return "\n".join(lines)

    @staticmethod
    def save_report(
        results: list[EvaluationResult],
        md_path: str | Path,
        json_path: str | Path | None = None,
        base_name: str = "Base",
        tuned_name: str = "Tuned",
    ) -> None:
        """Save report to disk in markdown and optionally JSON format."""
        md_file = Path(md_path)
        md_file.parent.mkdir(parents=True, exist_ok=True)
        md_content = RegressionReportGenerator.generate_markdown(results, base_name, tuned_name)
        md_file.write_text(md_content, encoding="utf-8")

        if json_path:
            json_file = Path(json_path)
            json_file.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(r) for r in results]
            json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


class EvaluationReport:
    """Convenience wrapper for evaluation result analysis and persistence."""

    def __init__(self, results: list[EvaluationResult]):
        self.results = results

    def get_summary(self) -> dict[str, Any]:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total) if total > 0 else 0.0,
        }

    def save_markdown(self, path: str | Path, base_name: str = "Base", tuned_name: str = "Tuned") -> None:
        RegressionReportGenerator.save_report(self.results, md_path=path, base_name=base_name, tuned_name=tuned_name)

    def save_json(self, path: str | Path) -> None:
        json_file = Path(path)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in self.results]
        json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
