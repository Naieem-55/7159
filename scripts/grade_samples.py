"""CLI entry point: `python -m scripts.grade_samples`

Grades every file in assignments/ and writes a Markdown report per student
into reports/, plus a combined summary.
"""
from __future__ import annotations

import json

from backend.config import settings
from backend.grading import grade_submission
from backend.models import GradeReport


def _report_to_markdown(report: GradeReport) -> str:
    lines = [f"# Grade Report — {report.student_name}", ""]
    for c in report.criteria:
        lines.append(f"## {c.criterion} ({c.awarded_marks}/{c.max_marks})")
        lines.append(f"- Justification: {c.justification}")
        if c.book_reference:
            lines.append(f"- Book reference: {c.book_reference}")
        lines.append("")
    lines.append(f"## Total: {report.total_score}/{report.max_score}")
    lines.append("")
    lines.append(f"## Overall feedback\n{report.overall_feedback}")
    lines.append("")
    if report.flags:
        lines.append("## Flags")
        for f in report.flags:
            lines.append(f"- {f}")
        lines.append("")
    lines.append("## Retrieved evidence used")
    for e in report.retrieved_evidence:
        lines.append(f"- [{e.chapter}, similarity={e.similarity}] {e.text[:150]}...")
    return "\n".join(lines)


def main() -> None:
    settings.reports_dir.mkdir(exist_ok=True)
    summary = []
    for path in sorted(settings.assignments_dir.glob("*.txt")):
        student_name = path.stem
        answer_text = path.read_text(encoding="utf-8")
        print(f"Grading {student_name}...")
        report = grade_submission(student_name, answer_text)

        md_path = settings.reports_dir / f"{student_name}.md"
        md_path.write_text(_report_to_markdown(report), encoding="utf-8")

        json_path = settings.reports_dir / f"{student_name}.json"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        summary.append({"student": student_name, "total": report.total_score,
                         "max": report.max_score, "flags": report.flags})
        print(f"  -> {report.total_score}/{report.max_score}  (flags: {len(report.flags)})")

    (settings.reports_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDone. See reports/ for full output.")


if __name__ == "__main__":
    main()
