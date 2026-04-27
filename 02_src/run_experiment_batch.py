from __future__ import annotations

import sys

from eskf_stack.analysis.experiment_batch import ExperimentBatchResult, run_experiment_batch


def _result_output_lines(result: ExperimentBatchResult) -> list[str]:
    preview_text = ", ".join(result.key_summary_preview_columns)
    if len(result.key_summary_columns) > len(result.key_summary_preview_columns):
        preview_text = f"{preview_text}, ..."
    category_preview_text = ", ".join(result.category_summary_preview_names)
    if len(result.category_summary_names) > len(result.category_summary_preview_names):
        category_preview_text = f"{category_preview_text}, ..."
    lines = [
        f"Experiment batch finished. Runs: {result.run_count}",
        f"Summary: {result.summary_path}",
        f"Key summary: {result.key_summary_path}",
        f"Manifest: {result.manifest_path}",
        f"Key summary columns: {len(result.key_summary_columns)}",
        f"Key summary preview: {preview_text}" if preview_text else "Key summary preview: none",
    ]
    if result.category_summary_paths:
        lines.append(
            f"Category summary names: {category_preview_text}" if category_preview_text else "Category summary names: none"
        )
        lines.append("Category summaries:")
        for category_name, summary_path in sorted(result.category_summary_paths.items()):
            metric_count = len(result.category_metric_columns.get(category_name, []))
            lines.append(f"- {category_name} ({metric_count} metrics): {summary_path}")
    else:
        lines.append("Category summaries: none")
    return lines


def main(argv: list[str] | None = None) -> int:
    config_paths = argv if argv else None
    result = run_experiment_batch(config_paths=config_paths)
    for line in _result_output_lines(result):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
