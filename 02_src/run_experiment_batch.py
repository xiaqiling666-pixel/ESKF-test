from __future__ import annotations

import sys

from eskf_stack.analysis.experiment_batch import run_experiment_batch


if __name__ == "__main__":
    config_paths = sys.argv[1:] or None
    result = run_experiment_batch(config_paths=config_paths)
    print(f"Experiment batch finished. Runs: {result.run_count}")
    print(f"Summary: {result.summary_path}")
    print(f"Key summary: {result.key_summary_path}")
