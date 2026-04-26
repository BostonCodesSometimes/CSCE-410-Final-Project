"""Central utility for creating experiment run directories.

All evaluation and pipeline output must go under:
    <project_root>/results/runs/<YYYYMMDD_HHMMSS>_<tag>/

Expected files per run folder:
    config.json            – serialized PipelineConfig + run metadata
    results.jsonl          – one JSON record per (question, method) pair
    summary.json           – per-method aggregate metrics
    metrics.csv            – flat CSV with one row per (question, method)
    logs.txt               – file-captured log output for the run
    eval_set_snapshot.json – copy of the eval set used (evaluation runs only)

Usage example:
    from run_dir import make_run_dir, write_config, write_summary, \
                        write_metrics_csv, write_eval_set_snapshot, log_path

    run = make_run_dir(tag="eval_001")
    write_config(run, {...})
    write_eval_set_snapshot(run, eval_set)
    # ... run evaluation, collect rows ...
    write_metrics_csv(run, csv_rows)
    write_summary(run, aggregate)
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

# Project root is two levels above this file:  src/run_dir.py  →  src/  →  project/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_ROOT = _PROJECT_ROOT / "results" / "runs"


def make_run_dir(tag: str = "run", root: str | Path | None = None) -> Path:
    """Create and return a new timestamped run directory.

    The directory is created immediately so callers can start writing files
    (e.g. logs.txt) before the run begins.

    Args:
        tag:  Short human-readable label appended after the timestamp,
              e.g. "eval_001"   → "20260423_150102_eval_001"
              e.g. "litepack_budget1800" → "20260423_150102_litepack_budget1800"
        root: Parent directory. Defaults to <project_root>/results/runs/.

    Returns:
        Path to the newly created (empty) directory.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{tag}" if tag else ts
    parent = Path(root) if root else _RUNS_ROOT
    run_path = parent / name
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def write_config(run_dir: Path, config_dict: dict) -> None:
    """Write config.json to *run_dir*."""
    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config_dict, fh, indent=2)


def write_summary(run_dir: Path, summary_dict: dict) -> None:
    """Write summary.json to *run_dir*."""
    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_dict, fh, indent=2)


def write_metrics_csv(
    run_dir: Path,
    rows: list[dict],
    filename: str = "metrics.csv",
) -> None:
    """Write a list of flat metric dicts as a CSV file to *run_dir*.

    All rows must share the same keys as the first row.  Extra keys in later
    rows are silently dropped; missing keys get an empty string.
    """
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(run_dir / filename, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_eval_set_snapshot(run_dir: Path, eval_set: list[dict]) -> None:
    """Write a snapshot copy of the eval set to *run_dir*."""
    with open(run_dir / "eval_set_snapshot.json", "w", encoding="utf-8") as fh:
        json.dump(eval_set, fh, indent=2, ensure_ascii=False)


def log_path(run_dir: Path) -> Path:
    """Return the canonical path to logs.txt inside *run_dir*."""
    return run_dir / "logs.txt"
