"""Evaluation runner for LitePack-RAG Phase 8.

Responsibilities:
  - load and validate eval_data.json
  - run selected methods over every question using compare_methods()
  - compute per-run metrics using evaluate_run()
  - write a lightweight JSONL output file (no embedding arrays)
  - print an aggregate summary table by method

This module has its own argparse CLI and is intentionally separate from
pipeline.py.  Do NOT add evaluation logic to pipeline.py.

Citation spot-check note (manual, not automated):
  After running evaluation, inspect a small sample of runs where
  cited_source_labels is non-empty.  For each [Sn] label in the answer,
  look up the PMID via citation_label_to_pmid and visit
  https://pubmed.ncbi.nlm.nih.gov/{pmid}/ to verify the claim.
  Record each citation as supports / partial / does_not_support.
  This is a HUMAN SPOT-CHECK, not a formal metric.  Do not automate it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=False)

from config import PipelineConfig
from retriever import Retriever
from generator import Generator
from pipeline import compare_methods
from metrics import evaluate_run
from run_dir import (
    make_run_dir,
    write_config,
    write_summary,
    write_metrics_csv,
    write_eval_set_snapshot,
    log_path,
)

logger = logging.getLogger(__name__)

# Metric keys that are averaged in the aggregate table (order matters for display)
_NUMERIC_METRIC_KEYS: list[str] = [
    "recall_at_5",
    "ndcg_at_5",
    "evidence_hit_rate",
    "redundancy_score",
    "avg_pairwise_similarity",
    "prompt_token_count",
    "budget_utilization",
    "rouge_l",
    "keyword_support_rate",
]


# --------------------------------------------------------------------------- #
# Dataset loading                                                              #
# --------------------------------------------------------------------------- #

def load_eval_set(path: str) -> list[dict]:
    """Load and validate an eval_data.json file.

    Each entry must contain at minimum:
      - "question"   (non-empty string)
      - "gold_pmids" (list of strings, may be empty)

    Optional fields that are preserved if present:
      - "reference_answer"  (defaults to "" if absent)
      - "question_type"
      - "keywords"

    Args:
        path: path to the JSON file.

    Returns:
        List of validated dicts, each guaranteed to have "reference_answer".

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if the top-level structure is not a list, or if any
                           entry is missing required fields.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Eval set not found: {path!r}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(
            f"eval_data.json must be a JSON array; got {type(data).__name__}"
        )

    validated: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {i} is not a dict: {item!r}")
        if "question" not in item or not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(f"Entry {i} is missing or has empty 'question'")
        if "gold_pmids" not in item or not isinstance(item["gold_pmids"], list):
            raise ValueError(f"Entry {i} is missing or has non-list 'gold_pmids'")

        entry = dict(item)
        entry.setdefault("reference_answer", "")
        validated.append(entry)

    logger.info("load_eval_set: loaded %d entries from %r", len(validated), path)
    return validated


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #

def _aggregate_results(results: list[dict]) -> dict[str, dict]:
    """Group metric dicts by method_name and compute per-method averages.

    Args:
        results: list of per-run metric dicts from evaluate_run().

    Returns:
        Dict mapping method_name -> {"runs": int, "means": {metric: float}}.
    """
    groups: dict[str, list[dict]] = {}
    for r in results:
        name = r.get("method_name", "unknown")
        groups.setdefault(name, []).append(r)

    aggregate: dict[str, dict] = {}
    for method, runs in groups.items():
        means: dict[str, float] = {}
        for key in _NUMERIC_METRIC_KEYS:
            values = [r[key] for r in runs if key in r and r[key] is not None]
            means[key] = statistics.mean(values) if values else 0.0
        aggregate[method] = {"runs": len(runs), "means": means}

    return aggregate


# --------------------------------------------------------------------------- #
# Printing                                                                     #
# --------------------------------------------------------------------------- #

def _print_aggregate_table(aggregate: dict[str, dict]) -> None:
    """Print one row per method with averaged metrics.

    Columns:
      METHOD  RUNS  recall@5  ndcg@5  hit_rate  redund  sim  tokens  util  rouge_l  kw_supp
    """
    if not aggregate:
        print("(no results to aggregate)")
        return

    header = (
        f"{'METHOD':<14} {'RUNS':>4}  "
        f"{'rec@5':>6}  {'ndcg@5':>6}  {'hit_rt':>6}  "
        f"{'redund':>6}  {'sim':>6}  "
        f"{'tokens':>7}  {'util':>5}  "
        f"{'rouge_l':>7}  {'kw_sup':>6}"
    )
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("AGGREGATE METRICS BY METHOD")
    print("=" * len(header))
    print(header)
    print(sep)

    for method, agg in sorted(aggregate.items()):
        m = agg["means"]
        print(
            f"{method:<14} {agg['runs']:>4}  "
            f"{m.get('recall_at_5', 0.0):>6.3f}  "
            f"{m.get('ndcg_at_5', 0.0):>6.3f}  "
            f"{m.get('evidence_hit_rate', 0.0):>6.3f}  "
            f"{m.get('redundancy_score', 0.0):>6.3f}  "
            f"{m.get('avg_pairwise_similarity', 0.0):>6.3f}  "
            f"{m.get('prompt_token_count', 0):>7.0f}  "
            f"{m.get('budget_utilization', 0.0):>5.3f}  "
            f"{m.get('rouge_l', 0.0):>7.4f}  "
            f"{m.get('keyword_support_rate', 0.0):>6.3f}"
        )

    print("=" * len(header))
    print()


# --------------------------------------------------------------------------- #
# JSONL serialization helpers                                                  #
# --------------------------------------------------------------------------- #

def _build_record(run_result: Any, metric_dict: dict) -> dict:
    """Build a lightweight, JSON-serializable record for one run.

    No embedding arrays are included.

    Args:
        run_result:  RunResult from pipeline.
        metric_dict: output of evaluate_run().

    Returns:
        Flat dict safe for json.dumps().
    """
    pc = run_result.packed_context
    gen = run_result.generation_result
    sr = run_result.selection_result

    # citation_label_to_pmid: {"S1": "38291047", "S2": "35714220"}
    citation_label_to_pmid: dict[str, str] = {
        label: cand.pmid
        for label, cand in pc.citation_map.items()
    }

    return {
        "question": run_result.query,
        "method_name": run_result.method_name,
        "budget": run_result.config.budget,
        "tokens_used": pc.tokens_used,
        "chunks_selected": metric_dict.get("chunks_selected", 0),
        "chunks_retrieved": metric_dict.get("chunks_retrieved", 0),
        "selected_pmids": list(pc.selected_pmids),
        "cited_source_labels": list(gen.cited_source_labels),
        "citation_label_to_pmid": citation_label_to_pmid,
        "empty_retrieval": metric_dict.get("empty_retrieval", False),
        "empty_selection": metric_dict.get("empty_selection", False),
        "metrics": {k: metric_dict[k] for k in _NUMERIC_METRIC_KEYS if k in metric_dict},
        "timings": run_result.timings,
    }


# --------------------------------------------------------------------------- #
# run_evaluation                                                               #
# --------------------------------------------------------------------------- #

def run_evaluation(
    eval_set: list[dict],
    methods: list[str],
    config: PipelineConfig,
    retriever: Retriever,
    embeddings_model: Any,
    generator: Generator,
    output_dir: str,
) -> None:
    """Run all methods over every question and write results.

    For each (question, method) pair:
      1. Call compare_methods() once per question (all methods in one batch).
      2. Call evaluate_run() on each RunResult.
      3. Build a lightweight JSONL record.

    Writes:
      {output_dir}/results.jsonl  — one JSON object per line

    Prints the aggregate summary table to stdout after all questions.

    Args:
        eval_set:         output of load_eval_set().
        methods:          list of method names e.g. ["similarity", "mmr", "litepack"].
        config:           base PipelineConfig (not mutated; per-method copies made inside compare_methods).
        retriever:        initialized Retriever instance.
        embeddings_model: LangChain embeddings object (retriever.embeddings_model).
        generator:        initialized Generator instance.
        output_dir:       directory for output files; created if absent.
    """
    run_path = Path(output_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # --- Config snapshot ---
    write_config(run_path, {**dataclasses.asdict(config), "methods": methods})

    # --- Eval-set snapshot ---
    write_eval_set_snapshot(run_path, eval_set)

    # --- File logging ---
    _log_file = log_path(run_path)
    _file_handler = logging.FileHandler(_log_file, encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(_file_handler)

    output_path = run_path / "results.jsonl"
    all_metric_dicts: list[dict] = []
    csv_rows: list[dict] = []

    with open(output_path, "w", encoding="utf-8") as out_fh:
        for q_idx, entry in enumerate(eval_set):
            question = entry["question"]
            gold_pmids = entry["gold_pmids"]
            reference_answer = entry.get("reference_answer", "")

            logger.info(
                "Evaluating question %d/%d: %r",
                q_idx + 1,
                len(eval_set),
                question[:60],
            )
            print(
                f"\n[{q_idx + 1}/{len(eval_set)}] {question[:80]}"
            )

            try:
                run_results = compare_methods(
                    question,
                    methods,
                    config,
                    retriever,
                    embeddings_model,
                    generator,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "compare_methods failed for question %d (%r): %s",
                    q_idx + 1,
                    question[:60],
                    exc,
                )
                continue

            for run_result in run_results:
                try:
                    metric_dict = evaluate_run(run_result, gold_pmids, reference_answer)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "evaluate_run failed for method %r, question %d: %s",
                        run_result.method_name,
                        q_idx + 1,
                        exc,
                    )
                    continue

                record = _build_record(run_result, metric_dict)
                out_fh.write(json.dumps(record) + "\n")
                out_fh.flush()

                all_metric_dicts.append(metric_dict)
                csv_rows.append(
                    {
                        "question_idx": q_idx + 1,
                        "question": question[:120],
                        "method": run_result.method_name,
                        **{k: metric_dict.get(k) for k in _NUMERIC_METRIC_KEYS},
                    }
                )

    print(f"\nResults written to: {output_path}")

    aggregate = _aggregate_results(all_metric_dicts)
    _print_aggregate_table(aggregate)

    # --- Summary and metrics CSV ---
    write_summary(run_path, aggregate)
    write_metrics_csv(run_path, csv_rows)

    # --- Tear down file logging ---
    logging.getLogger().removeHandler(_file_handler)
    _file_handler.close()

    print(f"Run directory: {run_path}")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluation",
        description="LitePack-RAG Phase 8 evaluation runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--eval-set",
        required=True,
        dest="eval_set",
        metavar="PATH",
        help="Path to eval_data.json.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["similarity", "mmr", "litepack"],
        choices=["similarity", "mmr", "litepack"],
        metavar="METHOD",
        help="Methods to compare.",
    )
    p.add_argument("--budget", type=int, default=1800, help="Token budget.")
    p.add_argument(
        "--top-n", type=int, default=20, dest="top_n",
        help="Candidates to retrieve from Pinecone.",
    )
    p.add_argument(
        "--use-keywords", action="store_true", dest="use_keywords",
        help="Enable keyword-overlap feature.",
    )
    p.add_argument(
        "--use-recency", action="store_true", dest="use_recency",
        help="Enable recency feature.",
    )
    p.add_argument(
        "--use-publication-type", action="store_true",
        dest="use_publication_type",
        help="Enable publication-type feature (stub).",
    )
    p.add_argument(
        "--use-mesh", action="store_true", dest="use_mesh",
        help="Enable MeSH feature (stub).",
    )
    p.add_argument(
        "--reference-year", type=int, default=None, dest="reference_year",
        help="Reference year for recency scoring.",
    )
    p.add_argument(
        "--output-dir",
        default="",
        dest="output_dir",
        metavar="DIR",
        help=(
            "Directory to write run outputs (config.json, results.jsonl, "
            "summary.json, metrics.csv, logs.txt, eval_set_snapshot.json). "
            "If empty (the default), a timestamped directory is created "
            "automatically under <project_root>/results/runs/."
        ),
    )
    p.add_argument(
        "--run-tag",
        default="eval_001",
        dest="run_tag",
        metavar="TAG",
        help="Short label appended to the auto-generated run directory name.",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Logging verbosity.",
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg_kwargs: dict = {
        "budget": args.budget,
        "top_n": args.top_n,
        "use_keywords": args.use_keywords,
        "use_recency": args.use_recency,
        "use_publication_type": args.use_publication_type,
        "use_mesh": args.use_mesh,
    }
    if args.reference_year is not None:
        cfg_kwargs["reference_year"] = args.reference_year

    config = PipelineConfig(**cfg_kwargs)

    eval_set = load_eval_set(args.eval_set)

    retriever = Retriever(config)
    generator = Generator(config)

    output_dir = args.output_dir if args.output_dir else str(make_run_dir(tag=args.run_tag))

    run_evaluation(
        eval_set=eval_set,
        methods=args.methods,
        config=config,
        retriever=retriever,
        embeddings_model=retriever.embeddings_model,
        generator=generator,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
