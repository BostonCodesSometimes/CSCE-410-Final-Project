"""Orchestration and CLI entrypoint for LitePack-RAG.

Responsibilities:
  - tie together retrieval, enrichment, ranking, selection, packing,
    and generation into run_once()
  - support one-query runs and side-by-side method comparisons via
    compare_methods()
  - provide a complete argparse CLI

Evaluation mode (metrics.py / evaluation.py) is NOT implemented here.
Do not add evaluation logic to this file.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from dataclasses import dataclass

import numpy as np

from config import PipelineConfig
from retriever import Retriever, Candidate
from features import enrich_candidates
from rankers import BaseRanker, SimilarityRanker, MMRRanker, LitePackRanker
from selector import SelectionResult, SelectionTraceStep, select_baseline, select_greedy
from packer import PackedContext, build_context
from generator import Generator, GenerationResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Method -> (ranker, selector) mapping                                         #
# --------------------------------------------------------------------------- #

METHOD_TO_RANKER_SELECTOR: dict[str, tuple[str, str]] = {
    "similarity": ("similarity", "baseline"),
    "mmr":        ("mmr",        "greedy"),
    "litepack":   ("litepack",   "greedy"),
}

_VALID_COMBOS: frozenset[tuple[str, str]] = frozenset({
    ("similarity", "baseline"),
    ("mmr",        "greedy"),
    ("litepack",   "greedy"),
})


# --------------------------------------------------------------------------- #
# RunResult dataclass                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class RunResult:
    """Complete record of one pipeline execution.

    Timings dict uses these fixed keys:
        retrieval_ms, enrichment_ms, ranking_ms, selection_ms,
        packing_ms, generation_ms, total_ms

    Metrics dict always contains at minimum:
        empty_retrieval (bool), empty_selection (bool),
        tokens_used (int), chunks_selected (int), chunks_retrieved (int)
    """

    method_name: str
    query: str
    config: PipelineConfig
    retrieved_candidates: list[Candidate]
    ranked_candidates: list[Candidate]
    selection_result: SelectionResult
    packed_context: PackedContext
    generation_result: GenerationResult
    metrics: dict
    timings: dict


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _make_ranker(ranker_name: str) -> BaseRanker:
    """Instantiate a ranker by name."""
    if ranker_name == "similarity":
        return SimilarityRanker()
    if ranker_name == "mmr":
        return MMRRanker()
    if ranker_name == "litepack":
        return LitePackRanker()
    raise ValueError(f"Unknown ranker: {ranker_name!r}")


def _empty_selection_result(budget: int, config: PipelineConfig) -> SelectionResult:
    return SelectionResult(
        selected_candidates=[],
        tokens_used=0,
        token_budget=budget,
        dropped_candidates=[],
        ordering_policy=config.ordering_policy,
        trace=[],
        selection_labels={},
    )


def _empty_packed_context() -> PackedContext:
    return PackedContext(
        context_text="",
        source_blocks=[],
        tokens_used=0,
        selected_pmids=[],
        citation_map={},
    )


def _stub_generation_result(answer_text: str, model_name: str) -> GenerationResult:
    return GenerationResult(
        answer_text=answer_text,
        cited_source_labels=[],
        raw_response="",
        model_name=model_name,
        prompt_tokens=0,
        completion_tokens=0,
    )


def _zero_timings() -> dict[str, float]:
    return {
        "retrieval_ms": 0.0,
        "enrichment_ms": 0.0,
        "ranking_ms": 0.0,
        "selection_ms": 0.0,
        "packing_ms": 0.0,
        "generation_ms": 0.0,
        "total_ms": 0.0,
    }


# --------------------------------------------------------------------------- #
# run_once                                                                     #
# --------------------------------------------------------------------------- #

def run_once(
    query: str,
    config: PipelineConfig,
    retriever: Retriever,
    embeddings_model,
    generator: Generator,
) -> RunResult:
    """Execute the full pipeline for a single query.

    Steps
    -----
    1.  Retrieve candidates and query embedding.
    2.  Empty-retrieval short-circuit (no LLM call if nothing retrieved).
    3.  Enrich candidates.
    4.  Instantiate the ranker and statically rank.
    5.  Run the selector under the token budget.
    6.  Empty-selection short-circuit (no LLM call if nothing fits).
    7.  Build packed context.
    8.  Generate the answer.
    9.  Compute inline metrics.
    10. Return RunResult.

    Args:
        query:            user query string.
        config:           PipelineConfig controlling all behavior.
        retriever:        initialized Retriever instance.
        embeddings_model: LangChain embeddings object (retriever.embeddings_model).
        generator:        initialized Generator instance.

    Returns:
        RunResult with all fields populated.
    """
    method_name = f"{config.ranker}+{config.selector}"
    t_start = time.monotonic()
    timings = _zero_timings()

    # ---------------------------------------------------------------------- #
    # Step 1: Retrieve                                                         #
    # ---------------------------------------------------------------------- #
    t0 = time.monotonic()
    candidates, query_embedding = retriever.retrieve(query, top_n=config.top_n)
    timings["retrieval_ms"] = (time.monotonic() - t0) * 1000.0

    # ---------------------------------------------------------------------- #
    # Step 2: Empty-retrieval short-circuit                                   #
    # ---------------------------------------------------------------------- #
    if len(candidates) == 0:
        logger.warning("run_once: no candidates retrieved for query %r", query)
        timings["total_ms"] = (time.monotonic() - t_start) * 1000.0
        return RunResult(
            method_name=method_name,
            query=query,
            config=config,
            retrieved_candidates=[],
            ranked_candidates=[],
            selection_result=_empty_selection_result(config.budget, config),
            packed_context=_empty_packed_context(),
            generation_result=_stub_generation_result(
                "No sources were retrieved for this query; unable to answer.",
                config.llm_model,
            ),
            metrics={"empty_retrieval": True, "empty_selection": False,
                     "tokens_used": 0, "chunks_selected": 0, "chunks_retrieved": 0},
            timings=timings,
        )

    # ---------------------------------------------------------------------- #
    # Step 3: Enrich                                                           #
    # ---------------------------------------------------------------------- #
    t0 = time.monotonic()
    candidates = enrich_candidates(
        candidates, query, query_embedding, embeddings_model, config
    )
    timings["enrichment_ms"] = (time.monotonic() - t0) * 1000.0

    # ---------------------------------------------------------------------- #
    # Step 4: Rank                                                             #
    # ---------------------------------------------------------------------- #
    t0 = time.monotonic()
    ranker = _make_ranker(config.ranker)
    ranked = ranker.rank(candidates, query_embedding, config)
    timings["ranking_ms"] = (time.monotonic() - t0) * 1000.0

    # ---------------------------------------------------------------------- #
    # Step 5: Select                                                           #
    # ---------------------------------------------------------------------- #
    t0 = time.monotonic()
    if config.selector == "baseline":
        selection_result = select_baseline(ranked, config.budget, config)
    elif config.selector == "greedy":
        selection_result = select_greedy(
            ranked, ranker, query_embedding, config.budget, config
        )
    else:
        raise ValueError(f"Unknown selector: {config.selector!r}")
    timings["selection_ms"] = (time.monotonic() - t0) * 1000.0

    # ---------------------------------------------------------------------- #
    # Step 6: Empty-selection short-circuit                                   #
    # ---------------------------------------------------------------------- #
    if len(selection_result.selected_candidates) == 0:
        logger.warning(
            "run_once: no candidates fit within budget=%d for query %r",
            config.budget,
            query,
        )
        timings["total_ms"] = (time.monotonic() - t_start) * 1000.0
        return RunResult(
            method_name=method_name,
            query=query,
            config=config,
            retrieved_candidates=candidates,
            ranked_candidates=ranked,
            selection_result=selection_result,
            packed_context=_empty_packed_context(),
            generation_result=_stub_generation_result(
                "No candidates fit within the token budget.",
                config.llm_model,
            ),
            metrics={"empty_retrieval": False, "empty_selection": True,
                     "tokens_used": 0, "chunks_selected": 0,
                     "chunks_retrieved": len(candidates)},
            timings=timings,
        )

    # ---------------------------------------------------------------------- #
    # Step 7: Pack context                                                     #
    # ---------------------------------------------------------------------- #
    t0 = time.monotonic()
    packed = build_context(selection_result, config)
    timings["packing_ms"] = (time.monotonic() - t0) * 1000.0

    # ---------------------------------------------------------------------- #
    # Step 8: Generate answer                                                  #
    # ---------------------------------------------------------------------- #
    t0 = time.monotonic()
    generation_result = generator.generate(query, packed, config)
    timings["generation_ms"] = (time.monotonic() - t0) * 1000.0

    timings["total_ms"] = (time.monotonic() - t_start) * 1000.0

    # ---------------------------------------------------------------------- #
    # Step 9: Inline metrics                                                   #
    # ---------------------------------------------------------------------- #
    metrics: dict = {
        "empty_retrieval": False,
        "empty_selection": False,
        "tokens_used": packed.tokens_used,
        "chunks_selected": len(selection_result.selected_candidates),
        "chunks_retrieved": len(candidates),
    }

    return RunResult(
        method_name=method_name,
        query=query,
        config=config,
        retrieved_candidates=candidates,
        ranked_candidates=ranked,
        selection_result=selection_result,
        packed_context=packed,
        generation_result=generation_result,
        metrics=metrics,
        timings=timings,
    )


# --------------------------------------------------------------------------- #
# compare_methods                                                              #
# --------------------------------------------------------------------------- #

def compare_methods(
    query: str,
    methods: list[str],
    config: PipelineConfig,
    retriever: Retriever,
    embeddings_model,
    generator: Generator,
) -> list[RunResult]:
    """Run multiple methods on the same query and print a comparison table.

    The shared *config* is never mutated.  A fresh copy is created per method
    using dataclasses.replace() with the method-specific ranker and selector.

    Args:
        query:            user query string.
        methods:          list of method names from {"similarity","mmr","litepack"}.
        config:           base PipelineConfig (not mutated).
        retriever:        initialized Retriever.
        embeddings_model: LangChain embeddings object.
        generator:        initialized Generator.

    Returns:
        List of RunResult, one per method, in the order *methods* was given.
        Unknown method names are skipped with a warning.
    """
    results: list[RunResult] = []

    for method in methods:
        if method not in METHOD_TO_RANKER_SELECTOR:
            logger.warning("compare_methods: unknown method %r; skipping", method)
            continue
        ranker_name, selector_name = METHOD_TO_RANKER_SELECTOR[method]
        method_config = dataclasses.replace(
            config, ranker=ranker_name, selector=selector_name
        )
        result = run_once(
            query, method_config, retriever, embeddings_model, generator
        )
        result.method_name = method  # use the friendly method name
        results.append(result)

    # Print a concise comparison table
    print("\n" + "=" * 76)
    print(f"{'METHOD':<14} {'TOKENS':>7} {'CHUNKS':>7}  ANSWER PREVIEW")
    print("-" * 76)
    for r in results:
        tokens = r.packed_context.tokens_used
        chunks = r.metrics.get("chunks_selected", 0)
        preview = r.generation_result.answer_text[:55].replace("\n", " ")
        print(f"{r.method_name:<14} {tokens:>7} {chunks:>7}  {preview!r}")
    print("=" * 76 + "\n")

    return results


# --------------------------------------------------------------------------- #
# Combination validator                                                        #
# --------------------------------------------------------------------------- #

def _validate_combo(ranker: str, selector: str) -> tuple[str, str]:
    """Validate or normalize a ranker/selector combination.

    If the combination is supported, returns it unchanged.
    If the ranker is known but paired with an unsupported selector, the
    recommended selector is substituted and a warning is logged.
    If the ranker itself is unknown, raises ValueError.

    Supported combinations:
        similarity + baseline
        mmr        + greedy
        litepack   + greedy

    Args:
        ranker:   ranker identifier string.
        selector: selector identifier string.

    Returns:
        (ranker, selector) tuple, possibly normalized.

    Raises:
        ValueError: if the ranker is not recognized.
    """
    if (ranker, selector) in _VALID_COMBOS:
        return ranker, selector

    recommended: dict[str, tuple[str, str]] = {
        "similarity": ("similarity", "baseline"),
        "mmr":        ("mmr",        "greedy"),
        "litepack":   ("litepack",   "greedy"),
    }
    if ranker in recommended:
        norm_ranker, norm_selector = recommended[ranker]
        logger.warning(
            "Unsupported combination (%r + %r); normalizing to (%r + %r).",
            ranker, selector, norm_ranker, norm_selector,
        )
        return norm_ranker, norm_selector

    raise ValueError(
        f"Unsupported ranker/selector combination: {ranker!r} + {selector!r}. "
        f"Valid combos: {sorted(_VALID_COMBOS)}"
    )


# --------------------------------------------------------------------------- #
# JSON serialization (no embedding arrays)                                     #
# --------------------------------------------------------------------------- #

class _NumpyEncoder(json.JSONEncoder):
    """Strip numpy arrays; coerce numpy scalar types to Python primitives."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return None  # embeddings are stripped; do not store vectors
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def _candidate_to_dict(c: Candidate) -> dict:
    """Serialize a Candidate without the embedding array."""
    return {
        "chunk_id": c.chunk_id,
        "pmid": c.pmid,
        "title": c.title,
        "year": c.year,
        "journal": c.journal,
        "url": c.url,
        "retrieval_score": float(c.retrieval_score),
        "final_score": float(c.final_score) if c.final_score is not None else None,
        "token_length": c.token_length,
        "keyword_overlap": c.keyword_overlap,
        "recency_score": c.recency_score,
        "score_breakdown": c.score_breakdown,
    }


def _run_result_to_dict(result: RunResult) -> dict:
    """Serialize a RunResult to a JSON-safe dict.

    Embedding arrays are stripped from all Candidate objects.
    """
    sr = result.selection_result
    pc = result.packed_context
    gen = result.generation_result

    return {
        "method_name": result.method_name,
        "query": result.query,
        "config": dataclasses.asdict(result.config),
        "retrieved_candidates": [
            _candidate_to_dict(c) for c in result.retrieved_candidates
        ],
        "ranked_candidates": [
            _candidate_to_dict(c) for c in result.ranked_candidates
        ],
        "selection_result": {
            "selected_candidates": [
                _candidate_to_dict(c) for c in sr.selected_candidates
            ],
            "tokens_used": sr.tokens_used,
            "token_budget": sr.token_budget,
            "dropped_candidates": [c.chunk_id for c in sr.dropped_candidates],
            "ordering_policy": sr.ordering_policy,
            "selection_labels": sr.selection_labels,
            "trace": [dataclasses.asdict(step) for step in sr.trace],
        },
        "packed_context": {
            "context_text": pc.context_text,
            "source_blocks": pc.source_blocks,
            "tokens_used": pc.tokens_used,
            "selected_pmids": pc.selected_pmids,
            "citation_map": {
                label: _candidate_to_dict(cand)
                for label, cand in pc.citation_map.items()
            },
        },
        "generation_result": dataclasses.asdict(gen),
        "metrics": result.metrics,
        "timings": result.timings,
    }


# --------------------------------------------------------------------------- #
# Display helpers                                                              #
# --------------------------------------------------------------------------- #

def _print_run_result(
    result: RunResult,
    show_trace: bool,
    show_scores: bool,
) -> None:
    """Pretty-print a RunResult to stdout."""
    sr = result.selection_result
    gen = result.generation_result
    pc = result.packed_context
    t = result.timings

    print(f"\n{'=' * 72}")
    print(f"Method  : {result.method_name}")
    print(f"Query   : {result.query}")
    print(
        f"Tokens  : {pc.tokens_used} / {result.config.budget}  "
        f"({result.metrics.get('chunks_selected', 0)} chunks selected, "
        f"{result.metrics.get('chunks_retrieved', 0)} retrieved)"
    )
    print(
        f"Timings : retrieval={t['retrieval_ms']:.0f}ms  "
        f"enrich={t['enrichment_ms']:.0f}ms  "
        f"rank={t['ranking_ms']:.0f}ms  "
        f"select={t['selection_ms']:.0f}ms  "
        f"pack={t['packing_ms']:.0f}ms  "
        f"gen={t['generation_ms']:.0f}ms  "
        f"total={t['total_ms']:.0f}ms"
    )

    if show_scores and sr.selected_candidates:
        print("\n--- Selected candidates ---")
        for c in sr.selected_candidates:
            label = sr.selection_labels.get(c.chunk_id, "?")
            fs = f"{c.final_score:.4f}" if c.final_score is not None else "n/a"
            print(
                f"  {label}  PMID={c.pmid}  "
                f"final_score={fs}  retrieval={c.retrieval_score:.4f}  "
                f"tokens={c.token_length}"
            )

    if show_trace and sr.trace:
        print("\n--- Selection trace ---")
        for step in sr.trace:
            print(
                f"  step={step.step_index}  {step.label}  "
                f"id={step.candidate_id}  score={step.marginal_score:.4f}  "
                f"cost={step.effective_token_cost}t  "
                f"cumulative={step.tokens_used_so_far}t"
            )

    print(f"\n--- Answer ---")
    print(gen.answer_text)
    if gen.cited_source_labels:
        print(f"\nCited   : {', '.join(gen.cited_source_labels)}")
    print(
        f"API     : prompt_tokens={gen.prompt_tokens}  "
        f"completion_tokens={gen.completion_tokens}  model={gen.model_name}"
    )
    print("=" * 72)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="LitePack-RAG: medical literature retrieval and answer generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--query", required=True, help="Query string.")
    p.add_argument(
        "--ranker", default="litepack",
        choices=["similarity", "mmr", "litepack"],
        help="Ranking method.",
    )
    p.add_argument(
        "--selector", default="greedy",
        choices=["baseline", "greedy"],
        help="Selection method.",
    )
    p.add_argument("--budget", type=int, default=1800, help="Token budget.")
    p.add_argument("--top-n", type=int, default=20, dest="top_n",
                   help="Candidates to retrieve from Pinecone.")
    p.add_argument("--use-keywords", action="store_true", dest="use_keywords",
                   help="Enable keyword-overlap feature.")
    p.add_argument("--use-recency", action="store_true", dest="use_recency",
                   help="Enable recency feature.")
    p.add_argument("--use-publication-type", action="store_true",
                   dest="use_publication_type",
                   help="Enable publication-type feature (stub).")
    p.add_argument("--use-mesh", action="store_true", dest="use_mesh",
                   help="Enable MeSH feature (stub).")
    p.add_argument("--reference-year", type=int, default=None,
                   dest="reference_year",
                   help="Reference year for recency scoring. Defaults to current year.")
    p.add_argument("--show-trace", action="store_true", dest="show_trace",
                   help="Print the greedy selection trace.")
    p.add_argument("--show-scores", action="store_true", dest="show_scores",
                   help="Print per-candidate scores after selection.")
    p.add_argument("--save-run", type=str, default=None, dest="save_run",
                   metavar="PATH",
                   help="Save run result(s) to this JSON file.")
    p.add_argument(
        "--compare", nargs="+",
        choices=["similarity", "mmr", "litepack"],
        metavar="METHOD",
        help="Compare multiple methods side by side (ignores --ranker/--selector).",
    )
    p.add_argument(
        "--log-level", default="WARNING",
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

    # Build base PipelineConfig from CLI args.
    cfg_kwargs: dict = {
        "budget": args.budget,
        "top_n": args.top_n,
        "use_keywords": args.use_keywords,
        "use_recency": args.use_recency,
        "use_publication_type": args.use_publication_type,
        "use_mesh": args.use_mesh,
        "ranker": args.ranker,
        "selector": args.selector,
    }
    if args.reference_year is not None:
        cfg_kwargs["reference_year"] = args.reference_year

    config = PipelineConfig(**cfg_kwargs)

    # Initialize shared components (retriever owns the embeddings model).
    retriever = Retriever(config)
    generator = Generator(config)

    if args.compare:
        # Compare mode: ranker/selector per method, ignore manual flags.
        results = compare_methods(
            args.query,
            args.compare,
            config,
            retriever,
            retriever.embeddings_model,
            generator,
        )
        for r in results:
            _print_run_result(r, args.show_trace, args.show_scores)

        if args.save_run:
            payload = [_run_result_to_dict(r) for r in results]
            with open(args.save_run, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, cls=_NumpyEncoder)
            print(f"Saved comparison run to {args.save_run!r}")

    else:
        # Single-run mode: validate / normalize ranker+selector combo.
        ranker, selector = _validate_combo(args.ranker, args.selector)
        config = dataclasses.replace(config, ranker=ranker, selector=selector)

        result = run_once(
            args.query,
            config,
            retriever,
            retriever.embeddings_model,
            generator,
        )
        _print_run_result(result, args.show_trace, args.show_scores)

        if args.save_run:
            payload = _run_result_to_dict(result)
            with open(args.save_run, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, cls=_NumpyEncoder)
            print(f"Saved run to {args.save_run!r}")


if __name__ == "__main__":
    main()
