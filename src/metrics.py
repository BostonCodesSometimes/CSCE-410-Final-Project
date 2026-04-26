"""Metrics module for LitePack-RAG Phase 8 evaluation.

Responsibilities:
  - retrieval-side metrics  (recall_at_k, ndcg_at_k, evidence_hit_rate)
  - selection-side metrics  (redundancy_score, avg_pairwise_similarity)
  - efficiency metrics      (prompt_token_count, budget_utilization)
  - answer-side metrics     (rouge_l, keyword_support_rate)
  - evaluate_run()          aggregates all metrics for one RunResult

All functions return safe defaults (0.0 / 0) on empty inputs.
None of these functions mutate their inputs.
"""
from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING

import numpy as np
from rouge_score import rouge_scorer

from retriever import Candidate
from packer import PackedContext

if TYPE_CHECKING:
    from pipeline import RunResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _safe_embeddings(candidates: list[Candidate]) -> list[np.ndarray]:
    """Return a list of embedding arrays, skipping candidates with None."""
    out = []
    for c in candidates:
        if c.embedding is not None:
            arr = np.asarray(c.embedding, dtype=np.float32)
            if arr.ndim == 1 and arr.size > 0:
                out.append(arr)
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors; returns 0.0 if either is zero."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mean_pairwise_cosine(embeddings: list[np.ndarray]) -> float:
    """Mean of all upper-triangle pairwise cosine similarities."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _cosine(embeddings[i], embeddings[j])
            count += 1
    return total / count if count > 0 else 0.0


# --------------------------------------------------------------------------- #
# Retrieval-side metrics                                                       #
# --------------------------------------------------------------------------- #

def recall_at_k(candidates: list[Candidate], gold_pmids: list[str], k: int) -> float:
    """Fraction of gold PMIDs present in the top-k retrieved candidates.

    Args:
        candidates: ordered list of retrieved Candidate objects (rank order).
        gold_pmids: list of ground-truth PMIDs.
        k:          cutoff rank.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 on empty inputs or k <= 0.
    """
    if not candidates or not gold_pmids or k <= 0:
        return 0.0
    top_k_pmids = {c.pmid for c in candidates[:k]}
    gold_set = set(gold_pmids)
    return len(top_k_pmids & gold_set) / len(gold_set)


def ndcg_at_k(candidates: list[Candidate], gold_pmids: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k using binary relevance.

    Relevance is 1 if candidate.pmid is in gold_pmids, else 0.
    Each gold PMID is counted at most once (first occurrence in the ranked list).
    This is the correct treatment for chunk-level retrieval with document-level
    gold labels: multiple chunks from the same paper do not inflate DCG beyond 1.0.

    Args:
        candidates: ordered list of retrieved Candidate objects.
        gold_pmids: ground-truth PMID list.
        k:          cutoff rank.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 on empty inputs or k <= 0.
    """
    if not candidates or not gold_pmids or k <= 0:
        return 0.0

    gold_set = set(gold_pmids)
    top_k = candidates[:k]

    # DCG: each gold PMID credited only on its first appearance in the ranking.
    seen_gold: set[str] = set()
    dcg = 0.0
    for i, c in enumerate(top_k):
        if c.pmid in gold_set and c.pmid not in seen_gold:
            dcg += 1.0 / math.log2(i + 2)
            seen_gold.add(c.pmid)

    # IDCG: ideal — one chunk per gold PMID at the highest ranks.
    n_relevant = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evidence_hit_rate(selected: list[Candidate], gold_pmids: list[str]) -> float:
    """Fraction of gold PMIDs present in the final selected candidate set.

    Args:
        selected:   list of Candidate objects chosen by the selector.
        gold_pmids: ground-truth PMID list.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 on empty inputs.
    """
    if not selected or not gold_pmids:
        return 0.0
    selected_pmids = {c.pmid for c in selected}
    gold_set = set(gold_pmids)
    return len(selected_pmids & gold_set) / len(gold_set)


# --------------------------------------------------------------------------- #
# Selection-side metrics                                                       #
# --------------------------------------------------------------------------- #

def redundancy_score(selected: list[Candidate]) -> float:
    """Mean pairwise cosine similarity among selected candidate embeddings.

    Lower is better (less redundancy).  Candidates with missing or empty
    embeddings are silently skipped.

    Args:
        selected: list of selected Candidate objects.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 if fewer than 2 candidates have
        valid embeddings.
    """
    embeddings = _safe_embeddings(selected)
    return _mean_pairwise_cosine(embeddings)


def avg_pairwise_similarity(selected: list[Candidate]) -> float:
    """Mean pairwise cosine similarity among selected candidate embeddings.

    Identical computation to redundancy_score; exposed as a separate named
    function for reporting clarity.

    Args:
        selected: list of selected Candidate objects.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 if fewer than 2 candidates have
        valid embeddings.
    """
    embeddings = _safe_embeddings(selected)
    return _mean_pairwise_cosine(embeddings)


# --------------------------------------------------------------------------- #
# Efficiency metrics                                                           #
# --------------------------------------------------------------------------- #

def prompt_token_count(packed: PackedContext) -> int:
    """Return the token count of the assembled prompt context.

    Args:
        packed: PackedContext output from packer.build_context().

    Returns:
        Integer token count (packed.tokens_used).
    """
    return packed.tokens_used


def budget_utilization(packed: PackedContext, budget: int) -> float:
    """Fraction of the token budget consumed by the packed context.

    Args:
        packed: PackedContext output from packer.build_context().
        budget: integer token budget from PipelineConfig.budget.

    Returns:
        Float in [0.0, ...].  Returns 0.0 if budget <= 0.
    """
    if budget <= 0:
        return 0.0
    return packed.tokens_used / budget


# --------------------------------------------------------------------------- #
# Answer-side metrics                                                          #
# --------------------------------------------------------------------------- #

def rouge_l(answer: str, reference: str) -> float:
    """ROUGE-L F-measure between answer and reference strings.

    Uses rouge_score.rouge_scorer with "rougeL" and tokenize=True (default).
    Returns the F-measure (fmeasure) scalar.

    Args:
        answer:    generated answer string.
        reference: gold reference answer string.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 if either string is empty.
    """
    if not answer or not reference:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, answer)
    return float(scores["rougeL"].fmeasure)


def keyword_support_rate(answer: str, context_text: str) -> float:
    """Heuristic lexical support rate: fraction of answer tokens found in context.

    This is a simple lexical overlap metric.  It is NOT a faithfulness metric
    and does NOT validate semantic entailment.  It measures only whether the
    words used in the answer appear somewhere in the packed context.

    Tokenization: lowercase, regex word-boundary split (\\b\\w+\\b).
    Stop words are NOT removed.  Short tokens (< 3 characters) are excluded
    to reduce noise from articles and prepositions.

    Args:
        answer:       generated answer string.
        context_text: full packed context string passed to the LLM.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 if answer or context_text is empty.

    Note:
        This is a heuristic lexical support metric, not an NLI-based
        faithfulness metric.  Do not report it as a rigorous faithfulness score.
    """
    if not answer or not context_text:
        return 0.0

    def _tokens(text: str) -> list[str]:
        return [t for t in re.findall(r"\b\w+\b", text.lower()) if len(t) >= 3]

    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0

    context_vocab = set(_tokens(context_text))
    supported = sum(1 for t in answer_tokens if t in context_vocab)
    return supported / len(answer_tokens)


# --------------------------------------------------------------------------- #
# evaluate_run: aggregate all metrics for one RunResult                       #
# --------------------------------------------------------------------------- #

def evaluate_run(
    run_result: "RunResult",
    gold_pmids: list[str],
    reference_answer: str,
) -> dict:
    """Compute all Phase 8 metrics for a single RunResult.

    The returned dict is flat, JSON-serializable, and uses consistent key names.
    run_result is never mutated.

    Args:
        run_result:       RunResult from pipeline.run_once() or compare_methods().
        gold_pmids:       list of ground-truth PMIDs for the query.
        reference_answer: gold reference answer string (may be empty).

    Returns:
        Flat dict with the following keys:

        method_name, query,
        empty_retrieval, empty_selection,
        recall_at_5, ndcg_at_5,
        evidence_hit_rate, redundancy_score, avg_pairwise_similarity,
        prompt_token_count, budget_utilization,
        rouge_l, keyword_support_rate,
        chunks_selected, chunks_retrieved
    """
    base = {
        "method_name": run_result.method_name,
        "query": run_result.query,
        "empty_retrieval": bool(run_result.metrics.get("empty_retrieval", False)),
        "empty_selection": bool(run_result.metrics.get("empty_selection", False)),
    }

    # Safe-default record for empty-retrieval short-circuit
    if run_result.metrics.get("empty_retrieval"):
        return {
            **base,
            "recall_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "evidence_hit_rate": 0.0,
            "redundancy_score": 0.0,
            "avg_pairwise_similarity": 0.0,
            "prompt_token_count": 0,
            "budget_utilization": 0.0,
            "rouge_l": 0.0,
            "keyword_support_rate": 0.0,
            "chunks_selected": 0,
            "chunks_retrieved": 0,
        }

    # Safe-default record for empty-selection short-circuit
    if run_result.metrics.get("empty_selection"):
        retrieved = run_result.retrieved_candidates
        k = min(5, len(retrieved)) if retrieved else 0
        return {
            **base,
            "recall_at_5": recall_at_k(retrieved, gold_pmids, k),
            "ndcg_at_5": ndcg_at_k(retrieved, gold_pmids, k),
            "evidence_hit_rate": 0.0,
            "redundancy_score": 0.0,
            "avg_pairwise_similarity": 0.0,
            "prompt_token_count": 0,
            "budget_utilization": 0.0,
            "rouge_l": 0.0,
            "keyword_support_rate": 0.0,
            "chunks_selected": 0,
            "chunks_retrieved": len(retrieved),
        }

    # Normal path
    retrieved = run_result.retrieved_candidates
    selected = run_result.selection_result.selected_candidates
    packed = run_result.packed_context
    answer = run_result.generation_result.answer_text
    budget = run_result.config.budget

    k = min(5, len(retrieved)) if retrieved else 0

    return {
        **base,
        "recall_at_5": recall_at_k(retrieved, gold_pmids, k),
        "ndcg_at_5": ndcg_at_k(retrieved, gold_pmids, k),
        "evidence_hit_rate": evidence_hit_rate(selected, gold_pmids),
        "redundancy_score": redundancy_score(selected),
        "avg_pairwise_similarity": avg_pairwise_similarity(selected),
        "prompt_token_count": prompt_token_count(packed),
        "budget_utilization": budget_utilization(packed, budget),
        "rouge_l": rouge_l(answer, reference_answer),
        "keyword_support_rate": keyword_support_rate(answer, packed.context_text),
        "chunks_selected": len(selected),
        "chunks_retrieved": len(retrieved),
    }
