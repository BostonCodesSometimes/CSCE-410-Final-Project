"""Budget-aware selection module for LitePack-RAG.

Responsibilities:
  - budget-aware subset selection
  - the greedy loop lives here and only here
  - selected candidates receive final_score and score_breakdown here

This module does NOT perform context formatting, ordering, or packing.
Those responsibilities belong to packer.py (Phase 5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import PipelineConfig
from retriever import Candidate
from rankers import BaseRanker
from packer import assign_label, format_chunk_header, count_tokens, effective_token_cost

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataclasses                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class SelectionTraceStep:
    """One step of the greedy selection trace.

    Records exactly what was chosen at each iteration of select_greedy(),
    why it was chosen (marginal_score and score_breakdown), and the running
    token count after inclusion.
    """
    step_index: int
    candidate_id: str
    label: str
    token_length: int
    effective_token_cost: int
    marginal_score: float
    score_breakdown: dict
    tokens_used_so_far: int


@dataclass
class SelectionResult:
    """Output produced by select_baseline() or select_greedy().

    Contains everything the downstream pipeline needs: the chosen candidates,
    their budget accounting, per-candidate labels, and the full greedy trace
    (populated only by select_greedy).
    """
    selected_candidates: list[Candidate]
    tokens_used: int
    token_budget: int
    dropped_candidates: list[Candidate]
    ordering_policy: str
    trace: list[SelectionTraceStep]
    selection_labels: dict  # chunk_id -> "S{n}"


# --------------------------------------------------------------------------- #
# Selection functions                                                          #
# --------------------------------------------------------------------------- #

def select_baseline(
    candidates: list[Candidate],
    budget: int,
    config: PipelineConfig,
) -> SelectionResult:
    """Include candidates in order until the token budget is exhausted.

    Assumes *candidates* are already in the desired order (e.g. sorted by
    retrieval_score descending by SimilarityRanker.rank()).

    No re-scoring, no greedy loop.  final_score and score_breakdown are NOT
    set on any candidate by this function.

    Args:
        candidates: pre-sorted list of Candidate objects.
        budget:     token budget (inclusive upper bound).
        config:     PipelineConfig with llm_model and ordering_policy.

    Returns:
        SelectionResult with trace=[] (trace is not populated for baseline).
    """
    if not candidates:
        return SelectionResult(
            selected_candidates=[],
            tokens_used=0,
            token_budget=budget,
            dropped_candidates=[],
            ordering_policy=config.ordering_policy,
            trace=[],
            selection_labels={},
        )

    selected: list[Candidate] = []
    dropped: list[Candidate] = []
    selection_labels: dict[str, str] = {}
    tokens_used = 0

    for candidate in candidates:
        label = assign_label(len(selected))
        cost = effective_token_cost(candidate, label, config.llm_model)

        if tokens_used + cost <= budget:
            selected.append(candidate)
            selection_labels[candidate.chunk_id] = label
            tokens_used += cost
        else:
            dropped.append(candidate)

    logger.info(
        "select_baseline: selected %d / %d candidates (%d tokens, budget %d)",
        len(selected),
        len(candidates),
        tokens_used,
        budget,
    )

    return SelectionResult(
        selected_candidates=selected,
        tokens_used=tokens_used,
        token_budget=budget,
        dropped_candidates=dropped,
        ordering_policy=config.ordering_policy,
        trace=[],
        selection_labels=selection_labels,
    )


def select_greedy(
    candidates: list[Candidate],
    ranker: BaseRanker,
    query_embedding,
    budget: int,
    config: PipelineConfig,
) -> SelectionResult:
    """Greedily select candidates by marginal score under the token budget.

    At each step the function evaluates every remaining candidate with
    ranker.marginal_score(), picks the highest-scoring one that fits within
    the remaining budget, and repeats until no candidate fits.

    This is the only place in the codebase where the greedy selection loop
    runs.  Rankers provide marginal_score(); they do not run any loop.

    On inclusion, chosen.final_score and chosen.score_breakdown are populated.

    Args:
        candidates:      pre-ranked list of Candidate objects (all enriched).
        ranker:          an instance of BaseRanker (MMRRanker, LitePackRanker,
                         or SimilarityRanker).
        query_embedding: np.ndarray of shape (dims,).
        budget:          token budget (inclusive upper bound).
        config:          PipelineConfig with llm_model and ordering_policy.

    Returns:
        SelectionResult with trace populated (one step per selected candidate).
    """
    if not candidates:
        return SelectionResult(
            selected_candidates=[],
            tokens_used=0,
            token_budget=budget,
            dropped_candidates=[],
            ordering_policy=config.ordering_policy,
            trace=[],
            selection_labels={},
        )

    selected: list[Candidate] = []
    remaining: list[Candidate] = list(candidates)
    selection_labels: dict[str, str] = {}
    trace: list[SelectionTraceStep] = []
    tokens_used = 0

    while remaining:
        tentative_label = assign_label(len(selected))

        best_candidate = None
        best_score: float = float("-inf")
        best_breakdown: dict = {}
        best_cost: int = 0

        for candidate in remaining:
            cost = effective_token_cost(candidate, tentative_label, config.llm_model)
            if tokens_used + cost > budget:
                continue

            score, breakdown = ranker.marginal_score(
                candidate, selected, query_embedding, config
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_breakdown = breakdown
                best_cost = cost

        if best_candidate is None:
            # No remaining candidate fits within the budget.
            break

        # Inclusion
        tokens_used += best_cost
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        selection_labels[best_candidate.chunk_id] = tentative_label

        # Populate selection-time fields on the chosen candidate.
        best_candidate.final_score = best_score
        best_candidate.score_breakdown = best_breakdown

        trace.append(
            SelectionTraceStep(
                step_index=len(trace),
                candidate_id=best_candidate.chunk_id,
                label=tentative_label,
                token_length=best_candidate.token_length or 0,
                effective_token_cost=best_cost,
                marginal_score=best_score,
                score_breakdown=best_breakdown,
                tokens_used_so_far=tokens_used,
            )
        )

        logger.debug(
            "select_greedy step %d: chose %s (label=%s score=%.4f cost=%d tokens_used=%d)",
            len(trace) - 1,
            best_candidate.chunk_id,
            tentative_label,
            best_score,
            best_cost,
            tokens_used,
        )

    dropped = remaining  # whatever did not get selected

    logger.info(
        "select_greedy: selected %d / %d candidates (%d tokens, budget %d)",
        len(selected),
        len(candidates),
        tokens_used,
        budget,
    )

    return SelectionResult(
        selected_candidates=selected,
        tokens_used=tokens_used,
        token_budget=budget,
        dropped_candidates=dropped,
        ordering_policy=config.ordering_policy,
        trace=trace,
        selection_labels=selection_labels,
    )
