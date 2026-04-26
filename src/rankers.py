"""Ranking/scoring module for LitePack-RAG.

Responsibilities:
  - scoring math only
  - no token counting
  - no budget logic
  - no greedy selection loop (that lives in selector.py)

MMRRanker.rank() and LitePackRanker.rank() are static pre-sorts only.
The iterative greedy selection loop is owned entirely by selector.select_greedy().
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine

from config import PipelineConfig
from retriever import Candidate

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cosine similarity helper                                                     #
# --------------------------------------------------------------------------- #

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Reshapes both inputs to (1, d) before calling sklearn so the function
    works safely regardless of whether the caller passes a 1-D or 2-D array.

    Args:
        a: first vector, shape (d,) or (1, d).
        b: second vector, shape (d,) or (1, d).

    Returns:
        Scalar cosine similarity in [-1, 1].
    """
    return float(_sklearn_cosine(a.reshape(1, -1), b.reshape(1, -1))[0][0])


# --------------------------------------------------------------------------- #
# BaseRanker                                                                   #
# --------------------------------------------------------------------------- #

class BaseRanker:
    """Abstract base class for all rankers.

    Subclasses must implement rank() and marginal_score().
    rank() returns a statically sorted candidate list; no greedy loop runs here.
    marginal_score() computes an incremental score given the already-selected set.
    """

    def rank(
        self,
        candidates: list[Candidate],
        query_embedding: np.ndarray,
        config: PipelineConfig,
    ) -> list[Candidate]:
        """Return a statically sorted copy of *candidates*.

        Must NOT populate final_score or score_breakdown on any candidate.
        Must NOT run a greedy selection loop.

        Args:
            candidates:      enriched Candidate objects from features.py.
            query_embedding: np.ndarray of shape (dims,).
            config:          PipelineConfig instance.

        Returns:
            Sorted list[Candidate].

        Raises:
            NotImplementedError: always — subclasses must override.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement rank()")

    def marginal_score(
        self,
        candidate: Candidate,
        selected_set: list[Candidate],
        query_embedding: np.ndarray,
        config: PipelineConfig,
    ) -> tuple[float, dict]:
        """Compute a marginal score for *candidate* given *selected_set*.

        Called repeatedly by selector.select_greedy() inside its greedy loop.
        Must NOT modify any field on *candidate*.

        Args:
            candidate:       the candidate being evaluated.
            selected_set:    candidates already chosen in this iteration.
            query_embedding: np.ndarray of shape (dims,).
            config:          PipelineConfig instance.

        Returns:
            A 2-tuple (score: float, breakdown: dict).

        Raises:
            NotImplementedError: always — subclasses must override.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement marginal_score()"
        )


# --------------------------------------------------------------------------- #
# SimilarityRanker                                                             #
# --------------------------------------------------------------------------- #

class SimilarityRanker(BaseRanker):
    """Ranks candidates by raw retrieval score (higher = better).

    marginal_score() returns the retrieval_score directly with no embedding
    computation.  Used by the similarity + baseline pipeline.
    """

    def rank(
        self,
        candidates: list[Candidate],
        query_embedding: np.ndarray,
        config: PipelineConfig,
    ) -> list[Candidate]:
        """Sort candidates by retrieval_score descending.

        Does not populate final_score or score_breakdown.

        Args:
            candidates:      list of enriched Candidate objects.
            query_embedding: not used by this ranker; accepted for API symmetry.
            config:          not used by this ranker; accepted for API symmetry.

        Returns:
            New list sorted by retrieval_score descending.
        """
        return sorted(candidates, key=lambda c: c.retrieval_score, reverse=True)

    def marginal_score(
        self,
        candidate: Candidate,
        selected_set: list[Candidate],
        query_embedding: np.ndarray,
        config: PipelineConfig,
    ) -> tuple[float, dict]:
        """Return retrieval_score as the marginal score.

        Args:
            candidate:       candidate being evaluated.
            selected_set:    not used; accepted for API symmetry.
            query_embedding: not used; accepted for API symmetry.
            config:          not used; accepted for API symmetry.

        Returns:
            (retrieval_score, {"relevance": retrieval_score})
        """
        score = float(candidate.retrieval_score)
        breakdown: dict = {"relevance": score}
        return score, breakdown


# --------------------------------------------------------------------------- #
# MMRRanker                                                                    #
# --------------------------------------------------------------------------- #

class MMRRanker(BaseRanker):
    """Maximal Marginal Relevance ranker.

    rank() is a static pre-sort only — the greedy selection loop runs in
    selector.select_greedy().

    marginal_score() computes the standard MMR criterion:
        score = lambda * relevance - (1 - lambda) * redundancy
    """

    def rank(
        self,
        candidates: list[Candidate],
        query_embedding: np.ndarray,
        config: PipelineConfig,
    ) -> list[Candidate]:
        """Static pre-sort by retrieval_score descending.

        Does NOT run a greedy MMR loop.  The greedy loop is owned by
        selector.select_greedy().

        Args:
            candidates:      list of enriched Candidate objects.
            query_embedding: not used here; accepted for API symmetry.
            config:          not used here; accepted for API symmetry.

        Returns:
            New list sorted by retrieval_score descending.
        """
        return sorted(candidates, key=lambda c: c.retrieval_score, reverse=True)

    def marginal_score(
        self,
        candidate: Candidate,
        selected_set: list[Candidate],
        query_embedding: np.ndarray,
        config: PipelineConfig,
    ) -> tuple[float, dict]:
        """Compute MMR marginal score.

        relevance  = cosine_sim(candidate.embedding, query_embedding)
        redundancy = max cosine_sim(candidate.embedding, s.embedding)
                     over all s in selected_set; 0.0 when selected_set is empty.
        score      = config.mmr_lambda * relevance
                     - (1 - config.mmr_lambda) * redundancy

        Args:
            candidate:       candidate being evaluated.
            selected_set:    candidates already selected in this iteration.
            query_embedding: np.ndarray of shape (dims,).
            config:          PipelineConfig with mmr_lambda.

        Returns:
            (score, {"relevance": relevance, "redundancy": redundancy})

        Raises:
            ValueError: if candidate.embedding is None.
        """
        if candidate.embedding is None:
            raise ValueError(
                f"MMRRanker.marginal_score: candidate.embedding is None "
                f"for chunk_id={candidate.chunk_id!r}. "
                "Run features.add_embeddings() before ranking."
            )

        relevance = _cosine_sim(candidate.embedding, query_embedding)

        if selected_set:
            redundancy = max(
                _cosine_sim(candidate.embedding, s.embedding)
                for s in selected_set
                if s.embedding is not None
            )
        else:
            redundancy = 0.0

        lam = config.mmr_lambda
        score = lam * relevance - (1.0 - lam) * redundancy

        breakdown: dict = {
            "relevance": relevance,
            "redundancy": redundancy,
        }
        return float(score), breakdown


# --------------------------------------------------------------------------- #
# LitePackRanker                                                               #
# --------------------------------------------------------------------------- #

class LitePackRanker(BaseRanker):
    """LitePack multi-factor ranker.

    rank() is a static pre-sort only — the greedy selection loop runs in
    selector.select_greedy().

    marginal_score() combines relevance, redundancy, keyword coverage,
    length penalty, and optional metadata bonuses into a single composite score.
    """

    def rank(
        self,
        candidates: list[Candidate],
        query_embedding: np.ndarray,
        config: PipelineConfig,
    ) -> list[Candidate]:
        """Static pre-sort by retrieval_score descending.

        Does NOT run a greedy LitePack loop.  The greedy loop is owned by
        selector.select_greedy().

        Args:
            candidates:      list of enriched Candidate objects.
            query_embedding: not used here; accepted for API symmetry.
            config:          not used here; accepted for API symmetry.

        Returns:
            New list sorted by retrieval_score descending.
        """
        return sorted(candidates, key=lambda c: c.retrieval_score, reverse=True)

    def marginal_score(
        self,
        candidate: Candidate,
        selected_set: list[Candidate],
        query_embedding: np.ndarray,
        config: PipelineConfig,
    ) -> tuple[float, dict]:
        """Compute LitePack composite marginal score.

        Components:
          relevance      = cosine_sim(candidate.embedding, query_embedding)
          redundancy     = max cosine similarity to selected_set; 0.0 if empty
          coverage       = candidate.keyword_overlap or 0.0
          length_penalty = min(token_length / max_chunk_token_length, 1.0)
          metadata_bonus = weighted average of enabled metadata feature scores

        Final score:
          alpha * relevance
          - beta  * redundancy
          + gamma * coverage
          - delta * length_penalty
          + epsilon * metadata_bonus

        Disabled metadata features do not affect the denominator of the
        weighted average (only enabled features contribute to it).

        Args:
            candidate:       candidate being evaluated.
            selected_set:    candidates already selected in this iteration.
            query_embedding: np.ndarray of shape (dims,).
            config:          PipelineConfig with all litepack_* weights and
                             feature toggles.

        Returns:
            (score, breakdown_dict) where breakdown contains keys:
              relevance, redundancy, coverage, length_penalty,
              metadata_bonus, recency_score, publication_type_score, mesh_score

        Raises:
            ValueError: if candidate.embedding is None.
        """
        if candidate.embedding is None:
            raise ValueError(
                f"LitePackRanker.marginal_score: candidate.embedding is None "
                f"for chunk_id={candidate.chunk_id!r}. "
                "Run features.add_embeddings() before ranking."
            )

        # ------------------------------------------------------------------ #
        # Core components                                                     #
        # ------------------------------------------------------------------ #
        relevance = _cosine_sim(candidate.embedding, query_embedding)

        if selected_set:
            redundancy = max(
                _cosine_sim(candidate.embedding, s.embedding)
                for s in selected_set
                if s.embedding is not None
            )
        else:
            redundancy = 0.0

        coverage = candidate.keyword_overlap or 0.0

        token_len = candidate.token_length if candidate.token_length is not None else 0
        length_penalty = min(token_len / config.max_chunk_token_length, 1.0)

        # ------------------------------------------------------------------ #
        # Metadata bonus — only enabled features enter the weighted average  #
        # ------------------------------------------------------------------ #
        metadata_components: list[tuple[float, float]] = []
        metadata_breakdown: dict[str, float] = {}

        if config.use_recency:
            r = float(candidate.recency_score or 0.0)
            metadata_components.append((config.meta_w_recency, r))
            metadata_breakdown["recency_score"] = r
        else:
            metadata_breakdown["recency_score"] = 0.0

        if config.use_publication_type:
            p = float(candidate.publication_type_score or 0.0)
            metadata_components.append((config.meta_w_pubtype, p))
            metadata_breakdown["publication_type_score"] = p
        else:
            metadata_breakdown["publication_type_score"] = 0.0

        if config.use_mesh:
            m = float(candidate.mesh_score or 0.0)
            metadata_components.append((config.meta_w_mesh, m))
            metadata_breakdown["mesh_score"] = m
        else:
            metadata_breakdown["mesh_score"] = 0.0

        if metadata_components:
            weighted_sum = sum(w * v for w, v in metadata_components)
            weight_total = sum(w for w, _ in metadata_components)
            metadata_bonus = weighted_sum / weight_total if weight_total > 0 else 0.0
        else:
            metadata_bonus = 0.0

        # ------------------------------------------------------------------ #
        # Final composite score                                               #
        # ------------------------------------------------------------------ #
        score = (
            config.litepack_alpha   * relevance
            - config.litepack_beta  * redundancy
            + config.litepack_gamma * coverage
            - config.litepack_delta * length_penalty
            + config.litepack_epsilon * metadata_bonus
        )

        breakdown: dict = {
            "relevance": relevance,
            "redundancy": redundancy,
            "coverage": coverage,
            "length_penalty": length_penalty,
            "metadata_bonus": metadata_bonus,
            **metadata_breakdown,
        }
        return float(score), breakdown
