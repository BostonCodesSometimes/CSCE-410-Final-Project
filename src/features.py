"""Feature enrichment module for LitePack-RAG.

Responsibilities:
  - add feature signals to Candidate objects after retrieval
  - batch re-embed candidate texts for downstream rankers
  - use only stdlib regex tokenization (no NLTK)

Nothing here performs ranking, selection, or scoring decisions.
All enrichment is in-place mutation of the same Candidate list returned
by retriever.py.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import tiktoken

from config import PipelineConfig
from retriever import Candidate

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Stdlib tokenization (no NLTK dependency)                                    #
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Return a set of lowercase alphanumeric tokens from *text*.

    Returns an empty set for falsy input so callers never get a TypeError.
    """
    return set(_TOKEN_RE.findall(text.lower())) if text else set()


# --------------------------------------------------------------------------- #
# Public enrichment functions                                                  #
# --------------------------------------------------------------------------- #

def add_token_lengths(
    candidates: list[Candidate],
    model: str,  # kept for API symmetry; encoding is fixed to cl100k_base
) -> list[Candidate]:
    """Populate ``candidate.token_length`` for every candidate.

    Uses the ``cl100k_base`` tiktoken encoding regardless of the model
    argument, which keeps the token budget comparable across model switches.

    Args:
        candidates: list of Candidate objects from retriever.retrieve().
        model:      embedding or LLM model string (unused internally; the
                    encoding is always cl100k_base per the plan).

    Returns:
        The same list, with ``token_length`` set on each candidate.
    """
    if not candidates:
        return candidates

    enc = tiktoken.get_encoding("cl100k_base")
    for c in candidates:
        c.token_length = len(enc.encode(c.text))

    logger.debug("add_token_lengths: set token_length for %d candidates", len(candidates))
    return candidates


def add_keyword_overlap(
    candidates: list[Candidate],
    query: str,
) -> list[Candidate]:
    """Populate ``candidate.keyword_overlap`` using Jaccard similarity.

    Token sets:
      - query tokens:     _tokenize(query)
      - candidate tokens: {k.lower() for k in c.keywords} | _tokenize(c.title)
                          falls back to _tokenize(c.text) when that set is empty

    Jaccard overlap = |query_tokens ∩ candidate_tokens| / |query_tokens ∪ candidate_tokens|

    Sets 0.0 when either set is empty.

    Args:
        candidates: list of Candidate objects.
        query:      raw query string from the user.

    Returns:
        The same list, with ``keyword_overlap`` set on each candidate.
    """
    if not candidates:
        return candidates

    query_tokens = _tokenize(query)

    for c in candidates:
        structured_tokens: set[str] = (
            {k.lower() for k in c.keywords} | _tokenize(c.title)
        )
        candidate_tokens = structured_tokens if structured_tokens else _tokenize(c.text)

        union = query_tokens | candidate_tokens
        if not query_tokens or not candidate_tokens or not union:
            c.keyword_overlap = 0.0
        else:
            c.keyword_overlap = len(query_tokens & candidate_tokens) / len(union)

    logger.debug(
        "add_keyword_overlap: set keyword_overlap for %d candidates", len(candidates)
    )
    return candidates


def add_recency_feature(
    candidates: list[Candidate],
    config: PipelineConfig,
) -> list[Candidate]:
    """Populate ``candidate.recency_score`` based on publication year.

    Formula:
        recency_score = max(0.0, 1.0 - (reference_year - year) / 20.0)

    A paper published in ``reference_year`` scores 1.0; one published
    20 or more years ago scores 0.0.  Missing or unparseable years score 0.0.

    Args:
        candidates: list of Candidate objects.
        config:     PipelineConfig carrying ``reference_year``.

    Returns:
        The same list, with ``recency_score`` set on each candidate.
    """
    if not candidates:
        return candidates

    ref_year = config.reference_year

    for c in candidates:
        year_str = c.year
        if not year_str or str(year_str).strip().upper() in ("", "N/A", "NONE", "NULL"):
            c.recency_score = 0.0
            continue
        try:
            year_int = int(str(year_str).strip()[:4])
            c.recency_score = max(0.0, 1.0 - (ref_year - year_int) / 20.0)
        except (ValueError, TypeError):
            logger.debug(
                "add_recency_feature: unparseable year %r for chunk_id=%s; using 0.0",
                year_str,
                c.chunk_id,
            )
            c.recency_score = 0.0

    logger.debug(
        "add_recency_feature: set recency_score for %d candidates (ref_year=%d)",
        len(candidates),
        ref_year,
    )
    return candidates


def add_embeddings(
    candidates: list[Candidate],
    embeddings_model: Any,
) -> list[Candidate]:
    """Populate ``candidate.embedding`` for candidates that are missing it.

    When the pipeline uses the direct Pinecone retrieval path
    (``Retriever.retrieve()`` with ``include_values=True``), each Candidate
    already has its stored embedding pre-populated.  In that case this
    function detects the pre-populated embeddings and returns immediately
    without making any OpenAI API call.

    When the legacy LangChain retrieval path is used
    (``Retriever._retrieve_langchain_legacy()``), stored vectors are not
    available, so ``candidate.embedding`` will be ``None`` and this function
    falls back to the original behaviour: one batched
    ``embeddings_model.embed_documents(...)`` call for all missing candidates.

    Only candidates whose ``embedding`` is ``None`` are sent to the API.
    Candidates that already have an embedding are left untouched.

    Args:
        candidates:       list of Candidate objects.
        embeddings_model: a LangChain embeddings object that exposes
                          ``embed_documents(list[str]) -> list[list[float]]``.

    Returns:
        The same list, with ``embedding`` set as a ``np.ndarray`` on every
        candidate that was previously missing one.
    """
    if not candidates:
        return candidates

    missing = [c for c in candidates if c.embedding is None]

    if not missing:
        logger.info(
            "add_embeddings: all %d candidate embeddings already present "
            "(from direct Pinecone retrieval); skipping embed_documents call.",
            len(candidates),
        )
        return candidates

    if len(missing) < len(candidates):
        logger.info(
            "add_embeddings: embedding %d missing candidate text(s) only "
            "(%d/%d already have embeddings).",
            len(missing),
            len(candidates) - len(missing),
            len(candidates),
        )
    else:
        logger.info(
            "add_embeddings: no pre-existing embeddings; "
            "embedding all %d candidate texts via embed_documents.",
            len(missing),
        )

    texts = [c.text for c in missing]
    vectors = embeddings_model.embed_documents(texts)

    for c, vector in zip(missing, vectors):
        c.embedding = np.array(vector, dtype=np.float32)

    logger.debug(
        "add_embeddings: embedded %d candidates (dim=%d)",
        len(missing),
        len(vectors[0]) if vectors else 0,
    )
    return candidates


# --------------------------------------------------------------------------- #
# Optional feature stubs (Phase 9)                                             #
# --------------------------------------------------------------------------- #

def add_publication_type_feature(
    candidates: list[Candidate],
    config: PipelineConfig,  # noqa: ARG001
) -> list[Candidate]:
    """Stub: publication-type scoring is not yet implemented.

    Sets ``publication_type_score = 0.0`` for all candidates and logs a
    warning.  Will be replaced when pubmed_client.py is available.

    Args:
        candidates: list of Candidate objects.
        config:     PipelineConfig (reserved for future use).

    Returns:
        The same list, with ``publication_type_score`` set to 0.0.
    """
    logger.warning(
        "add_publication_type_feature is not implemented yet; "
        "setting publication_type_score=0.0 for all candidates."
    )
    for c in candidates:
        c.publication_type_score = 0.0
    return candidates


def add_mesh_feature(
    candidates: list[Candidate],
    query: str,  # noqa: ARG001
    config: PipelineConfig,  # noqa: ARG001
) -> list[Candidate]:
    """Stub: MeSH-term scoring is not yet implemented.

    Sets ``mesh_score = 0.0`` for all candidates and logs a warning.
    Will be replaced when pubmed_client.py and MeSH data are available.

    Args:
        candidates: list of Candidate objects.
        query:      user query string (reserved for future use).
        config:     PipelineConfig (reserved for future use).

    Returns:
        The same list, with ``mesh_score`` set to 0.0.
    """
    logger.warning(
        "add_mesh_feature is not implemented yet; "
        "setting mesh_score=0.0 for all candidates."
    )
    for c in candidates:
        c.mesh_score = 0.0
    return candidates


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #

def enrich_candidates(
    candidates: list[Candidate],
    query: str,
    query_embedding: np.ndarray,  # noqa: ARG001 — reserved for Phase 3 rankers
    embeddings_model: Any,
    config: PipelineConfig,
) -> list[Candidate]:
    """Apply all enabled feature enrichers to *candidates* in order.

    Always runs:
      - add_token_lengths
      - add_embeddings

    Conditionally runs (controlled by config toggles):
      - add_keyword_overlap  when config.use_keywords is True
      - add_recency_feature  when config.use_recency is True
      - add_publication_type_feature (stub) when config.use_publication_type is True
      - add_mesh_feature (stub) when config.use_mesh is True

    This function does not perform any ranking or selection.  It only
    populates fields on existing Candidate objects; the list order and
    all retrieval fields are preserved exactly as received.

    Args:
        candidates:       list[Candidate] from retriever.retrieve().
        query:            raw query string.
        query_embedding:  np.ndarray of the query vector (shape: (dims,)).
                          Accepted here so the caller's signature is stable;
                          rankers will use it in Phase 3.
        embeddings_model: LangChain embeddings object.
        config:           PipelineConfig with feature toggles.

    Returns:
        The same list (enriched in-place), for convenience.
    """
    if not candidates:
        logger.debug("enrich_candidates: no candidates to enrich")
        return candidates

    # Always-on enrichers
    add_token_lengths(candidates, config.embedding_model)
    add_embeddings(candidates, embeddings_model)

    # Conditional enrichers
    if config.use_keywords:
        add_keyword_overlap(candidates, query)

    if config.use_recency:
        add_recency_feature(candidates, config)

    if config.use_publication_type:
        add_publication_type_feature(candidates, config)

    if config.use_mesh:
        add_mesh_feature(candidates, query, config)

    logger.info(
        "enrich_candidates: enriched %d candidates "
        "(keywords=%s recency=%s pubtype=%s mesh=%s)",
        len(candidates),
        config.use_keywords,
        config.use_recency,
        config.use_publication_type,
        config.use_mesh,
    )
    return candidates
