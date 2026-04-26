"""Context packing module for LitePack-RAG.

Responsibilities:
  - canonical token counting (single source of truth for the whole pipeline)
  - canonical label and header formatting (imported by selector.py)
  - ordering policy for display order
  - final prompt-context assembly: build_context()

Nothing here performs retrieval, ranking, or selection.

Circular-import note
--------------------
selector.py imports assign_label, format_chunk_header, count_tokens, and
effective_token_cost from this module at runtime.  build_context() takes a
SelectionResult argument, but packer.py must NOT import selector at runtime or
Python raises a circular import.  The SelectionResult annotation is guarded by
TYPE_CHECKING so it is a pure string annotation at runtime and never resolved.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import tiktoken

from config import PipelineConfig
from retriever import Candidate

if TYPE_CHECKING:
    from selector import SelectionResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Canonical token counting                                                     #
# --------------------------------------------------------------------------- #

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in *text* using the cl100k_base encoding.

    The *model* argument is accepted for API symmetry with the OpenAI client
    but is not used to select the encoding — cl100k_base is always used so
    that token budgets are stable across model switches.

    Args:
        text:  string to tokenize.
        model: model name (accepted for API symmetry; ignored internally).

    Returns:
        Token count as a non-negative integer.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# --------------------------------------------------------------------------- #
# Canonical label and header helpers                                           #
# (These are imported by selector.py — keep signatures stable.)               #
# --------------------------------------------------------------------------- #

def assign_label(index_zero_based: int) -> str:
    """Return the canonical selection label for a zero-based position index.

    Examples:
        assign_label(0) -> "S1"
        assign_label(4) -> "S5"

    Args:
        index_zero_based: zero-based position of the candidate in selection order.

    Returns:
        Label string "S{n}" where n = index_zero_based + 1.
    """
    return f"S{index_zero_based + 1}"


def format_chunk_header(candidate: Candidate, label: str) -> str:
    """Build the canonical chunk header string for a selected candidate.

    Format: "[{label}] PMID: {pmid} | Year: {year} | Title: {title}"

    Args:
        candidate: selected Candidate object.
        label:     selection label assigned to this candidate (e.g. "S1").

    Returns:
        Single-line header string.
    """
    return (
        f"[{label}] PMID: {candidate.pmid} "
        f"| Year: {candidate.year} "
        f"| Title: {candidate.title}"
    )


def effective_token_cost(candidate: Candidate, label: str, model: str) -> int:
    """Compute the token cost of including *candidate* in the context window.

    Cost covers the exact string that build_context() will produce:
        header + "\\n" + candidate.text + "\\n\\n"

    Budget checks in selector.py use this function so that the cost estimate
    is consistent with the final packed context.

    Args:
        candidate: Candidate object (must have .text, .pmid, .year, .title).
        label:     selection label (e.g. "S1").
        model:     model name forwarded to count_tokens.

    Returns:
        Integer token count for the full block.
    """
    header = format_chunk_header(candidate, label)
    block = header + "\n" + candidate.text + "\n\n"
    return count_tokens(block, model)


# --------------------------------------------------------------------------- #
# Ordering policy                                                              #
# (Lives here, not in selector.py, to prevent a selector<->packer import      #
#  cycle.  build_context() calls order_selected() directly.)                  #
# --------------------------------------------------------------------------- #

def order_selected(selected: list[Candidate], policy: str) -> list[Candidate]:
    """Reorder *selected* candidates for display according to *policy*.

    This function lives in packer.py so that build_context() can call it
    without importing from selector.py, which would create a circular import.

    Supported policies
    ------------------
    "score_desc"    Sort by final_score descending (default / fallback).
    "support_first" Sort by retrieval_score descending.
    "year_desc"     Sort by publication year descending; unparseable years last.
    "original_rank" Preserve the order in the given list.

    Unrecognized policy strings fall back to "score_desc" with a warning.

    Args:
        selected: list of selected Candidate objects.
        policy:   ordering policy string from PipelineConfig.ordering_policy.

    Returns:
        New list in the requested order (input list is not mutated).
    """
    if policy == "original_rank":
        return list(selected)

    if policy == "support_first":
        return sorted(selected, key=lambda c: c.retrieval_score, reverse=True)

    if policy == "year_desc":
        def _year_key(c: Candidate) -> int:
            try:
                return int(str(c.year).strip()[:4])
            except (ValueError, TypeError, AttributeError):
                return 0  # missing/unparseable years sort last

        return sorted(selected, key=_year_key, reverse=True)

    if policy != "score_desc":
        logger.warning(
            "order_selected: unrecognized policy %r; falling back to score_desc",
            policy,
        )

    # score_desc (default)
    return sorted(
        selected,
        key=lambda c: float(c.final_score) if c.final_score is not None else 0.0,
        reverse=True,
    )


# --------------------------------------------------------------------------- #
# PackedContext dataclass                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class PackedContext:
    """Output of build_context(): the assembled context string and metadata.

    Attributes:
        context_text:   full formatted context string to pass to the LLM.
        source_blocks:  list of per-block metadata dicts
                        (keys: label, pmid, year, title, journal, url,
                         chunk_id, token_length, retrieval_score, final_score).
        tokens_used:    actual token count of context_text.
        selected_pmids: PMIDs of selected sources in display order.
        citation_map:   maps source label -> Candidate (e.g. "S1" -> Candidate).
    """

    context_text: str
    source_blocks: list[dict]
    tokens_used: int
    selected_pmids: list[str]
    citation_map: dict[str, Candidate]


# --------------------------------------------------------------------------- #
# Context assembly                                                              #
# --------------------------------------------------------------------------- #

def build_context(
    selection_result: SelectionResult,
    config: PipelineConfig,
) -> PackedContext:
    """Assemble the final LLM prompt context from a SelectionResult.

    Labels assigned during selection are reused exactly — they are NOT
    recomputed from display order.  This guarantees that S3 in a source block
    always refers to the same PMID as S3 cited in the generated answer.

    Steps:
      1. Reorder selected candidates with order_selected().
      2. For each candidate, look up its label from selection_result.selection_labels.
      3. Build block strings: header + "\\n" + candidate.text.
      4. Join all blocks with "\\n\\n".
      5. Count tokens on the assembled context string.

    Args:
        selection_result: output from select_baseline() or select_greedy().
        config:           PipelineConfig carrying ordering_policy and llm_model.

    Returns:
        PackedContext with all fields populated.  Returns an empty PackedContext
        if selection_result.selected_candidates is empty.
    """
    if not selection_result.selected_candidates:
        logger.debug("build_context: no selected candidates; returning empty PackedContext")
        return PackedContext(
            context_text="",
            source_blocks=[],
            tokens_used=0,
            selected_pmids=[],
            citation_map={},
        )

    ordered = order_selected(
        selection_result.selected_candidates,
        policy=selection_result.ordering_policy,
    )

    blocks: list[str] = []
    source_blocks: list[dict] = []
    selected_pmids: list[str] = []
    citation_map: dict[str, Candidate] = {}

    for c in ordered:
        # Reuse the label assigned at selection time — do NOT reassign.
        label = selection_result.selection_labels[c.chunk_id]

        header = format_chunk_header(c, label)
        block_text = header + "\n" + c.text
        blocks.append(block_text)

        source_blocks.append({
            "label": label,
            "pmid": c.pmid,
            "year": c.year,
            "title": c.title,
            "journal": c.journal,
            "url": c.url,
            "chunk_id": c.chunk_id,
            "token_length": c.token_length,
            "retrieval_score": float(c.retrieval_score),
            "final_score": float(c.final_score) if c.final_score is not None else None,
        })
        selected_pmids.append(c.pmid)
        citation_map[label] = c

    context_text = "\n\n".join(blocks)
    tokens_used = count_tokens(context_text, config.llm_model)

    logger.info(
        "build_context: assembled %d blocks, %d tokens (budget %d)",
        len(blocks),
        tokens_used,
        selection_result.token_budget,
    )

    return PackedContext(
        context_text=context_text,
        source_blocks=source_blocks,
        tokens_used=tokens_used,
        selected_pmids=selected_pmids,
        citation_map=citation_map,
    )
