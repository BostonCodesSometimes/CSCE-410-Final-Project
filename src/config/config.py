"""Pipeline configuration dataclass.

Single source of truth for all LitePack-RAG defaults.
All other modules must accept a PipelineConfig rather than reading
raw global constants.  CLI flags override these values at runtime via
dataclasses.replace().
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PipelineConfig:
    # ------------------------------------------------------------------ #
    # Retrieval and model settings                                        #
    # ------------------------------------------------------------------ #
    top_n: int = 20
    budget: int = 1800
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dims: int = 512
    mmr_lambda: float = 0.6

    # ------------------------------------------------------------------ #
    # LitePack composite-score weights                                    #
    # ------------------------------------------------------------------ #
    litepack_alpha: float = 1.0    # relevance coefficient
    litepack_beta: float = 0.6     # redundancy penalty
    litepack_gamma: float = 0.3    # keyword-coverage bonus
    litepack_delta: float = 0.2    # length-penalty coefficient
    litepack_epsilon: float = 0.2  # metadata-bonus coefficient

    # ------------------------------------------------------------------ #
    # Metadata subweights (used only when the matching toggle is True)   #
    # ------------------------------------------------------------------ #
    meta_w_recency: float = 1.0
    meta_w_pubtype: float = 1.0
    meta_w_mesh: float = 1.0

    # ------------------------------------------------------------------ #
    # Token budget and reference year                                     #
    # ------------------------------------------------------------------ #
    max_chunk_token_length: int = 300
    reference_year: int = field(default_factory=lambda: datetime.now().year)

    # ------------------------------------------------------------------ #
    # Feature toggles                                                     #
    # ------------------------------------------------------------------ #
    use_recency: bool = False
    use_keywords: bool = False
    use_publication_type: bool = False
    use_mesh: bool = False

    # ------------------------------------------------------------------ #
    # MeSH enrichment mode                                                #
    # ------------------------------------------------------------------ #
    mesh_mode: str = "off"  # off | query_time | cached

    # ------------------------------------------------------------------ #
    # Cache paths                                                         #
    # ------------------------------------------------------------------ #
    pubmed_cache_dir: str = "cache/pubmed"
    mesh_cache_dir: str = "cache/mesh"
    embedding_cache_dir: str = "cache/embeddings"

    # ------------------------------------------------------------------ #
    # Ordering policy (used by packer.order_selected)                    #
    # ------------------------------------------------------------------ #
    ordering_policy: str = "score_desc"  # score_desc | support_first | year_desc | original_rank

    # ------------------------------------------------------------------ #
    # Ranker and selector identifiers                                     #
    # ------------------------------------------------------------------ #
    ranker: str = "litepack"   # similarity | mmr | litepack
    selector: str = "greedy"   # baseline | greedy
