"""Retriever module for LitePack-RAG.

Responsibilities:
  - embed the user query (one OpenAI call per retrieve())
  - query Pinecone directly with include_values=True so stored embeddings are
    returned in the same round-trip — no second embedding call needed
  - apply score-direction normalization (once, here, never elsewhere)
  - return Candidate objects (with .embedding pre-populated) and the query
    embedding as a numpy array

Nothing else.  No MMR, no ranking, no selection.

Architecture note
-----------------
The main retrieve() path calls self.index.query(..., include_values=True) so
that each Candidate.embedding is populated from the Pinecone-stored vector.
This eliminates the previously necessary second OpenAI embedding call that
features.add_embeddings() used to make for all retrieved chunks.

A legacy LangChain path (_retrieve_langchain_legacy) is preserved for
debugging/fallback; it does NOT return stored vectors, so features.py will
still need to re-embed candidates when that path is used.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from config import PipelineConfig
from pinecone_settings import get_pinecone_config

load_dotenv()

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Candidate dataclass                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    """A single retrieved chunk with all fields needed by downstream modules.

    Fields populated at retrieval time
    -----------------------------------
    chunk_id        unique chunk identifier  (doc.metadata["id"])
    doc_id          parent document id       (doc.metadata["doc_id"])
    pmid            PubMed ID                (doc.metadata["doc_id"])
    text            chunk text               (doc.page_content)
    metadata        raw metadata dict
    retrieval_score normalized score; higher always means more relevant
    url, title, year, journal, keywords  — convenience copies from metadata

    Fields populated by features.py
    --------------------------------
    embedding, token_length, keyword_overlap, recency_score,
    publication_type, publication_type_score, mesh_terms, mesh_score

    Fields populated by selector.py
    --------------------------------
    redundancy_score, final_score, score_breakdown
    """

    chunk_id: str
    doc_id: str
    pmid: str
    text: str
    metadata: dict
    retrieval_score: float
    url: str
    title: str
    year: str
    journal: str
    keywords: list = field(default_factory=list)

    # Populated by features.py
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    token_length: Optional[int] = None
    keyword_overlap: Optional[float] = None
    recency_score: Optional[float] = None
    publication_type: Optional[str] = None
    publication_type_score: Optional[float] = None
    mesh_terms: list = field(default_factory=list)
    mesh_score: Optional[float] = None

    # Populated by selector.py
    redundancy_score: Optional[float] = None
    final_score: Optional[float] = None
    score_breakdown: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Retriever class                                                               #
# --------------------------------------------------------------------------- #

class Retriever:
    """Embeds a query, queries Pinecone, and returns normalized Candidates.

    Score-direction normalization is applied exactly once inside retrieve().
    No other module should re-interpret or re-invert retrieval_score.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize embeddings model, Pinecone index, and detect metric.

        Args:
            config: PipelineConfig with embedding_model and embedding_dims.

        Raises:
            ValueError: if PINECONE_API_KEY is absent.
        """
        api_key, index_name, host, _ = get_pinecone_config()
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY is missing. "
                "Set it via environment variable or pinecone_creds.txt."
            )

        # Expose to caller (features.py re-uses this for candidate embeddings).
        self.config = config
        self.embeddings_model = OpenAIEmbeddings(
            model=config.embedding_model,
            dimensions=config.embedding_dims,
        )

        pc = Pinecone(api_key=api_key)

        # ------------------------------------------------------------------ #
        # Detect index metric so retrieve() can normalize score direction.   #
        #                                                                     #
        # cosine / dotproduct  ->  higher raw score = more relevant (keep)   #
        # euclidean            ->  lower raw score  = more relevant (negate) #
        # ------------------------------------------------------------------ #
        try:
            desc = pc.describe_index(index_name)
            raw_metric = getattr(desc, "metric", None)
            self.index_metric: str = str(raw_metric).lower() if raw_metric else "cosine"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not read index metric from describe_index (%s). "
                "Defaulting to 'cosine' (higher-is-better).",
                exc,
            )
            self.index_metric = "cosine"

        # self.index is the raw Pinecone Index used by the main retrieve() path.
        # self.vectorstore wraps the same index for the legacy LangChain path.
        self.index = pc.Index(index_name, host=host) if host else pc.Index(index_name)
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings_model,
        )

        logger.info(
            "Retriever initialized  index=%s  metric=%s",
            index_name,
            self.index_metric,
        )

    def retrieve(
        self, query: str, top_n: int
    ) -> tuple:
        """Embed query once, query Pinecone directly, return Candidates with embeddings.

        Uses the raw Pinecone index (self.index) with include_values=True so
        each returned Candidate.embedding is populated from the stored vector.
        This means features.add_embeddings() will detect pre-populated
        embeddings and skip the second OpenAI call entirely.

        Score normalization is applied exactly once here:
          - cosine or dotproduct: raw score is already higher-is-better.
          - euclidean:            raw score is a distance; negate it so that
                                  higher retrieval_score always means more
                                  relevant (consistent with the rest of the
                                  pipeline).

        Args:
            query:  natural-language query string.
            top_n:  maximum number of candidates to return.

        Returns:
            A 2-tuple:
              candidates      list[Candidate] sorted by retrieval_score desc,
                              each with .embedding pre-populated as np.float32.
              query_embedding np.ndarray of shape (embedding_dims,).

        If Pinecone returns no results the function returns ([], query_embedding)
        and does not raise.
        """
        logger.info(
            "retrieve: embedding query via OpenAI (model=%s)",
            self.config.embedding_model,
        )
        query_vector = self.embeddings_model.embed_query(query)
        query_embedding = np.array(query_vector, dtype=np.float32)

        logger.info(
            "retrieve: querying Pinecone directly  top_k=%d  include_values=True  metric=%s",
            top_n,
            self.index_metric,
        )
        response = self.index.query(
            vector=query_vector,
            top_k=top_n,
            include_metadata=True,
            include_values=True,
        )

        matches = response.matches
        if not matches:
            logger.warning("Pinecone query returned no results for query: %r", query)
            return [], query_embedding

        candidates: list = []
        for match in matches:
            # One-time normalization: all downstream code assumes higher = better.
            if self.index_metric == "euclidean":
                retrieval_score = -float(match.score)
            else:
                retrieval_score = float(match.score)

            meta = match.metadata or {}

            # Extract text robustly; LangChain stores it under "text" by default.
            text = (
                meta.get("text")
                or meta.get("page_content")
                or meta.get("chunk_text")
                or ""
            )
            if not text:
                logger.warning(
                    "retrieve: no text field found in Pinecone metadata for id=%r; "
                    "using empty string. Check stored metadata keys.",
                    match.id,
                )

            # chunk_id from the Pinecone record id; doc_id/pmid from metadata.
            chunk_id = match.id
            doc_id = meta.get("doc_id", "")

            # Populate embedding from the stored Pinecone vector values so that
            # features.add_embeddings() can skip the second OpenAI call.
            values = match.values
            embedding = np.array(values, dtype=np.float32) if values else None

            c = Candidate(
                chunk_id=chunk_id,
                doc_id=doc_id,
                pmid=doc_id,
                text=text,
                metadata=dict(meta),
                retrieval_score=retrieval_score,
                url=meta.get("url", ""),
                title=meta.get("title", ""),
                year=meta.get("year", "N/A"),
                journal=meta.get("journal", "Unknown"),
                keywords=meta.get("keywords", []),
            )
            c.embedding = embedding
            candidates.append(c)

        n_with_emb = sum(1 for c in candidates if c.embedding is not None)
        logger.info(
            "retrieve: returned %d candidates  embeddings_from_pinecone=%d/%d",
            len(candidates),
            n_with_emb,
            len(candidates),
        )
        return candidates, query_embedding

    def _retrieve_langchain_legacy(
        self, query: str, top_n: int
    ) -> tuple:
        """Legacy LangChain retrieval method — retained for debugging/fallback only.

        Uses PineconeVectorStore.similarity_search_with_score() which does NOT
        return the stored Pinecone vector values.  Consequently, the returned
        Candidate objects have .embedding = None, and features.add_embeddings()
        will need to make a second OpenAI embedding API call to populate them.

        Do NOT use this as the default path.  Use retrieve() instead, which
        calls self.index.query(..., include_values=True) and pre-populates
        Candidate.embedding from the stored vectors, eliminating the extra
        embedding call.

        Args:
            query:  natural-language query string.
            top_n:  maximum number of candidates to return.

        Returns:
            A 2-tuple (candidates, query_embedding) with the same structure as
            retrieve(), except Candidate.embedding is always None.
        """
        query_vector = self.embeddings_model.embed_query(query)
        query_embedding = np.array(query_vector, dtype=np.float32)

        results = self.vectorstore.similarity_search_with_score(query, k=top_n)

        if not results:
            logger.warning(
                "_retrieve_langchain_legacy: similarity_search_with_score "
                "returned no results for query: %r",
                query,
            )
            return [], query_embedding

        candidates: list = []
        for doc, score in results:
            # One-time normalization: all downstream code assumes higher = better.
            if self.index_metric == "euclidean":
                retrieval_score = -float(score)
            else:
                retrieval_score = float(score)

            candidates.append(
                Candidate(
                    chunk_id=doc.metadata["id"],
                    doc_id=doc.metadata["doc_id"],
                    pmid=doc.metadata["doc_id"],
                    text=doc.page_content,
                    metadata=dict(doc.metadata),
                    retrieval_score=retrieval_score,
                    url=doc.metadata.get("url", ""),
                    title=doc.metadata.get("title", ""),
                    year=doc.metadata.get("year", "N/A"),
                    journal=doc.metadata.get("journal", "Unknown"),
                    keywords=doc.metadata.get("keywords", []),
                )
            )

        return candidates, query_embedding
