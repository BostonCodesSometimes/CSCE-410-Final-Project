"""Unit tests for features.add_embeddings().

Verifies the short-circuit behaviour introduced alongside the direct Pinecone
retrieval path (Retriever.retrieve() with include_values=True):

  - If all Candidate objects already have a non-None embedding, embed_documents
    must NOT be called at all.
  - If some Candidate objects are missing embeddings, embed_documents must be
    called with ONLY the texts of the missing candidates; candidates that
    already have embeddings must not be re-embedded or mutated.
  - If ALL Candidate objects are missing embeddings (legacy LangChain path),
    embed_documents is called for every candidate, preserving the original
    batch behaviour.

Run from the project root:

    python -m pytest tests/test_add_embeddings.py -v
    # or without pytest:
    python tests/test_add_embeddings.py
"""
from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import MagicMock, call
from dataclasses import field

import numpy as np

# ---------------------------------------------------------------------------
# Make src/ importable without installing the package.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import add_embeddings  # noqa: E402  (after sys.path manipulation)
from retriever import Candidate      # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    chunk_id: str = "c1",
    text: str = "sample chunk text",
    embedding: np.ndarray | None = None,
) -> Candidate:
    """Return a minimal Candidate with only the fields add_embeddings() cares about."""
    c = Candidate(
        chunk_id=chunk_id,
        doc_id="d1",
        pmid="d1",
        text=text,
        metadata={},
        retrieval_score=0.9,
        url="",
        title="Test Title",
        year="2023",
        journal="Test Journal",
    )
    c.embedding = embedding
    return c


def _fake_embedding(dim: int = 4) -> np.ndarray:
    """Return a deterministic fake embedding vector."""
    return np.ones(dim, dtype=np.float32)


def _mock_embeddings_model(*texts_per_call: list[str], dim: int = 4):
    """Build a MagicMock whose embed_documents() returns fake vectors.

    Each positional argument is the list of texts expected in one call.
    The mock returns ``len(texts)`` fake vectors per call.
    """
    mock = MagicMock()
    mock.embed_documents.side_effect = [
        [_fake_embedding(dim).tolist() for _ in texts]
        for texts in texts_per_call
    ]
    return mock


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestAddEmbeddingsAllPresent(unittest.TestCase):
    """embed_documents must NOT be called when all candidates have embeddings."""

    def test_no_api_call_when_all_have_embeddings(self):
        existing_emb = _fake_embedding()
        candidates = [
            _make_candidate("c1", "text one", embedding=existing_emb.copy()),
            _make_candidate("c2", "text two", embedding=existing_emb.copy()),
            _make_candidate("c3", "text three", embedding=existing_emb.copy()),
        ]
        model = MagicMock()

        result = add_embeddings(candidates, model)

        model.embed_documents.assert_not_called()
        self.assertIs(result, candidates)  # same list returned in-place

    def test_existing_embeddings_not_mutated(self):
        original = _fake_embedding()
        candidates = [_make_candidate("c1", embedding=original.copy())]
        model = MagicMock()

        add_embeddings(candidates, model)

        np.testing.assert_array_equal(candidates[0].embedding, original)


class TestAddEmbeddingsAllMissing(unittest.TestCase):
    """All-missing case: embed_documents called once with all texts."""

    def test_all_texts_sent_to_api(self):
        candidates = [
            _make_candidate("c1", "alpha"),
            _make_candidate("c2", "beta"),
            _make_candidate("c3", "gamma"),
        ]
        model = _mock_embeddings_model(["alpha", "beta", "gamma"])

        add_embeddings(candidates, model)

        model.embed_documents.assert_called_once_with(["alpha", "beta", "gamma"])

    def test_embeddings_populated_on_all_candidates(self):
        candidates = [
            _make_candidate("c1", "alpha"),
            _make_candidate("c2", "beta"),
        ]
        model = _mock_embeddings_model(["alpha", "beta"])

        add_embeddings(candidates, model)

        for c in candidates:
            self.assertIsInstance(c.embedding, np.ndarray)
            self.assertEqual(c.embedding.dtype, np.float32)


class TestAddEmbeddingsPartialMissing(unittest.TestCase):
    """Partial-missing case: only missing candidates are re-embedded."""

    def setUp(self):
        self.pre_emb = _fake_embedding() * 99.0  # distinctive value
        self.candidates = [
            _make_candidate("c1", "has embedding", embedding=self.pre_emb.copy()),
            _make_candidate("c2", "needs embedding"),          # None
            _make_candidate("c3", "also has embedding", embedding=self.pre_emb.copy()),
            _make_candidate("c4", "also needs embedding"),     # None
        ]
        # API will be called with exactly the two missing texts.
        self.model = _mock_embeddings_model(["needs embedding", "also needs embedding"])

    def test_only_missing_texts_sent_to_api(self):
        add_embeddings(self.candidates, self.model)

        self.model.embed_documents.assert_called_once_with(
            ["needs embedding", "also needs embedding"]
        )

    def test_pre_existing_embeddings_unchanged(self):
        add_embeddings(self.candidates, self.model)

        np.testing.assert_array_equal(self.candidates[0].embedding, self.pre_emb)
        np.testing.assert_array_equal(self.candidates[2].embedding, self.pre_emb)

    def test_missing_candidates_now_have_embeddings(self):
        add_embeddings(self.candidates, self.model)

        self.assertIsInstance(self.candidates[1].embedding, np.ndarray)
        self.assertIsInstance(self.candidates[3].embedding, np.ndarray)

    def test_embed_documents_called_exactly_once(self):
        add_embeddings(self.candidates, self.model)

        self.assertEqual(self.model.embed_documents.call_count, 1)


class TestAddEmbeddingsEdgeCases(unittest.TestCase):
    """Empty list and single-candidate edge cases."""

    def test_empty_list_returns_immediately(self):
        model = MagicMock()
        result = add_embeddings([], model)
        model.embed_documents.assert_not_called()
        self.assertEqual(result, [])

    def test_single_candidate_with_embedding(self):
        c = _make_candidate("c1", embedding=_fake_embedding())
        model = MagicMock()
        add_embeddings([c], model)
        model.embed_documents.assert_not_called()

    def test_single_candidate_without_embedding(self):
        c = _make_candidate("c1", "solo text")
        model = _mock_embeddings_model(["solo text"])
        add_embeddings([c], model)
        model.embed_documents.assert_called_once_with(["solo text"])
        self.assertIsNotNone(c.embedding)


# ---------------------------------------------------------------------------
# Entry point for plain `python tests/test_add_embeddings.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
