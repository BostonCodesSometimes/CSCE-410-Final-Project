"""Build eval_data.json automatically from the Pinecone corpus.

Approach (synthetic eval set):
  1. List all vector IDs from the Pinecone index.
  2. Fetch each vector's metadata and chunk text (no embeddings needed).
  3. Group chunks by PMID to reconstruct each paper's full abstract text.
  4. Sample N papers at random (default 20).
  5. For each paper, call GPT-4o-mini to generate:
       - one clinical question the abstract can answer
       - a 2-3 sentence reference answer
       - a question_type label
       - 3-5 keywords
  6. Write the result to eval_data.json.

Tradeoff to disclose in your writeup:
  Synthetic eval sets guarantee that gold PMIDs are present in the corpus,
  which inflates absolute recall numbers.  However, since all three methods
  (similarity, MMR, LitePack) are evaluated on the same set, METHOD
  COMPARISONS remain valid and fair.  Report this as a synthetic dataset.

Usage:
  python build_eval_set.py [--n 20] [--output eval_data.json] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time

import openai
from dotenv import load_dotenv
from pinecone import Pinecone

from pinecone_settings import get_pinecone_config

load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Pinecone fetch helpers                                                       #
# --------------------------------------------------------------------------- #

def _list_all_ids(index) -> list[str]:
    """Return all vector IDs in the index using the list() iterator."""
    ids: list[str] = []
    try:
        for batch in index.list():
            if isinstance(batch, list):
                ids.extend(batch)
            elif hasattr(batch, "__iter__"):
                ids.extend(list(batch))
        logger.info("list_all_ids: found %d vector IDs", len(ids))
    except Exception as exc:
        logger.error("list_all_ids failed: %s", exc)
        raise
    return ids


def _fetch_in_batches(index, ids: list[str], batch_size: int = 100) -> list:
    """Fetch vector objects in batches; returns flat list of Vector objects."""
    all_vectors: list = []
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        try:
            result = index.fetch(ids=batch)
            # result.vectors is a dict of {id: Vector}
            all_vectors.extend(result.vectors.values())
        except Exception as exc:
            logger.warning("fetch batch %d-%d failed: %s", i, i + batch_size, exc)
        time.sleep(0.05)  # stay well under rate limits
    logger.info("fetch_in_batches: fetched %d vector objects", len(all_vectors))
    return all_vectors


def _group_by_pmid(vectors: list) -> dict[str, dict]:
    """Group fetched Vector objects by doc_id (PMID).

    Returns a dict: pmid -> {"title", "year", "journal", "url", "chunks": [...]}
    Each Vector object has a .metadata attribute (dict).
    """
    papers: dict[str, dict] = {}
    for vec in vectors:
        meta = vec.metadata or {}
        pmid = str(meta.get("doc_id") or "").strip()
        if not pmid:
            continue

        if pmid not in papers:
            papers[pmid] = {
                "pmid": pmid,
                "title": meta.get("title", "Unknown Title"),
                "year": meta.get("year", "N/A"),
                "journal": meta.get("journal", "Unknown"),
                "url": meta.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                "chunks": [],
            }

        chunk_text = meta.get("text") or meta.get("page_content") or meta.get("chunk_text") or ""
        chunk_num = int(meta.get("chunk_number", 0))
        papers[pmid]["chunks"].append((chunk_num, chunk_text))

    # Sort chunks by chunk_number so the reconstructed text reads in order
    for paper in papers.values():
        paper["chunks"].sort(key=lambda x: x[0])
        paper["abstract_text"] = " ".join(t for _, t in paper["chunks"] if t)

    return papers


# --------------------------------------------------------------------------- #
# LLM question generation                                                      #
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = (
    "You are a medical education expert creating evaluation questions for a "
    "retrieval-augmented generation (RAG) system. "
    "Your questions will be used to test whether the system can retrieve and "
    "answer clinical questions from PubMed abstracts."
)

_USER_TEMPLATE = """\
Below is the text of a PubMed abstract (PMID: {pmid}, Title: {title}).

--- ABSTRACT ---
{abstract}
--- END ABSTRACT ---

Generate a JSON object with exactly these fields:
{{
  "question": "<one focused clinical question whose complete answer is in this abstract>",
  "reference_answer": "<a 2-4 sentence answer drawn only from the abstract above>",
  "question_type": "<one of: treatment | diagnosis | mechanism | prognosis | epidemiology | other>",
  "keywords": ["<3 to 5 key medical terms from the abstract>"]
}}

Rules:
- The question must be answerable from this abstract alone.
- Do NOT invent information not in the abstract.
- The question should be specific enough that a general web search would not trivially answer it.
- Output valid JSON only, no extra text.
"""


def _generate_qa(
    client: openai.OpenAI,
    pmid: str,
    title: str,
    abstract: str,
    model: str = "gpt-4o-mini",
) -> dict | None:
    """Call the LLM to generate one QA entry for a paper.

    Returns a dict with keys: question, reference_answer, question_type, keywords
    or None on failure.
    """
    prompt = _USER_TEMPLATE.format(
        pmid=pmid,
        title=title,
        abstract=abstract[:3000],  # stay well within context limits
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        qa = json.loads(raw)

        # Validate required keys
        if not qa.get("question") or not qa.get("reference_answer"):
            logger.warning("LLM response missing required keys for PMID %s", pmid)
            return None

        return {
            "question": qa["question"].strip(),
            "gold_pmids": [pmid],
            "reference_answer": qa["reference_answer"].strip(),
            "question_type": qa.get("question_type", "other"),
            "keywords": qa.get("keywords", []),
        }
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error for PMID %s: %s", pmid, exc)
        return None
    except Exception as exc:
        logger.warning("LLM call failed for PMID %s: %s", pmid, exc)
        return None


# --------------------------------------------------------------------------- #
# Main build function                                                          #
# --------------------------------------------------------------------------- #

def build_eval_set(
    n: int,
    output_path: str,
    seed: int,
    model: str,
) -> None:
    """Full pipeline: Pinecone -> group by PMID -> generate QA -> write JSON."""

    # --- Connect to Pinecone ---
    api_key, index_name, host, _ = get_pinecone_config()
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name, host=host) if host else pc.Index(index_name)
    print(f"Connected to Pinecone index: {index_name!r}")

    # --- List all vector IDs ---
    print("Listing all vector IDs from index...")
    all_ids = _list_all_ids(index)
    if not all_ids:
        raise RuntimeError("No vectors found in index. Has data been loaded?")
    print(f"Found {len(all_ids)} total vectors")

    # --- Fetch metadata ---
    print("Fetching vector metadata (no embeddings)...")
    vectors = _fetch_in_batches(index, all_ids)

    # --- Group by PMID ---
    papers = _group_by_pmid(vectors)
    unique_pmids = sorted(papers.keys())
    print(f"Found {len(unique_pmids)} unique papers (PMIDs)")

    # --- Sample N papers ---
    rng = random.Random(seed)
    sample_pmids = rng.sample(unique_pmids, min(n, len(unique_pmids)))
    print(f"Sampling {len(sample_pmids)} papers for question generation")

    # --- Generate QA pairs ---
    client = openai.OpenAI()
    entries: list[dict] = []

    for i, pmid in enumerate(sample_pmids):
        paper = papers[pmid]
        abstract = paper["abstract_text"].strip()
        title = paper["title"]

        if not abstract:
            print(f"  [{i+1}/{len(sample_pmids)}] PMID {pmid}: no abstract text, skipping")
            continue

        print(f"  [{i+1}/{len(sample_pmids)}] PMID {pmid}: {title[:60]!r}")
        qa = _generate_qa(client, pmid, title, abstract, model=model)

        if qa:
            entries.append(qa)
            print(f"    -> Q: {qa['question'][:70]!r}")
        else:
            print(f"    -> FAILED (skipped)")

        time.sleep(0.3)  # light rate-limit buffer

    if not entries:
        raise RuntimeError("No QA entries were generated. Check logs for errors.")

    # --- Write output ---
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(entries)} entries to {output_path!r}")
    print("\nSample entry:")
    print(json.dumps(entries[0], indent=2))
    print(
        "\nNOTE: This is a synthetic eval set.  gold_pmids are guaranteed to be "
        "in the corpus, which inflates absolute recall.  Method comparisons "
        "(similarity vs MMR vs LitePack) remain fair and valid."
    )


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        prog="build_eval_set",
        description="Auto-generate eval_data.json from the Pinecone corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n", type=int, default=20,
                   help="Number of papers to sample.")
    p.add_argument("--output", default="eval_data.json",
                   help="Output JSON file path.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible sampling.")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="OpenAI model for question generation.")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   dest="log_level")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    build_eval_set(
        n=args.n,
        output_path=args.output,
        seed=args.seed,
        model=args.model,
    )


if __name__ == "__main__":
    main()
