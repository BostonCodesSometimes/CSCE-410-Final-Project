import csv
import time
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone
from openai import OpenAI

from queries import all_queries


# ============================================================
# 1) CONFIG
# ============================================================
PINECONE_API_KEY = "pcsk_6bpWTg_GyLo5YRcS2kNHnSGkEVkPFiR3bSvJiuVP9T2nDb5NK4J7v3FdJvSTgPtjAm9Xz5"
OPENAI_API_KEY = "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"

INDEX_NAME = "lite-rag"
NAMESPACE = "__default__"

# Embedding model confirmed for this project
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512

# Retrieval settings
INITIAL_TOP_K = 15
MIN_ADAPTIVE_K = 3
MAX_ADAPTIVE_K = 10
SCORE_DROP_THRESHOLD = 0.01
FINAL_TOP_K = 5
LAMBDA_PARAM = 0.7

OUTPUT_CSV = "adaptive_mmr_results.csv"


# ============================================================
# 2) VALIDATION + CLIENTS
# ============================================================
def validate_keys() -> None:
    if not PINECONE_API_KEY or PINECONE_API_KEY == "PASTE_YOUR_PINECONE_API_KEY_HERE":
        raise ValueError("Please paste your real Pinecone API key in PINECONE_API_KEY.")
    if not OPENAI_API_KEY or OPENAI_API_KEY == "PASTE_YOUR_OPENAI_API_KEY_HERE":
        raise ValueError("Please paste your real OpenAI API key in OPENAI_API_KEY.")


validate_keys()

pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# 3) QUERY EMBEDDING
# ============================================================
def embed_query(query: str) -> np.ndarray:
    """
    Convert query text into a 512-dimensional embedding so it matches
    the Pinecone index dimension.
    """
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


# ============================================================
# 4) HELPERS
# ============================================================
def get_index_dimension(stats: Any) -> int:
    """
    Pinecone describe_index_stats() can come back in slightly different shapes.
    """
    if isinstance(stats, dict):
        if "dimension" in stats:
            return stats["dimension"]
    if hasattr(stats, "dimension"):
        return stats.dimension
    raise ValueError("Could not determine Pinecone index dimension from index stats.")


def normalize_matches(raw_matches: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert Pinecone matches into plain Python dictionaries.
    """
    matches = []

    for match in raw_matches:
        if isinstance(match, dict):
            matches.append({
                "id": match.get("id"),
                "score": match.get("score", 0.0),
                "values": match.get("values", []) or [],
                "metadata": match.get("metadata", {}) or {},
            })
        else:
            matches.append({
                "id": getattr(match, "id", None),
                "score": getattr(match, "score", 0.0),
                "values": list(match.values) if getattr(match, "values", None) is not None else [],
                "metadata": getattr(match, "metadata", {}) or {},
            })

    return matches


def retrieve_from_pinecone(
    index,
    query_embedding: np.ndarray,
    top_k: int = 15,
    namespace: str = "__default__",
) -> List[Dict[str, Any]]:
    """
    Retrieve candidates from Pinecone.
    include_values=True is necessary for MMR since we need chunk vectors.
    """
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        include_values=True,
        namespace=namespace,
    )

    raw_matches = results["matches"] if isinstance(results, dict) else results.matches
    return normalize_matches(raw_matches)


def deduplicate_by_doc_id(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only the first chunk per document, reducing repeated chunks
    from the same article before adaptive selection and MMR.
    """
    seen = set()
    deduped = []

    for match in matches:
        metadata = match.get("metadata", {})
        doc_id = metadata.get("doc_id") or match.get("id")

        if doc_id not in seen:
            deduped.append(match)
            seen.add(doc_id)

    return deduped


def adaptive_selection(
    matches: List[Dict[str, Any]],
    min_k: int = 3,
    max_k: int = 10,
    threshold: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Adaptive candidate selection using score decay.

    Logic:
    - Always keep the first result.
    - Keep adding results until the score drop becomes too large,
      but only after reaching min_k.
    - Never exceed max_k.
    """
    if not matches:
        return []

    limit = min(len(matches), max_k)
    selected = [matches[0]]

    for i in range(1, limit):
        prev_score = float(matches[i - 1].get("score", 0.0))
        curr_score = float(matches[i].get("score", 0.0))
        drop = prev_score - curr_score

        if drop > threshold and len(selected) >= min_k:
            break

        selected.append(matches[i])

    return selected


def build_embedding_matrix(matches: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Build a valid embedding matrix from candidate matches.
    Drops any match whose vector is missing or wrong-sized.
    """
    valid_matches = []
    vectors = []

    for match in matches:
        values = match.get("values", [])
        if not values:
            continue

        vec = np.array(values, dtype=np.float32)
        if vec.shape[0] != EMBEDDING_DIMENSIONS:
            continue

        valid_matches.append(match)
        vectors.append(vec)

    if not vectors:
        return [], np.array([], dtype=np.float32)

    return valid_matches, np.vstack(vectors)


def mmr_rerank(
    query_embedding: np.ndarray,
    matches: List[Dict[str, Any]],
    final_top_k: int = 5,
    lambda_param: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    MMR:
        score(d_i) = lambda * relevance(query, d_i)
                     - (1-lambda) * max_similarity(d_i, selected_docs)

    Higher lambda  -> more relevance
    Lower lambda   -> more diversity
    """
    if not matches:
        return []

    valid_matches, chunk_embeddings = build_embedding_matrix(matches)
    if len(valid_matches) == 0:
        return []

    query_sim = cosine_similarity(
        chunk_embeddings,
        query_embedding.reshape(1, -1)
    ).flatten()

    selected_indices = []
    candidate_indices = list(range(len(valid_matches)))

    # Start with most relevant chunk
    first_idx = int(np.argmax(query_sim))
    selected_indices.append(first_idx)
    candidate_indices.remove(first_idx)

    # Select remaining chunks using MMR
    while len(selected_indices) < min(final_top_k, len(valid_matches)) and candidate_indices:
        best_score = -np.inf
        best_idx = None

        for idx in candidate_indices:
            relevance = query_sim[idx]

            sim_to_selected = cosine_similarity(
                chunk_embeddings[idx].reshape(1, -1),
                chunk_embeddings[selected_indices]
            ).max()

            mmr_score = (lambda_param * relevance) - ((1.0 - lambda_param) * sim_to_selected)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    return [valid_matches[i] for i in selected_indices]


def preview_text(text: str, limit: int = 250) -> str:
    return (text or "")[:limit].replace("\n", " ").strip()


def print_results(stage_name: str, query: str, results: List[Dict[str, Any]], limit: int = 5) -> None:
    print(f"\n{'=' * 90}")
    print(stage_name)
    print(f"Query: {query}")
    print(f"{'=' * 90}")

    if not results:
        print("No results found.")
        return

    for rank, match in enumerate(results[:limit], start=1):
        metadata = match.get("metadata", {})
        print(f"\nRank #{rank}")
        print("Pinecone Score:", match.get("score"))
        print("Doc ID:", metadata.get("doc_id"))
        print("Title:", metadata.get("title"))
        print("Journal:", metadata.get("journal"))
        print("Year:", metadata.get("year"))
        print("Chunk Number:", metadata.get("chunk_number"))
        print("Text Preview:", preview_text(metadata.get("text", ""), 250))


def append_stage_rows(
    output_rows: List[Dict[str, Any]],
    query: str,
    stage: str,
    results: List[Dict[str, Any]],
    runtime: float,
    initial_count: int,
    dedup_count: int,
    adaptive_count: int,
) -> None:
    for rank, match in enumerate(results, start=1):
        metadata = match.get("metadata", {})
        output_rows.append({
            "query": query,
            "stage": stage,
            "rank": rank,
            "pinecone_score": match.get("score"),
            "doc_id": metadata.get("doc_id"),
            "title": metadata.get("title"),
            "journal": metadata.get("journal"),
            "year": metadata.get("year"),
            "chunk_number": metadata.get("chunk_number"),
            "url": metadata.get("url"),
            "text_preview": preview_text(metadata.get("text", ""), 500),
            "runtime_seconds": round(runtime, 4),
            "initial_count": initial_count,
            "dedup_count": dedup_count,
            "adaptive_count": adaptive_count,
        })


def save_results_to_csv(rows: List[Dict[str, Any]], output_csv: str) -> None:
    fieldnames = [
        "query",
        "stage",
        "rank",
        "pinecone_score",
        "doc_id",
        "title",
        "journal",
        "year",
        "chunk_number",
        "url",
        "text_preview",
        "runtime_seconds",
        "initial_count",
        "dedup_count",
        "adaptive_count",
    ]

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 5) MAIN PIPELINE
# ============================================================
def main() -> None:
    index = pc.Index(INDEX_NAME)

    # Check Pinecone index stats
    stats = index.describe_index_stats()
    print("\n--- INDEX STATS ---")
    print(stats)

    dimension = get_index_dimension(stats)
    print("\nIndex dimension:", dimension)

    if dimension != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Dimension mismatch: Pinecone index is {dimension}, "
            f"but query embedding is set to {EMBEDDING_DIMENSIONS}."
        )

    output_rows = []

    print(f"\nLoaded {len(all_queries)} queries from queries.py")

    for q_num, query in enumerate(all_queries, start=1):
        print(f"\n\nProcessing query {q_num}/{len(all_queries)}")
        start_time = time.time()

        # Step A: Embed query
        query_embedding = embed_query(query)

        # Step B: Initial retrieval from Pinecone
        initial_matches = retrieve_from_pinecone(
            index=index,
            query_embedding=query_embedding,
            top_k=INITIAL_TOP_K,
            namespace=NAMESPACE,
        )

        # Step C: Deduplicate by document
        deduped_matches = deduplicate_by_doc_id(initial_matches)

        # Step D: Adaptive selection
        adaptive_matches = adaptive_selection(
            deduped_matches,
            min_k=MIN_ADAPTIVE_K,
            max_k=MAX_ADAPTIVE_K,
            threshold=SCORE_DROP_THRESHOLD,
        )

        # Step E: MMR reranking
        final_matches = mmr_rerank(
            query_embedding=query_embedding,
            matches=adaptive_matches,
            final_top_k=FINAL_TOP_K,
            lambda_param=LAMBDA_PARAM,
        )

        runtime = time.time() - start_time

        # Print baseline + final for visibility
        print_results("BASELINE INITIAL RETRIEVAL", query, initial_matches, limit=FINAL_TOP_K)
        print_results("FINAL ADAPTIVE + MMR RESULTS", query, final_matches, limit=FINAL_TOP_K)

        print(f"\nRuntime: {runtime:.4f} seconds")
        print(f"Initial retrieved: {len(initial_matches)}")
        print(f"After deduplication: {len(deduped_matches)}")
        print(f"After adaptive selection: {len(adaptive_matches)}")
        print(f"Final after MMR: {len(final_matches)}")

        # Save both baseline and final stage for comparison
        append_stage_rows(
            output_rows=output_rows,
            query=query,
            stage="baseline_initial",
            results=initial_matches[:FINAL_TOP_K],
            runtime=runtime,
            initial_count=len(initial_matches),
            dedup_count=len(deduped_matches),
            adaptive_count=len(adaptive_matches),
        )

        append_stage_rows(
            output_rows=output_rows,
            query=query,
            stage="adaptive_mmr_final",
            results=final_matches,
            runtime=runtime,
            initial_count=len(initial_matches),
            dedup_count=len(deduped_matches),
            adaptive_count=len(adaptive_matches),
        )

    save_results_to_csv(output_rows, OUTPUT_CSV)
    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()