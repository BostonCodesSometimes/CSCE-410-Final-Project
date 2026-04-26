import csv
import re
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from openai import OpenAI, RateLimitError
from pinecone import Pinecone
from sklearn.metrics.pairwise import cosine_similarity

from queries import all_queries

# ============================================================
# CONFIG
# ============================================================

PINECONE_API_KEY = "PASTE_YOUR_PINECONE_KEY_HERE"
OPENAI_API_KEY = "PASTE_YOUR_OPENAI_KEY_HERE"

INDEX_NAME = "lite-rag"
NAMESPACE = "__default__"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512

INITIAL_TOP_K = 20
FINAL_TOP_K = 5
MAX_ADAPTIVE_K = 10

# None means run all queries from queries.py
TEST_QUERY_LIMIT = None

OUTPUT_CSV = "adaptive_mmr_results_auto_all_queries.csv"

STOPWORDS = {
    "and", "or", "the", "a", "an", "for", "of", "to", "in", "on", "with",
    "by", "from", "at", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "major", "generalized", "disorder"
}

PRIORITY_TERMS = {
    "treatment", "therapy", "management", "maintenance",
    "intervention", "interventions", "clinical", "efficacy", "trial"
}

# ============================================================
# CLIENTS
# ============================================================

if PINECONE_API_KEY == "PASTE_YOUR_PINECONE_KEY_HERE":
    raise ValueError("Paste your Pinecone API key.")

if OPENAI_API_KEY == "PASTE_YOUR_OPENAI_KEY_HERE":
    raise ValueError("Paste your OpenAI API key.")

pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# EMBEDDING
# ============================================================

def embed_query(query: str) -> np.ndarray:
    try:
        print("Calling OpenAI embedding API...")
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        print("OpenAI embedding received.")
        return np.array(response.data[0].embedding, dtype=np.float32)

    except RateLimitError as e:
        raise RuntimeError("OpenAI quota/rate-limit issue.") from e

# ============================================================
# BASIC HELPERS
# ============================================================

def get_index_dimension(stats: Any) -> int:
    if isinstance(stats, dict) and "dimension" in stats:
        return stats["dimension"]
    if hasattr(stats, "dimension"):
        return stats.dimension
    raise ValueError("Could not determine Pinecone index dimension.")


def normalize_matches(raw_matches: List[Any]) -> List[Dict[str, Any]]:
    matches = []

    for match in raw_matches:
        if isinstance(match, dict):
            matches.append({
                "id": match.get("id"),
                "score": float(match.get("score", 0.0) or 0.0),
                "values": match.get("values", []) or [],
                "metadata": match.get("metadata", {}) or {},
            })
        else:
            matches.append({
                "id": getattr(match, "id", None),
                "score": float(getattr(match, "score", 0.0) or 0.0),
                "values": list(match.values) if getattr(match, "values", None) is not None else [],
                "metadata": getattr(match, "metadata", {}) or {},
            })

    return matches


def retrieve_from_pinecone(index, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=INITIAL_TOP_K,
        include_metadata=True,
        include_values=True,
        namespace=NAMESPACE,
    )

    raw_matches = results["matches"] if isinstance(results, dict) else results.matches
    return normalize_matches(raw_matches)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def preview_text(text: str, limit: int = 250) -> str:
    return (text or "")[:limit].replace("\n", " ").strip()


def extract_query_terms(query: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9\-]+", query.lower())
    return sorted({
        t for t in tokens
        if len(t) >= 4 and t not in STOPWORDS
    })

# ============================================================
# STRONGER KEYWORD / INTENT BONUS
# ============================================================

def compute_keyword_bonus(query: str, title: str, text: str) -> Tuple[float, str]:
    terms = extract_query_terms(query)

    title_l = clean_text(title)
    text_l = clean_text(text)

    bonus = 0.0
    matched = []

    for term in terms:
        in_title = term in title_l
        in_text = term in text_l

        if not in_title and not in_text:
            continue

        matched.append(term)

        if term in {"therapy", "treatment", "management", "maintenance"}:
            if in_title:
                bonus += 0.08
            elif in_text:
                bonus += 0.035

        elif term in {"intervention", "interventions", "efficacy", "clinical", "trial"}:
            if in_title:
                bonus += 0.05
            elif in_text:
                bonus += 0.02

        else:
            if in_title:
                bonus += 0.02
            elif in_text:
                bonus += 0.005

    return min(bonus, 0.15), ", ".join(matched)


def annotate_matches(query: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    annotated = []

    for match in matches:
        metadata = match.get("metadata", {})
        title = metadata.get("title", "") or ""
        text = metadata.get("text", "") or ""

        bonus, matched_terms = compute_keyword_bonus(query, title, text)

        enriched = dict(match)
        enriched["keyword_bonus"] = round(bonus, 6)
        enriched["matched_terms"] = matched_terms
        enriched["hybrid_score"] = float(match.get("score", 0.0)) + bonus

        annotated.append(enriched)

    return annotated

# ============================================================
# FULLY ADAPTIVE QUERY PROFILE
# ============================================================

def infer_query_profile(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [
        float(m.get("hybrid_score", m.get("score", 0.0)))
        for m in matches[:10]
    ]

    if len(scores) < 3:
        return {
            "profile": "small_pool",
            "adaptive_threshold": 0.02,
            "relevance_floor": 0.80,
            "lambda_param": 0.85,
            "min_final_pool": 3,
            "min_adaptive_k": 3,
        }

    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]

    largest_gap = max(gaps)
    score_span = scores[0] - scores[-1]
    top_gap = scores[0] - scores[1]

    if largest_gap < 0.018 and score_span < 0.10:
        return {
            "profile": "ambiguous_broad",
            "adaptive_threshold": 0.035,
            "relevance_floor": 0.80,
            "lambda_param": 0.75,
            "min_final_pool": 6,
            "min_adaptive_k": 5,
        }

    if top_gap > 0.045 or largest_gap > 0.055:
        return {
            "profile": "confident_narrow",
            "adaptive_threshold": 0.018,
            "relevance_floor": 0.88,
            "lambda_param": 0.88,
            "min_final_pool": 4,
            "min_adaptive_k": 3,
        }

    return {
        "profile": "balanced",
        "adaptive_threshold": 0.025,
        "relevance_floor": 0.84,
        "lambda_param": 0.82,
        "min_final_pool": 5,
        "min_adaptive_k": 4,
    }

# ============================================================
# SELECTION HELPERS
# ============================================================

def deduplicate_by_doc_id(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_doc = {}

    for match in matches:
        metadata = match.get("metadata", {})
        doc_id = metadata.get("doc_id") or match.get("id")

        current = best_by_doc.get(doc_id)

        if current is None:
            best_by_doc[doc_id] = match
        else:
            if float(match["hybrid_score"]) > float(current["hybrid_score"]):
                best_by_doc[doc_id] = match

    deduped = list(best_by_doc.values())
    deduped.sort(key=lambda x: float(x["hybrid_score"]), reverse=True)
    return deduped


def adaptive_selection(
    matches: List[Dict[str, Any]],
    threshold: float,
    min_k: int,
    max_k: int,
) -> List[Dict[str, Any]]:

    if not matches:
        return []

    limit = min(len(matches), max_k)
    selected = [matches[0]]

    for i in range(1, limit):
        prev_score = float(matches[i - 1]["hybrid_score"])
        curr_score = float(matches[i]["hybrid_score"])
        drop = prev_score - curr_score

        if drop > threshold and len(selected) >= min_k:
            break

        selected.append(matches[i])

    return selected


def apply_relevance_floor(
    matches: List[Dict[str, Any]],
    floor: float,
    min_final_pool: int,
) -> List[Dict[str, Any]]:

    if not matches:
        return []

    top_score = float(matches[0]["hybrid_score"])
    cutoff = top_score * floor

    kept = [
        m for m in matches
        if float(m["hybrid_score"]) >= cutoff
    ]

    if len(kept) < min_final_pool:
        kept = matches[:min(min_final_pool, len(matches))]

    return kept


def build_embedding_matrix(matches: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
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


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr

    min_v = float(np.min(arr))
    max_v = float(np.max(arr))

    if abs(max_v - min_v) < 1e-12:
        return np.ones_like(arr, dtype=np.float32)

    return ((arr - min_v) / (max_v - min_v)).astype(np.float32)

# ============================================================
# ADAPTIVE MMR
# ============================================================

def mmr_rerank(
    query_embedding: np.ndarray,
    matches: List[Dict[str, Any]],
    final_top_k: int,
    lambda_param: float,
) -> List[Dict[str, Any]]:

    valid_matches, chunk_embeddings = build_embedding_matrix(matches)

    if not valid_matches:
        return []

    query_sim = cosine_similarity(
        chunk_embeddings,
        query_embedding.reshape(1, -1)
    ).flatten()

    hybrid_scores = np.array(
        [float(m["hybrid_score"]) for m in valid_matches],
        dtype=np.float32
    )

    query_sim_norm = minmax_normalize(query_sim)
    hybrid_norm = minmax_normalize(hybrid_scores)

    blended_relevance = (0.70 * query_sim_norm) + (0.30 * hybrid_norm)

    selected = []
    candidates = list(range(len(valid_matches)))

    first_idx = int(np.argmax(blended_relevance))
    selected.append(first_idx)
    candidates.remove(first_idx)

    diagnostics = {
        first_idx: {
            "query_sim": float(query_sim[first_idx]),
            "blended_relevance": float(blended_relevance[first_idx]),
            "max_sim_to_selected": 0.0,
            "mmr_score": float(blended_relevance[first_idx]),
        }
    }

    while len(selected) < min(final_top_k, len(valid_matches)) and candidates:
        best_idx = None
        best_score = -np.inf
        best_sim = 0.0

        for idx in candidates:
            sim_to_selected = cosine_similarity(
                chunk_embeddings[idx].reshape(1, -1),
                chunk_embeddings[selected]
            ).max()

            mmr_score = (
                lambda_param * blended_relevance[idx]
                - (1 - lambda_param) * sim_to_selected
            )

            if mmr_score > best_score:
                best_score = float(mmr_score)
                best_idx = idx
                best_sim = float(sim_to_selected)

        selected.append(best_idx)
        candidates.remove(best_idx)

        diagnostics[best_idx] = {
            "query_sim": float(query_sim[best_idx]),
            "blended_relevance": float(blended_relevance[best_idx]),
            "max_sim_to_selected": best_sim,
            "mmr_score": best_score,
        }

    final = []

    for rank, idx in enumerate(selected, start=1):
        m = dict(valid_matches[idx])
        m["mmr_selected_rank"] = rank
        m["query_sim"] = round(diagnostics[idx]["query_sim"], 6)
        m["blended_relevance"] = round(diagnostics[idx]["blended_relevance"], 6)
        m["max_sim_to_selected"] = round(diagnostics[idx]["max_sim_to_selected"], 6)
        m["mmr_score"] = round(diagnostics[idx]["mmr_score"], 6)
        final.append(m)

    return final

# ============================================================
# OUTPUT HELPERS
# ============================================================

def print_results(stage_name: str, query: str, results: List[Dict[str, Any]], limit: int = 5) -> None:
    print(f"\n{'=' * 100}")
    print(stage_name)
    print(f"Query: {query}")
    print(f"{'=' * 100}")

    for rank, match in enumerate(results[:limit], start=1):
        metadata = match.get("metadata", {})

        print(f"\nRank #{rank}")
        print("Pinecone Score:", round(float(match.get("score", 0.0)), 6))
        print("Hybrid Score  :", round(float(match.get("hybrid_score", 0.0)), 6))
        print("Keyword Bonus :", round(float(match.get("keyword_bonus", 0.0)), 6))

        if "mmr_score" in match:
            print("Query Sim     :", match["query_sim"])
            print("MMR Score     :", match["mmr_score"])
            print("Doc Diversity :", match["max_sim_to_selected"])

        print("Doc ID        :", metadata.get("doc_id"))
        print("Title         :", metadata.get("title"))
        print("Journal       :", metadata.get("journal"))
        print("Year          :", metadata.get("year"))
        print("Chunk Number  :", metadata.get("chunk_number"))
        print("Matched Terms :", match.get("matched_terms", ""))
        print("Text Preview  :", preview_text(metadata.get("text", ""), 250))


def append_rows(
    rows: List[Dict[str, Any]],
    query: str,
    stage: str,
    results: List[Dict[str, Any]],
    runtime: float,
    profile: Dict[str, Any],
    initial_count: int,
    dedup_count: int,
    adaptive_count: int,
    filtered_count: int,
) -> None:

    for rank, match in enumerate(results, start=1):
        metadata = match.get("metadata", {})

        rows.append({
            "query": query,
            "stage": stage,
            "rank": rank,
            "profile": profile["profile"],
            "adaptive_threshold": profile["adaptive_threshold"],
            "relevance_floor": profile["relevance_floor"],
            "lambda_param": profile["lambda_param"],
            "min_final_pool": profile["min_final_pool"],
            "pinecone_score": round(float(match.get("score", 0.0)), 6),
            "hybrid_score": round(float(match.get("hybrid_score", 0.0)), 6),
            "keyword_bonus": round(float(match.get("keyword_bonus", 0.0)), 6),
            "matched_terms": match.get("matched_terms", ""),
            "query_sim": match.get("query_sim", ""),
            "blended_relevance": match.get("blended_relevance", ""),
            "max_sim_to_selected": match.get("max_sim_to_selected", ""),
            "mmr_score": match.get("mmr_score", ""),
            "doc_id": metadata.get("doc_id"),
            "title": metadata.get("title"),
            "journal": metadata.get("journal"),
            "year": metadata.get("year"),
            "chunk_number": metadata.get("chunk_number"),
            "text_preview": preview_text(metadata.get("text", ""), 500),
            "runtime_seconds": round(runtime, 4),
            "initial_count": initial_count,
            "dedup_count": dedup_count,
            "adaptive_count": adaptive_count,
            "filtered_count": filtered_count,
        })


def save_csv(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()
    print("\n--- INDEX STATS ---")
    print(stats)

    dimension = get_index_dimension(stats)
    print("\nIndex dimension:", dimension)

    if dimension != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Dimension mismatch: index={dimension}, expected={EMBEDDING_DIMENSIONS}"
        )

    output_rows = []

    limited_queries = all_queries if TEST_QUERY_LIMIT is None else all_queries[:TEST_QUERY_LIMIT]
    print(f"\nLoaded {len(limited_queries)} queries from queries.py")

    for q_num, query in enumerate(limited_queries, start=1):
        print(f"\n\nProcessing query {q_num}/{len(limited_queries)}")
        start_time = time.time()

        try:
            query_embedding = embed_query(query)
        except RuntimeError as e:
            print(f"Skipping query due to embedding error: {e}")
            continue

        initial_matches = retrieve_from_pinecone(index, query_embedding)
        baseline_matches = annotate_matches(query, initial_matches)

        scored_matches = annotate_matches(query, initial_matches)
        deduped_matches = deduplicate_by_doc_id(scored_matches)

        profile = infer_query_profile(deduped_matches)

        adaptive_matches = adaptive_selection(
            deduped_matches,
            threshold=profile["adaptive_threshold"],
            min_k=profile["min_adaptive_k"],
            max_k=MAX_ADAPTIVE_K,
        )

        filtered_matches = apply_relevance_floor(
            adaptive_matches,
            floor=profile["relevance_floor"],
            min_final_pool=profile["min_final_pool"],
        )

        final_matches = mmr_rerank(
            query_embedding=query_embedding,
            matches=filtered_matches,
            final_top_k=FINAL_TOP_K,
            lambda_param=profile["lambda_param"],
        )

        runtime = time.time() - start_time

        print("\nAUTO PROFILE SELECTED:")
        print(profile)

        print_results("BASELINE INITIAL RETRIEVAL", query, baseline_matches, FINAL_TOP_K)
        print_results("FINAL ADAPTIVE + MMR RESULTS", query, final_matches, FINAL_TOP_K)

        print(f"\nRuntime: {runtime:.4f} seconds")
        print(f"Initial retrieved        : {len(initial_matches)}")
        print(f"After deduplication      : {len(deduped_matches)}")
        print(f"After adaptive selection : {len(adaptive_matches)}")
        print(f"After relevance floor    : {len(filtered_matches)}")
        print(f"Final after MMR          : {len(final_matches)}")

        append_rows(
            rows=output_rows,
            query=query,
            stage="baseline_initial",
            results=baseline_matches[:FINAL_TOP_K],
            runtime=runtime,
            profile=profile,
            initial_count=len(initial_matches),
            dedup_count=len(deduped_matches),
            adaptive_count=len(adaptive_matches),
            filtered_count=len(filtered_matches),
        )

        append_rows(
            rows=output_rows,
            query=query,
            stage="adaptive_mmr_final",
            results=final_matches,
            runtime=runtime,
            profile=profile,
            initial_count=len(initial_matches),
            dedup_count=len(deduped_matches),
            adaptive_count=len(adaptive_matches),
            filtered_count=len(filtered_matches),
        )

        save_csv(output_rows)
        print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
