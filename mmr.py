import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mmr(query_embedding, chunk_embeddings, k=5, lambda_param=0.5):
    """
    Selects the top k that are the most relevant and unique

    param query_embedding: vector representation of the query
    param chunk_embeddings: matrix of vector representations for each chunk
        shape: (number of retrieved chunks, embedding dimension)
    param k: number of chunks to select
    param lambda_param: balance between relevance and diversity (0 to 1)
    """

    # compute relevance of each chunk to the query
    # each value in query_sim is a decimal (essentially a percentage) measure of how relevant the chunk is to the query
    query_sim = cosine_similarity(
        chunk_embeddings, query_embedding.reshape(1, -1)
    ).flatten()

    selected = [] # indices of selected chunks
    candidates = list(range(len(chunk_embeddings))) # indices of all chunks to consider

    # to start off, pick most relevant chunk first
    first = np.argmax(query_sim)
    selected.append(first)
    candidates.remove(first)

    # keep selecting until we have k chunks or run out of candidates
    while len(selected) < k and candidates:
        best_score = -np.inf    # best score so far
        best_idx = None         # index of chunk with best score

        for idx in candidates: # iterate through remaining candidates
            relevance = query_sim[idx] # relevance of this chunk to the query

            # compute similarity to already selected chunks
            sim_to_selected = cosine_similarity(
                chunk_embeddings[idx].reshape(1, -1),   # current candidate chunk embedding
                chunk_embeddings[selected]              # each selected chunk's embedding
            ).max() # penalty for the most redundant chunk

            # compute MMR score: balance relevance and diversity
            score = (
                lambda_param * relevance                # reward for relevance
                - (1 - lambda_param) * sim_to_selected  # penalty for redundancy
            )

            # keep the best scoring candidate
            if score > best_score:
                best_score = score
                best_idx = idx

        # add the best candidate to selected and remove from candidates
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected