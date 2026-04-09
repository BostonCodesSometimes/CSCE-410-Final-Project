from sklearn.metrics.pairwise import cosine_similarity

"""
1. Pick the most relevant chunk first
2. For each remaining chunk, score it with:
   - MMR = λ * Relevance - (1-λ) * Similarity
3. Pick the best one
4. Repeat until you hit your limit
"""

"The functions below were simply copy-pasted from the following article: "
"https://medium.com/tech-that-works/maximal-marginal-relevance-to-rerank-results-in-unsupervised-keyphrase-extraction-22d95015c7c5"

def maximal_marginal_relevance(sentence_vector, phrases, embedding_matrix, lambda_constant=0.5, threshold_terms=10):
    """
    Return ranked phrases using MMR. Cosine similarity is used as similarity measure.
    :param sentence_vector: Query vector
    :param phrases: list of candidate phrases
    :param embedding_matrix: matrix having index as phrases and values as vector
    :param lambda_constant: 0.5 to balance diversity and accuracy. if lambda_constant is high, then higher accuracy. If lambda_constant is low then high diversity.
    :param threshold_terms: number of terms to include in result set
    :return: Ranked phrases with score
    """
    # todo: Use cosine similarity matrix for lookup among phrases instead of making call everytime.
    s = []
    r = sorted(phrases, key=lambda x: x[1], reverse=True)
    r = [i[0] for i in r]
    while len(r) > 0:
        score = 0
        phrase_to_add = ''
        for i in r:
            first_part = cosine_similarity([sentence_vector], [embedding_matrix.loc[i]])[0][0]
            second_part = 0
            for j in s:
                cos_sim = cosine_similarity([embedding_matrix.loc[i]], [embedding_matrix.loc[j[0]]])[0][0]
                if cos_sim > second_part:
                    second_part = cos_sim
            equation_score = lambda_constant*(first_part)-(1-lambda_constant) * second_part
            if equation_score > score:
                score = equation_score
                phrase_to_add = i
        if phrase_to_add == '':
            phrase_to_add = i
        r.remove(phrase_to_add)
        s.append((phrase_to_add, score))
    return (s, s[:threshold_terms])[threshold_terms > len(s)]

def club_similar_keywords(emb_mat, sim_score=0.9):
    """
    :param emb_mat: matrix having vectors with words as index
    :param sim_score: 0.9 by default
    :return: returns list of unique words from index after combining words which has similarity score of more than
    0.9
    """
    if len(emb_mat) == 0:
        return 'NA'
    xx = cosine_similarity(emb_mat)
    final_keywords = set(emb_mat.index)
    N = len(emb_mat.index)
    dd = {}
    for i in range(N):
        for j in range(N):
            if (float(xx[i][j]) > sim_score) and (i != j):
                try:
                    dd[emb_mat.index[i]].append(emb_mat.index[j])
                except:
                    dd[emb_mat.index[i]] = []
                    dd[emb_mat.index[i]].append(emb_mat.index[j])
    removed_keywords = []
    for key in dd:
        for val in dd[key]:
            if key not in removed_keywords:
                removed_keywords += dd[key]
                try:
                    final_keywords.remove(val)
                except:
                    pass
    return final_keywords