"""
Hybrid retrieval: combines semantic similarity with a recency signal.
"""

import datetime
from config import model

def precompute_recency_scores(all_chunks):
    # Parsed once here and reused below, instead of re-parsing the same
    # ISO string a second time per chunk.
    dates = [datetime.date.fromisoformat(chunk.date) for chunk in all_chunks]
    min_date, max_date = min(dates), max(dates)
    span_days = (max_date - min_date).days

    recency_scores = []
    for chunk_date in dates:
        if span_days == 0:
            # All documents share the same date: recency carries no signal,
            # avoid division by zero, use a neutral score.
            recency_scores.append(0.5)
        else:
            recency_scores.append((chunk_date - min_date).days / span_days)
    return recency_scores

def retrieve(query, all_chunks, embeddings_matrix, recency_scores, top_k=3, alpha=0.9):
    query_vector = model.encode(query, normalize_embeddings=True)
    semantic_scores = embeddings_matrix @ query_vector

    results = []
    for i in range(len(all_chunks)):
        semantic_score = float(semantic_scores[i])
        final_score = alpha * semantic_score + (1 - alpha) * recency_scores[i]
        # semantic_score and final_score are kept separate: the former drives
        # threshold and confidence, the latter only the final ranking.
        results.append((all_chunks[i], semantic_score, final_score))

    return sorted(results, key=lambda r: r[2], reverse=True)[:top_k]

# --- Empirical calibration of MIN_SCORE_THRESHOLD (performed once during development) ---
# Applied to semantic_score alone, not final_score: final_score is blended
# with the recency bonus and would inflate weak matches (see main.py).
# retrieve("What is today's weather?")[0][1]                -> semantic_score ~0.02  (out-of-domain)
# retrieve("Best pizza recipe")[0][1]                        -> semantic_score ~0.01  (out-of-domain)
# retrieve("...operative incident...")[0][1]                 -> semantic_score ~0.44  (weakest genuine on-domain top match)
#
# There is a wide gap between out-of-domain noise (~0.02) and the weakest
# genuine on-domain top match (~0.44). MIN_SCORE_THRESHOLD is set to 0.15:
# well above the noise floor, well below genuine matches.
MIN_SCORE_THRESHOLD = 0.15