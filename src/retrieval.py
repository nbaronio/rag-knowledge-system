"""
Hybrid retrieval: combines semantic similarity with a recency signal.
"""

from datetime import date
from config import model

def precompute_date_bounds(all_chunks):
    dates = [date.fromisoformat(chunk.date) for chunk in all_chunks]
    return min(dates), max(dates)

def retrieve(query, all_chunks, embeddings_matrix, min_date, max_date, top_k=3, alpha=0.75):
    query_vector = model.encode(query, normalize_embeddings=True)
    semantic_scores = embeddings_matrix @ query_vector

    normalized_recency_list = []
    for chunk in all_chunks:
        real_date = date.fromisoformat(chunk.date)
        normalized_recency = (real_date - min_date) / (max_date - min_date)
        normalized_recency_list.append(normalized_recency)

    final_scores = []
    for i in range(len(all_chunks)):
        final_score = alpha * semantic_scores[i] + (1 - alpha) * normalized_recency_list[i]
        final_scores.append((all_chunks[i], final_score))

    final_scores_sorted = sorted(final_scores, key=lambda p: p[1], reverse=True)
    return final_scores_sorted[:top_k]

# --- Empirical calibration of min_score_threshold (performed once during development) ---
# retrieve("What's the weather today?")[0][1]                    # 0.2652609
# retrieve("Best pizza recipe")[0][1]                             # 0.2434846
# retrieve("data access procedure for customer data")[0][1]       # 0.66481614
#
# There is a substantial gap between out-of-domain and on-domain scores.
# Based on these empirically observed values, min_score_threshold is set to 0.4.
MIN_SCORE_THRESHOLD = 0.4
