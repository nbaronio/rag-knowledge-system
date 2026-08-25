"""
Main orchestration: wires ingestion, indexing, retrieval, confidence, and
generation together, then runs the demo queries.
"""

from ingestion import load_documents
from indexing import build_all_chunks, build_embeddings_matrix
from retrieval import precompute_recency_scores, retrieve, MIN_SCORE_THRESHOLD
from generation import generate_answer

documents = load_documents("docs")
assert len(documents) == 6, f"Expected 6 documents, found {len(documents)}"

all_chunks = build_all_chunks(documents)
embeddings_matrix = build_embeddings_matrix(all_chunks)
recency_scores = precompute_recency_scores(all_chunks)

def answer_query(query, top_k=3, alpha=0.9, min_score_threshold=MIN_SCORE_THRESHOLD):
    results = retrieve(query, all_chunks, embeddings_matrix, recency_scores, top_k, alpha)

    # Filter on semantic relevance alone: weak chunks shouldn't enter the
    # context just because a recency bonus dragged their final score up.
    relevant_results = [r for r in results if r[1] > min_score_threshold]

    if relevant_results:
        return generate_answer(query, relevant_results)

    return {
        "answer": "No relevant sources found for this query",
        "key_points": [],
        "sources_used": [],
        "confidence": "low",
        "reasoning_confidence": "Retrieval score below threshold"
    }

if __name__ == "__main__":
    # Test 1: Recency boost - the system must prefer the newer document
    result_1 = answer_query("Users that need access to customer data need to do what?")
    print(result_1)

    # Test 2: Discrimination within the same domain
    result_2 = answer_query("Which team do need to check consents before sending any communications?")
    print(result_2)

    # Test 3: Discrimination across distant domains
    result_3 = answer_query("How do we onboard a new colleague?")
    print(result_3)

    # Test 4: False lexical similarity
    result_4 = answer_query("What is the procedure in case some operative incident happens?")
    print(result_4)

    # Test 5: Out of domain / minimum threshold
    result_5 = answer_query("What is today's weather?")
    print(result_5)
