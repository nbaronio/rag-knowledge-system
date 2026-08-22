"""
Confidence scoring: derives a baseline confidence label from the top
retrieval score. This is a ceiling passed to the LLM, which can only lower
it, never raise it (see generation.py / SYSTEM_INSTRUCTIONS).
"""

def score_to_confidence(score):
    if score > 0.7:
        confidence = "high"
    elif 0.5 < score <= 0.7:
        confidence = "medium"
    else:
        confidence = "low"
    return confidence
