"""
Prompt construction, LLM call, and structured JSON parsing.
"""

import json
from config import client, GENERATION_MODEL_NAME
from confidence import score_to_confidence

SYSTEM_INSTRUCTIONS = """
You are a virtual assistant that answers business-related questions.
You are provided with a set of documents that represent your entire knowledge base. You must not use any knowledge that is not contained in these documents.
If the question is not relevant to the content of the documents, you must state this explicitly, for example with the phrase "I don't have information on the requested topic."
Always cite the source you are using to answer, using the document id as the identifier.
The output must be in JSON format, following the exact schema provided below. You must not change field names or take any liberties in interpreting this template.
JSON structure: {"answer", "key_points", "sources_used", "confidence", "reasoning_confidence"}
The "key_points" field is a list of short strings summarizing the main takeaways of the answer.
The "sources_used" field is composed of the following fields: "sources_used": [{"id": "...", "title": "...", "date": "..."}]
You will receive a retrieval score based on how well the answer matches the requested information. If you believe it is necessary, you may lower this score, but you must NEVER raise it.
Do not wrap the JSON in markdown code blocks or backticks. Return raw JSON only.
"""

def build_context(top_results):
    formatted_lines = ""
    for chunk, semantic_score, final_score in top_results:
        formatted_lines += f"[Source: {chunk.parent_doc_id} | {chunk.title} | {chunk.date}]\n{chunk.text}\n\n"
    return formatted_lines

def build_prompt(query, top_results):
    context = build_context(top_results)
    # Confidence is based on semantic relevance alone: the recency bonus
    # should not make an answer look more trustworthy than the content justifies.
    confidence = score_to_confidence(top_results[0][1])
    return f"""
    SYSTEM INSTRUCTIONS:
    {SYSTEM_INSTRUCTIONS}

    CONTEXT:
    {context}

    QUESTION:
    {query}

    RETRIEVAL CONFIDENCE:
    {confidence}
    """

def clean_json_text(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()

def generate_answer(query, top_results):
    prompt = build_prompt(query, top_results)

    try:
        response = client.models.generate_content(model=GENERATION_MODEL_NAME, contents=prompt)
    except Exception as e:
        # A timeout, rate limit, or 500 error should not crash the whole script
        return {"error": "Generation call failed", "detail": str(e)}

    try:
        parsed = json.loads(clean_json_text(response.text))
    except json.JSONDecodeError:
        parsed = {"error": "JSON parsing failed", "raw_output": response.text}

    return parsed
