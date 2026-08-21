# Step 1. Docstring
"""
rag-knowledge-system - DataPulse Spa
LLM Project - ProfAI

Answers natural language questions using a fixed corpus of 6 hardcoded
company documents (policies, guides, FAQs), returning a structured JSON
response with cited sources and a confidence indicator.

REQUIREMENTS
-------------
1. pip install sentence-transformers google-genai python-dotenv
2. Create a .env file in the same folder with GEMINI_API_KEY

ARCHITECTURAL CHOICES
----------------------
- Chunking: since these are FAQ-like documents, fixed-size word-based
  chunking was chosen, with configurable max_words/overlap (a guard
  prevents overlap >= max_words, which would otherwise cause an infinite
  loop). In this corpus every document produces a single chunk (all under
  threshold), but the function is generic and also handles longer documents.

- Embedding: an enriched text (title + category + chunk text) is embedded
  instead of the raw content alone, since titles and categories are short
  and informative in this corpus, improving semantic discrimination
  between similar documents.

- Indexing: no external vector index (FAISS/Chroma) is used. With about
  6 chunks, exhaustive cosine similarity computation via numpy is already
  instantaneous; a dedicated index would add dependencies and complexity
  with no practical benefit at this corpus scale.

- Hybrid retrieval: final_score = alpha * semantic_score + (1 - alpha) * recency,
  with alpha = 0.75. Semantic similarity dominates; recency acts as a
  tie-breaker (not an absolute filter) — this is the mechanism that
  correctly ranks the most up-to-date version of the same policy
  (doc_1 vs doc_2) by recency when both are semantically similar.

- Retrieval threshold (min_score_threshold = 0.4): calibrated empirically
  by testing out-of-domain queries (max score observed 0.25-0.27) against
  relevant queries (min score observed 0.5-0.66). Below this threshold,
  the system skips the LLM call entirely and returns a fixed
  "no relevant sources" response.

- Confidence: hybrid approach. A baseline confidence (high/medium/low) is
  computed from the top retrieval score (>0.7 / 0.5-0.7 / <0.5) and passed
  to the LLM, which is instructed to only ever lower it, never raise it —
  to avoid overconfident answers when the retrieved sources don't
  adequately cover the question.

HOW TO VALIDATE
-------------
5 test queries are defined at the end of the script, each preceded by a
comment describing what it is meant to demonstrate (recency boost,
semantic discrimination within/across domains, resistance to superficial
lexical matches, out-of-domain threshold behavior). Check the sources_used
and confidence fields of each result against the expected behavior noted
in the comment above each call.
"""

# Step 2. Imports
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from datetime import date
from google import genai
from dotenv import load_dotenv
import os
import json

# Step 3. Environment setup
load_dotenv()
# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
# Gemini API client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# Step 4. Document and Chunk dataclasses
@dataclass
class Document:
    id: str
    title: str
    category: str
    source_type: str
    date: str
    content: str

@dataclass
class Chunk:
    chunk_id: str
    parent_doc_id: str
    text: str
    title: str
    category: str
    source_type: str
    date: str

# Step 5. Corpus definition (doc1 - doc6)
doc1 = Document(
    id="doc_1",
    title="Data Access Policy v1",
    category="compliance",
    source_type="policy",
    date="2022-01-10",
    content="Employees requesting access to customer data must submit a written request to their direct manager. The manager reviews the request and forwards it to the IT Security team, who evaluates it within 10 business days. Approved requests grant access for a fixed period of 90 days, after which access is automatically revoked and must be requested again. Access logs are reviewed quarterly by the Compliance team. Employees are required to complete a general data protection training session once per year. Any suspected misuse of customer data must be reported to the direct manager, who will escalate to Compliance if necessary. This policy applies to all departments handling customer records, including Sales, Support, and Product."
)
doc2 = Document(
    id="doc_2",
    title="Data Access Policy v2",
    category="compliance",
    source_type="policy",
    date="2024-06-01",
    content="Employees requesting access to customer data must submit a request through the internal Access Management Portal, selecting the specific data category and business justification. Requests are automatically routed to the IT Security team, who must approve or reject within 3 business days using a risk-based checklist. Approved access is granted for a maximum of 30 days and requires re-justification for renewal; no automatic extensions are permitted. Access logs are reviewed monthly, and any anomaly triggers an automatic alert to the Compliance team. Employees must complete GDPR-specific training every six months. Suspected misuse must be reported directly to Compliance via the incident reporting tool, bypassing the manager, to avoid conflicts of interest. This policy supersedes all prior versions and applies company-wide."
)
doc3 = Document(
    id="doc_3",
    title="Customer Consent Register Guide",
    category="compliance",
    source_type="guide",
    date="2023-09-15",
    content="The Customer Consent Register is the authoritative source for tracking customer permissions regarding data usage, marketing communications, and third-party data sharing. Every customer interaction that involves collecting or updating consent must be logged in the register within 24 hours. The register stores the consent type, timestamp, channel of collection, and an expiration date where applicable. Marketing and Sales teams must verify active consent in the register before initiating any outbound communication. The Legal team audits the register quarterly to ensure compliance with applicable data protection regulations. Discrepancies between the register and customer-reported preferences must be resolved within 5 business days. This guide does not cover data access request procedures, which are governed separately by the Data Access Policy."
)
doc4 = Document(
    id="doc_4",
    title="Employee Onboarding Guide",
    category="onboarding",
    source_type="guide",
    date="2023-05-20",
    content="New employees complete a structured onboarding program during their first two weeks. Day one includes account provisioning, workstation setup, and an introduction to company tools including email, chat, and the internal knowledge base. During the first week, new hires attend sessions covering company culture, organizational structure, and role-specific expectations set by their manager. The second week focuses on hands-on training with the tools relevant to their team, paired with a designated mentor for guidance. HR schedules a 30-day check-in to address any open questions and collect feedback on the onboarding experience. Employees are expected to complete all mandatory compliance training modules, including data protection and workplace conduct, within the first 30 days. Onboarding completion is tracked in the HR system."
)
doc5 = Document(
    id='doc_5',
    title="Product FAQ: Reporting Module",
    category="product_faq",
    source_type="faq",
    date="2024-02-28",
    content="The Reporting Module allows users to generate, schedule, and export analytical reports based on live company data. Reports can be created using pre-built templates or customized through the drag-and-drop query builder. Scheduled reports are delivered automatically via email in PDF or CSV format, at a frequency configurable by the user (daily, weekly, or monthly). Access to specific datasets within the Reporting Module is controlled by the user's role and department permissions, configured separately by an administrator. Common issues include delayed report generation during peak hours, typically resolved by reducing the query's date range. Exported files are retained on the platform for 90 days before automatic deletion. For advanced customization, users can consult the Reporting Module technical documentation available in the internal knowledge base."
)
doc6 = Document(
    id='doc_6',
    title="Incident Escalation Procedure",
    category="operations",
    source_type="report",
    date="2024-08-01",
    content="This procedure defines how operational incidents affecting production systems must be reported and escalated. Any employee identifying a system malfunction must immediately notify the on-call engineer through the incident management platform, providing a severity classification. Severity 1 incidents (full outage) trigger automatic escalation to the Engineering Lead and Operations Manager within 15 minutes if unacknowledged. Severity 2 and 3 incidents follow a standard queue with response times of 2 and 8 business hours respectively. Once resolved, the on-call engineer documents the root cause and corrective actions in the incident log. A post-incident review is mandatory for all Severity 1 events, held within 5 business days. This procedure is unrelated to data access requests or customer data handling, which fall under separate compliance policies."
)

# Combine all documents into a single list
documents = [doc1, doc2, doc3, doc4, doc5, doc6]

# Step 6. Chunking function definition
def chunk_document(doc, max_words=200, overlap=20):
    words = doc.content.split()
    chunks = []
    start = 0
    chunk_index = 0

    # Raise an error if overlap is greater than or equal to max_words, which would cause an infinite loop
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")

    # In this exercise this branch is the only one used, since all documents have len < 200 words.
    if len(words) <= max_words:
        chunk = Chunk(
            chunk_id = f"{doc.id}_chunk_0",
            parent_doc_id = doc.id,
            text = " ".join(words),
            title = doc.title,
            category = doc.category,
            source_type = doc.source_type,
            date = doc.date
        )
        chunks.append(chunk)

    # Kept for completeness, to handle longer documents in general
    else:
        while start < len(words):
            window = words[start : start + max_words]
            chunk_text = " ".join(window)
            chunk = Chunk(
                chunk_id = f"{doc.id}_chunk_{chunk_index}",
                parent_doc_id = doc.id,
                text = chunk_text,
                title = doc.title,
                category = doc.category,
                source_type = doc.source_type,
                date = doc.date
            )
            chunks.append(chunk)
            chunk_index += 1
            start = start + (max_words - overlap)

    return chunks

# Build a flat list of chunks, including inherited metadata
all_chunks = []
for document in documents:
    all_chunks.extend(chunk_document(document))

# Step 7. Embedding
# Build an enriched text list, so that title and category are embedded alongside the chunk text
enriched_texts = []
for chunk in all_chunks:
    enriched_text = f" {chunk.title}. {chunk.category}. {chunk.text}"
    enriched_texts.append(enriched_text)

# Vectors are normalized directly during encoding (normalize_embeddings=True) to simplify the cosine similarity calculation
embeddings_matrix = model.encode(enriched_texts, normalize_embeddings=True)

# Sanity check: the matrix should have 6 rows (one per chunk) and 384 columns (all-MiniLM-L6-v2 vector size)
# embeddings_matrix.shape

# Consistency check between chunks and embeddings
# assert len(all_chunks) == embeddings_matrix.shape[0]

# Step 8. Precompute dates needed for the recency score
dates = []
for chunk in all_chunks:
    single_date = date.fromisoformat(chunk.date)
    dates.append(single_date)

min_date = min(dates)
max_date = max(dates)

# Step 9. Retrieval
# Finds the best top_k chunks based on a score that combines semantic relevance and document recency
def retrieve(query, top_k=3, alpha=0.75):
    query_vector = model.encode(query, normalize_embeddings=True)
    semantic_scores = embeddings_matrix @ query_vector

    normalized_recency_list = []
    for chunk in all_chunks:
        real_date = date.fromisoformat(chunk.date)

        normalized_recency = (real_date - min_date) / (max_date - min_date)
        normalized_recency_list.append(normalized_recency)

    final_scores = []
    for i in range(len(all_chunks)):
        final_score = alpha * semantic_scores[i] + (1-alpha) * normalized_recency_list[i]
        tuple_score = (all_chunks[i], final_score)

        final_scores.append(tuple_score)

    final_scores_sorted = sorted(final_scores, key=lambda p: p[1], reverse=True)
    top_final_scores = final_scores_sorted[:top_k]

    return top_final_scores

# --- Empirical calibration of min_score_threshold (performed once during development) ---
# print(retrieve("What's the weather today?")[0][1])   # 0.2652609
# print(retrieve("Best pizza recipe")[0][1])            # 0.2434846
# print(retrieve("data access procedure for customer data")[0][1])  # 0.66481614
#
# There is a substantial gap between out-of-domain and on-domain scores.
# Based on these empirically observed values, min_score_threshold is set to 0.4.

# Step 10. Prompt construction
# Builds a context string from the top retrieval results
def build_context(top_results):
    formatted_lines = ""
    for chunk, score in top_results:
        formatted_line = f"[Source: {chunk.parent_doc_id} | {chunk.title} | {chunk.date}]\n{chunk.text}\n\n"
        formatted_lines += formatted_line

    return formatted_lines

def score_to_confidence(score):
    if score > 0.7: confidence = "high"
    elif 0.5 < score <= 0.7: confidence = "medium"
    else: confidence = "low"

    return confidence

SYSTEM_INSTRUCTIONS = """
You are a virtual assistant that answers business-related questions.
You are provided with a set of documents that represent your entire knowledge base. You must not use any knowledge that is not contained in these documents.
If the question is not relevant to the content of the documents, you must state this explicitly, for example with the phrase "I don't have information on the requested topic."
Always cite the source you are using to answer, using the document id as the identifier.
The output must be in JSON format, following the exact schema provided below. You must not change field names or take any liberties in interpreting this template.
JSON structure: {"answer", "sources_used", "confidence", "reasoning_confidence"}
The "sources_used" field is composed of the following fields: "sources_used": [{"id": "...", "title": "...", "date": "..."}]
You will receive a retrieval score based on how well the answer matches the requested information. If you believe it is necessary, you may lower this score, but you must NEVER raise it.
Do not wrap the JSON in markdown code blocks or backticks. Return raw JSON only.
"""

def build_prompt(query, top_results):

    context = build_context(top_results)
    confidence = score_to_confidence(top_results[0][1])
    prompt = f"""
    SYSTEM INSTRUCTIONS:
    {SYSTEM_INSTRUCTIONS}

    CONTEXT:
    {context}

    QUESTION:
    {query}

    RETRIEVAL CONFIDENCE:
    {confidence}
    """

    return prompt

# Step 11. Generation + parsing

def clean_json_text(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()

# Main orchestration function
def answer_query(query, top_k=3, alpha=0.75, min_score_threshold=0.4):

    results = retrieve(query)
    if results[0][1] > min_score_threshold:
        prompt = build_prompt(query, results)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents = prompt
        )

        raw_text = response.text

        try:
            cleaned = clean_json_text(raw_text)
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = {"error": "JSON parsing failed", "raw_output": raw_text}
    else:
        parsed = {"answer": "No relevant sources found for this query",
                    "sources_used": [],
                    "confidence":"low",
                    "reasoning_confidence": "Retrieval score below threshold"
                   }

    return parsed

# Step 12. Demo

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