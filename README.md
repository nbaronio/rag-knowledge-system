# RAG Knowledge Management System

Retrieval-Augmented Generation system to query internal company documentation (policies, guides, FAQs) in natural language, returning synthetic answers with cited sources and a confidence indicator.

## Goal

Answer employee questions using only the company's own documentation, with a focus on **traceability**: every answer must point back to the exact source(s) it came from, with a confidence signal — never a plausible-sounding answer that isn't grounded in the provided documents.

## Pipeline

- **Ingestion**: 6 documents (policies, guides, FAQs, incident reports) as `.txt` files under `docs/`, each with a metadata header (`id`, `title`, `category`, `source_type`, `date`) followed by a `---` separator and the content. `ingestion.py` parses and validates every file (missing separator or missing required field raises), covering compliance, onboarding, product, and operations, with two conflicting versions of the same policy (`doc_1`/`doc_2`) to test recency handling.
- **Chunking**: fixed-size word-based chunking (`max_words`/`overlap` configurable). On this corpus every document fits in a single chunk; the sliding-window branch for longer documents is exercised by `indexing.py`'s own self-test (see Running).
- **Embedding**: `sentence-transformers` (`all-MiniLM-L6-v2`), applied to an enriched text (title + category + chunk text) rather than raw content alone, to improve semantic discrimination between similar documents.
- **Indexing**: in-memory numpy matrix, no external vector database — with ~6 chunks, brute-force cosine similarity is already instantaneous.
- **Hybrid retrieval**: `final_score = alpha * semantic_score + (1 - alpha) * recency`, `alpha = 0.9`. Semantic similarity dominates and drives ranking; the relevance threshold and the confidence label are both computed from `semantic_score` alone, so the recency component only ever acts as a tie-breaker between already-relevant documents — it can never make an irrelevant chunk pass the threshold or look more trustworthy than its content justifies.
- **Generation**: dynamically built prompt sent to the Gemini API, with strict instructions to answer only from the provided context and to output structured JSON (`answer`, `key_points`, `sources_used`, `confidence`, `reasoning_confidence`). The network call is wrapped so a timeout/rate-limit/5xx returns an error payload instead of crashing the run.
- **Confidence**: hybrid score — a baseline confidence (high/medium/low) is computed from the semantic retrieval score and passed to the LLM, which may only lower it, never raise it.

## Results

All 5 test queries behave as designed:
- A query about data access correctly returns the **updated** policy (`doc_2`, 2024) as primary source instead of the superseded one (`doc_1`, 2022) — confirming the recency tie-break works even though `doc_1`'s raw semantic similarity is actually the higher of the two.
- Queries on consent, onboarding, and incident handling each retrieve the correct, distinct document, without being confused by shared category or superficial keyword overlap (e.g. "procedure" appears in both the access policies and the incident report, but the incident query correctly retrieves `doc_6`).
- An out-of-domain query ("What is today's weather?") falls below the semantic-relevance threshold and returns a fixed "no relevant sources" response, without calling the LLM.

Example output:
```json
{
  "answer": "Under the current Data Access Policy (v2), employees requesting access to customer data must submit a request through the internal Access Management Portal, selecting the specific data category and providing a business justification. Under the prior, superseded policy (v1), employees were required to submit a written request to their direct manager.",
  "key_points": [
    "Requests for customer data access must be submitted via the internal Access Management Portal.",
    "Employees must select the specific data category and provide a business justification.",
    "This current procedure supersedes the older Policy v1, which required written submission to a direct manager."
  ],
  "sources_used": [
    {"id": "doc_2", "title": "Data Access Policy v2", "date": "2024-06-01"},
    {"id": "doc_1", "title": "Data Access Policy v1", "date": "2022-01-10"}
  ],
  "confidence": "medium",
  "reasoning_confidence": "Doc 2 provides the current procedure for requesting access to customer data through the Access Management Portal, explicitly stating it supersedes Doc 1."
}
```

**Conclusion**: on this small, deliberately-designed corpus, the hybrid retrieval mechanism (semantic + recency) correctly resolves versioning conflicts and domain discrimination, and the confidence/threshold logic — both driven by semantic relevance alone — prevents ungrounded or overconfident answers on out-of-domain queries.

## Recommendations

- Calibrate `MIN_SCORE_THRESHOLD` again on any new/larger corpus — the current value (0.15, applied to `semantic_score` alone) was fit empirically on this specific 6-document set (see the calibration comment in `retrieval.py`).
- `min_date`/`max_date` are derived once from the corpus at startup; ingesting new documents later requires recomputing `recency_scores` (via `precompute_recency_scores`) so the recency component stays meaningful.
- For a larger corpus, replace the in-memory numpy index with a proper vector store (FAISS/Chroma).
- Add a proper `origin`/department metadata field if the source system of each document needs to be tracked beyond `category`/`source_type`.

## Structure
```
docs/               6 source documents (metadata header + "---" + content)
src/
  config.py         shared clients: embedding model, Gemini client
  ingestion.py       parses docs/*.txt into Document objects, validates required metadata
  indexing.py        chunking (word-based, overlap-aware) and embedding
  retrieval.py        hybrid semantic + recency scoring and ranking
  confidence.py       maps a semantic score to a high/medium/low confidence label
  generation.py       prompt construction, Gemini call, structured JSON parsing
  main.py             orchestration + the 5 demo queries
  build.py            concatenates the modules above into a single rag_system.py deliverable
```

## Running

1. `pip install -r requirements.txt`
2. Create a `.env` file (see `.env.example`) with your `GEMINI_API_KEY`
3. `python src/main.py` (run from the repo root, so the `docs/` relative path resolves)
4. Optional: `python src/ingestion.py` and `python src/indexing.py` each run a small self-test (parsing edge cases; the multi-chunk sliding-window branch) independently of the full pipeline
5. Optional: `python src/build.py` concatenates the modules into a single `rag_system.py` file

## Stack

Python · sentence-transformers · numpy · google-genai (Gemini) · python-dotenv
