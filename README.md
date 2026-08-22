# RAG Knowledge Management System

Retrieval-Augmented Generation system to query internal company documentation (policies, guides, FAQs) in natural language, returning synthetic answers with cited sources and a confidence indicator.

## Goal

Answer employee questions using only the company's own documentation, with a focus on **traceability**: every answer must point back to the exact source(s) it came from, with a confidence signal — never a plausible-sounding answer that isn't grounded in the provided documents.

## Pipeline

- **Ingestion**: 6 source documents (policies, guides, FAQs, incident reports) stored as `.txt` files in `docs/`, each with a metadata header (`id`, `title`, `category`, `source_type`, `date`) followed by the document content. The corpus includes two conflicting versions of the same policy (`doc_1`/`doc_2`) to test recency handling.
- **Chunking**: fixed-size word-based chunking (`max_words`/`overlap` configurable). On this corpus every document fits in a single chunk, but the function generalizes to longer documents.
- **Embedding**: `sentence-transformers` (`all-MiniLM-L6-v2`), applied to an enriched text (title + category + chunk text) rather than raw content alone, to improve semantic discrimination between similar documents.
- **Indexing**: in-memory numpy matrix, no external vector database — with ~6 chunks, brute-force cosine similarity is already instantaneous.
- **Hybrid retrieval**: `final_score = alpha * semantic_score + (1 - alpha) * recency`, `alpha = 0.75`. Semantic similarity dominates; recency acts as a tie-breaker so that, between two similar documents, the more recent one wins.
- **Generation**: dynamically built prompt sent to the Gemini API, with strict instructions to answer only from the provided context and to output structured JSON.
- **Confidence**: hybrid score — a baseline confidence (high/medium/low) is computed from the retrieval score and passed to the LLM, which may only lower it, never raise it.

## Results

All 5 test queries behave as designed:
- A query about data access correctly returns the **updated** policy (`doc_2`, 2024) instead of the superseded one (`doc_1`, 2022), confirming the recency boost works.
- Queries on consent, onboarding, and incident handling each retrieve the correct, distinct document, without being confused by shared category or superficial keyword overlap (e.g. "procedure" appears in both the access policies and the incident report, but the incident query correctly retrieves `doc_6`).
- An out-of-domain query ("What is today's weather?") falls below the retrieval threshold and returns a fixed "no relevant sources" response, without calling the LLM.

Example output:
```json
{
  "answer": "According to the Data Access Policy v2, employees requesting access to customer data must submit a request through the internal Access Management Portal, selecting the specific data category and providing a business justification.",
  "sources_used": [
    {"id": "doc_2", "title": "Data Access Policy v2", "date": "2024-06-01"}
  ],
  "confidence": "medium",
  "reasoning_confidence": "The answer directly reflects the current Data Access Policy (v2), which explicitly supersedes prior versions."
}
```

**Conclusion**: on this small, deliberately-designed corpus, the hybrid retrieval mechanism (semantic + recency) correctly resolves versioning conflicts and domain discrimination, and the confidence/threshold logic prevents ungrounded answers on out-of-domain queries.

## Recommendations

- Calibrate `min_score_threshold` again on any new/larger corpus — the current value (0.4) was fit empirically on this specific 6-document set.
- For a larger corpus, replace the in-memory numpy index with a proper vector store (FAISS/Chroma) and re-enable the chunking logic for real multi-chunk documents.
- Add a proper `origin`/department metadata field if the source system of each document needs to be tracked beyond `category`/`source_type`.

## Structure

docs/ source documents (.txt), one per file, each with a metadata header
src/
  config.py shared setup: embedding model, Gemini client
  ingestion.py reads and parses documents from docs/
  indexing.py chunking + embedding
  retrieval.py hybrid retrieval (semantic + recency) and threshold calibration notes
  confidence.py retrieval-score-to-confidence-label mapping
  generation.py prompt construction, LLM call, JSON parsing
  main.py wires everything together, runs the demo queries
  build.py concatenates src/ modules into a single rag_system.py deliverable


## Running

1. `pip install -r requirements.txt`
2. Create a `.env` file (see `.env.example`) with your `GEMINI_API_KEY`
3. `python src/main.py`

To generate the single-file deliverable:

python build.py

This produces `rag_system.py` by concatenating all `src/` modules in dependency order and stripping internal cross-module imports.

## Stack

Python · sentence-transformers · numpy · google-genai (Gemini) · python-dotenv
