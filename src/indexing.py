"""
Chunking and embedding: turns Document objects into embedded Chunk objects.
"""

from dataclasses import dataclass
from config import model

@dataclass
class Chunk:
    chunk_id: str
    parent_doc_id: str
    text: str
    title: str
    category: str
    source_type: str
    date: str

def chunk_document(doc, max_words=200, overlap=20):
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")

    words = doc.content.split()
    chunks = []
    start = 0
    chunk_index = 0

    if len(words) <= max_words:
        chunks.append(Chunk(
            chunk_id=f"{doc.id}_chunk_0", parent_doc_id=doc.id, text=" ".join(words),
            title=doc.title, category=doc.category, source_type=doc.source_type, date=doc.date
        ))
    else:
        while start < len(words):
            window = words[start : start + max_words]
            chunks.append(Chunk(
                chunk_id=f"{doc.id}_chunk_{chunk_index}", parent_doc_id=doc.id,
                text=" ".join(window), title=doc.title, category=doc.category,
                source_type=doc.source_type, date=doc.date
            ))
            chunk_index += 1
            start += max_words - overlap

    return chunks

def build_all_chunks(documents):
    all_chunks = []
    for document in documents:
        all_chunks.extend(chunk_document(document))
    return all_chunks

def build_embeddings_matrix(all_chunks):
    # Enriched text (title + category + chunk text) is embedded instead of
    # the raw content alone, to improve semantic discrimination between
    # similar documents.
    enriched_texts = []
    for chunk in all_chunks:
        enriched_text = f" {chunk.title}. {chunk.category}. {chunk.text}"
        enriched_texts.append(enriched_text)

    # Vectors are normalized directly during encoding to simplify cosine similarity
    return model.encode(enriched_texts, normalize_embeddings=True)

if __name__ == "__main__":
    from ingestion import Document

    # Document over max_words: exercises the sliding-window overlap branch,
    # never hit by the real corpus (all six documents are under 200 words).
    long_doc = Document(
        id="test_long", title="Long Doc", category="test", source_type="policy",
        date="2024-01-01", content=" ".join(f"word{i}" for i in range(500))
    )
    long_chunks = chunk_document(long_doc, max_words=200, overlap=20)
    assert len(long_chunks) > 1, "500-word document should be split into multiple chunks"
    assert long_chunks[0].text.split()[-1] == long_chunks[1].text.split()[19], "expected 20-word overlap between consecutive windows"

    # Document under max_words: exercises the single-chunk branch
    short_doc = Document(
        id="test_short", title="Short Doc", category="test", source_type="policy",
        date="2024-01-01", content="just a few words"
    )
    short_chunks = chunk_document(short_doc)
    assert len(short_chunks) == 1

    # overlap >= max_words must raise instead of looping forever
    try:
        chunk_document(short_doc, max_words=10, overlap=10)
        assert False, "expected ValueError when overlap >= max_words"
    except ValueError:
        pass

    print("indexing.py: all assertions passed")
