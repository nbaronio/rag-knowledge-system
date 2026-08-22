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
    words = doc.content.split()
    chunks = []
    start = 0
    chunk_index = 0

    # Raise an error if overlap >= max_words, which would cause an infinite loop
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")

    # In this exercise this branch is the only one used, since all documents have len < 200 words.
    if len(words) <= max_words:
        chunk = Chunk(
            chunk_id=f"{doc.id}_chunk_0",
            parent_doc_id=doc.id,
            text=" ".join(words),
            title=doc.title,
            category=doc.category,
            source_type=doc.source_type,
            date=doc.date
        )
        chunks.append(chunk)

    # Kept for completeness, to handle longer documents in general
    else:
        while start < len(words):
            window = words[start : start + max_words]
            chunk_text = " ".join(window)
            chunk = Chunk(
                chunk_id=f"{doc.id}_chunk_{chunk_index}",
                parent_doc_id=doc.id,
                text=chunk_text,
                title=doc.title,
                category=doc.category,
                source_type=doc.source_type,
                date=doc.date
            )
            chunks.append(chunk)
            chunk_index += 1
            start = start + (max_words - overlap)

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
