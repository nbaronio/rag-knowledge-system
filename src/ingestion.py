"""
Ingestion module: reads and parses the company documents from the docs/
folder. Each source file contains a metadata header (key: value lines),
a "---" separator, and the document content.
"""

from pathlib import Path
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    title: str
    category: str
    source_type: str
    date: str
    content: str

REQUIRED_FIELDS = {"id", "title", "category", "source_type", "date"}

def parse_document_file(file_path):
    raw = file_path.read_text(encoding="utf-8")

    if "---" not in raw:
        raise ValueError(f"{file_path.name}: '---' separator missing")

    header, content = raw.split("---", 1)

    metadata = {}
    for line in header.strip().splitlines():
        if ":" not in line:
            raise ValueError(f"{file_path.name}: malformed line '{line}'")
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    missing = REQUIRED_FIELDS - metadata.keys()
    if missing:
        raise ValueError(f"{file_path.name}: missing metadata {missing}")

    return Document(
        id=metadata["id"],
        title=metadata["title"],
        category=metadata["category"],
        source_type=metadata["source_type"],
        date=metadata["date"],
        content=content.strip()
    )

def load_documents(folder_path="docs"):
    folder = Path(folder_path)
    files = sorted(folder.glob("*.txt"))

    if not files:
        raise FileNotFoundError(f"No .txt file found in {folder}")

    return [parse_document_file(f) for f in files]

if __name__ == "__main__":
    import tempfile

    def _parse_text(text):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.txt"
            file_path.write_text(text, encoding="utf-8")
            return parse_document_file(file_path)

    # Well-formed document parses correctly
    doc = _parse_text(
        "id: t1\ntitle: Test\ncategory: cat\nsource_type: policy\ndate: 2024-01-01\n---\nSome content."
    )
    assert doc.id == "t1" and doc.content == "Some content."

    # Missing '---' separator
    try:
        _parse_text("id: t1\ntitle: Test\ncategory: cat\nsource_type: policy\ndate: 2024-01-01")
        assert False, "expected ValueError for missing separator"
    except ValueError:
        pass

    # Missing required metadata field (source_type)
    try:
        _parse_text("id: t1\ntitle: Test\ncategory: cat\ndate: 2024-01-01\n---\nSome content.")
        assert False, "expected ValueError for missing required field"
    except ValueError:
        pass

    print("ingestion.py: all assertions passed")
