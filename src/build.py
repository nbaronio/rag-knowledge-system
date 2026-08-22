"""
Concatenates the src/ modules into a single rag_system.py deliverable file,
stripping internal cross-module imports (config, ingestion, indexing,
retrieval, confidence, generation) since everything ends up in one
namespace. External imports (sentence_transformers, google.genai, etc.) are
kept as-is.
"""

from pathlib import Path

MODULE_ORDER = [
    "config.py",
    "ingestion.py",
    "indexing.py",
    "retrieval.py",
    "confidence.py",
    "generation.py",
    "main.py",
]

INTERNAL_MODULES = {"config", "ingestion", "indexing", "retrieval", "confidence", "generation", "main"}

def strip_internal_imports(lines):
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("from ") or stripped.startswith("import "):
            first_token = stripped.split()[1].split(".")[0]
            if first_token in INTERNAL_MODULES:
                continue
        cleaned.append(line)
    return cleaned

def build(src_dir="src", output_path="rag_system.py"):
    src_path = Path(src_dir)
    output_lines = []

    for filename in MODULE_ORDER:
        file_path = src_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing module: {file_path}")

        lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
        cleaned_lines = strip_internal_imports(lines)

        output_lines.append(f"\n# ==== {filename} ====\n")
        output_lines.extend(cleaned_lines)

    Path(output_path).write_text("".join(output_lines), encoding="utf-8")
    print(f"Built {output_path} from {len(MODULE_ORDER)} modules")

if __name__ == "__main__":
    build()
