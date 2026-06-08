"""
rag/loader.py
-------------
Responsibility: Load myth_facts.json and convert each entry into a
flat, searchable Document chunk with metadata preserved.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class KBDocument:
    chunk_text: str
    doc_id: str
    myth: str
    verdict: str
    explanation: str
    category: str
    crop: str
    source: str
    source_url: str
    confidence_basis: str
    tags: List[str] = field(default_factory=list)


class KBLoader:
    REQUIRED_FIELDS = [
        "id", "myth", "swahili_myth", "sheng_myth",
        "verdict", "explanation", "crop", "category",
        "source", "source_url", "confidence_basis", "tags"
    ]
    VALID_VERDICTS = {"Myth", "Fact", "Partially True"}

    def __init__(self, data_path: Optional[str] = None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            project_root = Path(__file__).parent.parent
            self.data_path = project_root / "data" / "myth_facts.json"

    def load(self) -> List[KBDocument]:
        raw = self._read_json()
        entries = self._extract_entries(raw)
        documents = []
        skipped = 0
        for entry in entries:
            error = self._validate_entry(entry)
            if error:
                print(f"[KBLoader] WARNING — skipping '{entry.get('id', 'UNKNOWN')}': {error}")
                skipped += 1
                continue
            documents.append(self._build_document(entry))
        print(f"[KBLoader] Loaded {len(documents)} documents ({skipped} skipped).")
        return documents

    def load_raw(self) -> List[dict]:
        raw = self._read_json()
        return self._extract_entries(raw)

    def get_metadata(self) -> dict:
        raw = self._read_json()
        return raw.get("metadata", {})

    def _read_json(self) -> dict:
        if not self.data_path.exists():
            raise FileNotFoundError(f"[KBLoader] Not found: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_entries(self, raw: dict) -> List[dict]:
        if "entries" not in raw:
            raise ValueError("[KBLoader] Missing 'entries' key in myth_facts.json")
        return raw["entries"]

    def _validate_entry(self, entry: dict) -> Optional[str]:
        for field_name in self.REQUIRED_FIELDS:
            if field_name not in entry:
                return f"missing field '{field_name}'"
            if isinstance(entry[field_name], str) and not entry[field_name].strip():
                return f"empty field '{field_name}'"
        if entry["verdict"] not in self.VALID_VERDICTS:
            return f"invalid verdict '{entry['verdict']}'"
        if not isinstance(entry["tags"], list):
            return "tags must be a list"
        return None

    def _build_chunk_text(self, entry: dict) -> str:
        tags_str = ", ".join(entry.get("tags", []))
        return (
            f"CLAIM (EN): {entry['myth']}\n"
            f"CLAIM (SW): {entry['swahili_myth']}\n"
            f"CLAIM (SH): {entry['sheng_myth']}\n"
            f"VERDICT: {entry['verdict']}\n"
            f"EXPLANATION: {entry['explanation']}\n"
            f"TAGS: {tags_str}"
        )

    def _build_document(self, entry: dict) -> KBDocument:
        return KBDocument(
            chunk_text=self._build_chunk_text(entry),
            doc_id=entry["id"],
            myth=entry["myth"],
            verdict=entry["verdict"],
            explanation=entry["explanation"],
            category=entry["category"],
            crop=entry["crop"],
            source=entry["source"],
            source_url=entry["source_url"],
            confidence_basis=entry["confidence_basis"],
            tags=entry["tags"],
        )