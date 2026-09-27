"""Content-addressed JSON cache so a second question about the same video does
not re-run perception or repeat identical VLM calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

def stable_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]

class JsonCache:
    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        return self.directory / namespace / f"{key}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(namespace, key)
        if path.exists():
            self.hits += 1
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        self.misses += 1
        return None

    def put(self, namespace: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle, default=str)
