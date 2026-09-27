"""Two-level answer cache for follow-up questions (lecture 08 pattern):
exact-key lookup first, then embedding similarity (cosine ≥ threshold) so
"who took the laptop?" and "which person removed the laptop?" reuse one LLM
call. In-process by default; drop-in Redis backend when REDIS_URL is set.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import numpy as np

from .store import HashingEmbedder

class SemanticCache:
    def __init__(self, threshold: float = 0.85, ttl_s: float = 6 * 3600, embedder=None) -> None:
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.embedder = embedder or HashingEmbedder()
        self.entries: dict[str, list[dict[str, Any]]] = {}
        self.hits = {"exact": 0, "semantic": 0, "miss": 0}
        self.redis = None
        url = os.environ.get("REDIS_URL")
        if url:
            try:
                import redis

                self.redis = redis.Redis.from_url(url)
                self.redis.ping()
            except Exception:
                self.redis = None

    @staticmethod
    def _key(scope: str, question: str) -> str:
        return "sc:" + hashlib.sha1(f"{scope}|{question.strip().lower()}".encode("utf-8")).hexdigest()

    def get(self, scope: str, question: str) -> dict[str, Any] | None:
        key = self._key(scope, question)
        now = time.time()
        if self.redis is not None:
            raw = self.redis.get(key)
            if raw:
                self.hits["exact"] += 1
                return {"answer": json.loads(raw)["answer"], "kind": "exact"}
        for entry in self.entries.get(scope, []):
            if entry["key"] == key and now - entry["t"] < self.ttl_s:
                self.hits["exact"] += 1
                return {"answer": entry["answer"], "kind": "exact"}
        qvec = np.asarray(self.embedder.embed([question])[0], dtype=np.float32)
        best, best_sim = None, 0.0
        for entry in self.entries.get(scope, []):
            if now - entry["t"] >= self.ttl_s:
                continue
            sim = float(np.dot(qvec, entry["vec"]))
            if sim > best_sim:
                best, best_sim = entry, sim
        if best is not None and best_sim >= self.threshold:
            self.hits["semantic"] += 1
            return {"answer": best["answer"], "kind": "semantic", "similarity": round(best_sim, 3)}
        self.hits["miss"] += 1
        return None

    def put(self, scope: str, question: str, answer: str) -> None:
        key = self._key(scope, question)
        vec = np.asarray(self.embedder.embed([question])[0], dtype=np.float32)
        self.entries.setdefault(scope, []).append({"key": key, "question": question, "answer": answer, "vec": vec, "t": time.time()})
        if self.redis is not None:
            self.redis.setex(key, int(self.ttl_s), json.dumps({"answer": answer}))

    def stats(self) -> dict[str, Any]:
        total = sum(self.hits.values())
        return {**self.hits, "hit_rate": round((self.hits["exact"] + self.hits["semantic"]) / total, 3) if total else None}
