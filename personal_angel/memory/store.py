"""SQLite-backed hierarchical memory with hybrid (lexical + vector) retrieval.

  episodic   – what happened in earlier investigations (per event, per run)
  semantic   – stable facts about an environment / policy documents (RAG)
  procedural – investigation strategies that worked ("when X, do Y")

Embeddings come from the local OpenAI-compatible endpoint when available
(nomic-embed-text on Ollama, Qwen3-Embedding on vLLM) and otherwise from a
feature-hashing encoder so retrieval works fully offline.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

_TOKEN = re.compile(r"[a-záéíóúñü0-9_]+", re.I)

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]

class HashingEmbedder:
    name = "hashing_256"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            toks = tokenize(text)
            for i, tok in enumerate(toks):
                grams = [tok] + ([toks[i - 1] + "_" + tok] if i > 0 else [])
                for g in grams:
                    h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
                    vec[h % self.dim] += 1.0 if (h >> 8) % 2 == 0 else -1.0
            norm = float(np.linalg.norm(vec)) or 1.0
            out.append((vec / norm).tolist())
        return out

class EndpointEmbedder:
    def __init__(self, llm, fallback: HashingEmbedder) -> None:
        self.llm = llm
        self.fallback = fallback
        self.name = f"endpoint:{getattr(llm, 'embedding_model', 'unknown')}"
        self._dead = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._dead:
            vectors = self.llm.embed(texts)
            if vectors:
                return [list(np.asarray(v, dtype=np.float32) / (np.linalg.norm(v) or 1.0)) for v in vectors]
            self._dead = True
            self.name = self.fallback.name + "(fallback)"
        return self.fallback.embed(texts)

class MemoryStore:
    def __init__(self, path: str | Path, embedder, policy_docs: list[Path] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY AUTOINCREMENT, tier TEXT, "
            "scope TEXT, text TEXT, meta TEXT, embedding BLOB, created REAL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_tier ON memory(tier)")
        self.db.commit()
        for doc in policy_docs or []:
            self.index_document(doc)

    def remember(self, tier: str, text: str, meta: dict[str, Any] | None = None, scope: str = "global") -> int:
        vec = np.asarray(self.embedder.embed([text])[0], dtype=np.float32)
        cur = self.db.execute("INSERT INTO memory(tier, scope, text, meta, embedding, created) VALUES (?,?,?,?,?,?)",
                              (tier, scope, text, json.dumps(meta or {}), vec.tobytes(), time.time()))
        self.db.commit()
        return int(cur.lastrowid)

    def index_document(self, path: Path) -> int:
        if not path.exists():
            return 0
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        existing = self.db.execute("SELECT COUNT(*) FROM memory WHERE scope=?", (f"doc:{digest}",)).fetchone()[0]
        if existing:
            return 0
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = [c.strip() for c in re.split(r"\n\s*\n|\n(?=#)", text) if len(c.strip()) > 40]
        for i, chunk in enumerate(chunks):
            self.remember("semantic", chunk, {"source": path.name, "chunk": i, "kind": "policy"}, scope=f"doc:{digest}")
        return len(chunks)

    def search(self, query: str, tiers: tuple[str, ...] = ("episodic", "semantic", "procedural"),
               top_k: int = 4, prefer_source: str | None = None, scope: str | None = None) -> list[dict[str, Any]]:
        """Hybrid retrieval. `prefer_source` boosts chunks whose document/source name contains the
        string (e.g. 'vehicle' for CAR_CABIN); `scope` boosts memories recorded at the same location."""
        rows = self.db.execute("SELECT id, tier, scope, text, meta, embedding, created FROM memory WHERE tier IN (%s)"
                               % ",".join("?" * len(tiers)), tiers).fetchall()
        if not rows:
            return []
        qvec = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        qtoks = set(tokenize(query))
        n_docs = len(rows)
        df: dict[str, int] = {}
        doc_tokens = []
        for row in rows:
            toks = set(tokenize(row[3]))
            doc_tokens.append(toks)
            for t in toks:
                df[t] = df.get(t, 0) + 1
        results = []
        for row, toks in zip(rows, doc_tokens):
            vec = np.frombuffer(row[5], dtype=np.float32)
            if vec.shape != qvec.shape:
                cos = 0.0
            else:
                cos = float(np.dot(vec, qvec))
            overlap = qtoks & toks
            lexical = sum(math.log(1 + n_docs / df[t]) for t in overlap) / (math.log(1 + n_docs) * max(len(qtoks), 1))
            recency = 1.0 / (1.0 + (time.time() - row[6]) / 86400.0)
            score = 0.55 * cos + 0.35 * lexical + 0.10 * recency
            meta = json.loads(row[4] or "{}")
            if prefer_source and prefer_source.lower() in str(meta.get("source", "")).lower():
                score += 0.25
            if scope and row[2] == scope:
                score += 0.15
            results.append({"id": row[0], "tier": row[1], "scope": row[2], "text": row[3],
                            "meta": meta, "score": round(score, 4),
                            "components": {"vector": round(cos, 3), "lexical": round(lexical, 3), "recency": round(recency, 3)}})
        results.sort(key=lambda r: -r["score"])
        return [r for r in results[:top_k] if r["score"] > 0.05]

    def count(self, tier: str | None = None) -> int:
        if tier:
            return int(self.db.execute("SELECT COUNT(*) FROM memory WHERE tier=?", (tier,)).fetchone()[0])
        return int(self.db.execute("SELECT COUNT(*) FROM memory").fetchone()[0])

    def reflect(self, run_id: str, scenario: str, events: list[dict[str, Any]], decisions: list[dict[str, Any]],
                final_answer: str | None, location: str) -> dict[str, int]:
        """Compress an investigation into reusable memories (Generative-Agents style)."""
        written = {"episodic": 0, "procedural": 0, "semantic": 0}
        for event in events:
            if event.get("kind") == "normal_activity":
                continue
            text = (f"Episode {run_id}: {event['kind']} at {location} — {event['subject']} {event['action']} "
                    f"{event.get('obj') or ''} ({event['start_s']:.1f}s–{event['end_s']:.1f}s, confidence "
                    f"{event['confidence']:.2f}). Outcome: {final_answer or 'n/a'}"[:600])
            self.remember("episodic", text, {"run_id": run_id, "kind": event["kind"], "location": location}, scope=location)
            written["episodic"] += 1
        executed = [d for d in decisions if d.get("executed")]
        if executed and events:
            kinds = sorted({e["kind"] for e in events if e.get("kind") != "normal_activity"})
            steps = " → ".join(d["action"] for d in executed)
            text = f"Strategy: when {', '.join(kinds)} is observed at a {location.lower()}, the successful sequence was: {steps}."
            self.remember("procedural", text, {"run_id": run_id, "kinds": kinds}, scope=location)
            written["procedural"] += 1
        return written

    def close(self) -> None:
        self.db.close()

def create_memory(config: dict[str, Any], llm=None, project_root: Path | None = None) -> MemoryStore:
    project_root = project_root or Path(".")
    hashing = HashingEmbedder()
    embedder = hashing
    if str(config.get("embedding_backend", "hashing")) == "openai_compatible" and llm is not None and getattr(llm, "is_real", False):
        embedder = EndpointEmbedder(llm, hashing)
    docs = [project_root / p for p in config.get("policy_docs", [])]
    return MemoryStore(project_root / str(config.get("path", "runs/memory/angel_memory.sqlite")), embedder, docs)
