"""Event / evidence / entity graph with personalized PageRank.

Why PageRank here (measurable reason): the master agent must decide *which*
evidence to spend an expensive VLM call on. Ranking evidence nodes by
personalized PageRank seeded at the current hypothesis event favours evidence
that is shared by several events/entities (e.g. a frame that shows both the
knife track and the person track), which the ablation script compares against
naive "highest detector confidence first" ordering.
"""
from __future__ import annotations

from typing import Any

import networkx as nx

from ..schema import Event, Evidence

class EventGraph:
    def __init__(self, alpha: float = 0.85, pagerank_enabled: bool = True) -> None:
        self.g = nx.DiGraph()
        self.alpha = alpha
        self.pagerank_enabled = pagerank_enabled
        self._evidence_scores: dict[str, float] = {}

    def build(self, events: list[Event], evidence: dict[str, Evidence]) -> "EventGraph":
        g = self.g
        for ev_id, ev in evidence.items():
            g.add_node(ev_id, type="evidence", kind=ev.kind, t=ev.start_s, label=ev.description[:80])
            self._evidence_scores[ev_id] = float(ev.score)
        ordered = sorted(events, key=lambda e: e.start_s)
        for event in ordered:
            g.add_node(event.event_id, type="event", kind=event.kind, t=event.start_s,
                       label=f"{event.subject} {event.action} {event.obj or ''}".strip())
            g.add_node(event.subject, type="entity", label=event.subject)
            g.add_edge(event.event_id, event.subject, rel="INVOLVES")
            g.add_edge(event.subject, event.event_id, rel="PARTICIPATES_IN")
            if event.obj:
                g.add_node(event.obj, type="entity", label=event.obj)
                g.add_edge(event.event_id, event.obj, rel=event.action)
                g.add_edge(event.obj, event.event_id, rel="INVOLVED_IN")
            g.add_node(event.location, type="location", label=event.location)
            g.add_edge(event.event_id, event.location, rel="LOCATED_IN")
            for ev_id in event.evidence_ids:
                if ev_id in g:
                    g.add_edge(ev_id, event.event_id, rel="SUPPORTS")
                    g.add_edge(event.event_id, ev_id, rel="CITES")
        for a, b in zip(ordered, ordered[1:]):
            g.add_edge(a.event_id, b.event_id, rel="BEFORE")
            if a.subject == b.subject or (b.obj and a.obj == b.obj):
                g.add_edge(a.event_id, b.event_id, rel="MAY_CAUSE")
        return self

    def rank_evidence(self, seed_event_ids: list[str], top_k: int = 6) -> list[tuple[str, float]]:
        if self.g.number_of_nodes() == 0:
            return []
        if not self.pagerank_enabled:
            ranked = sorted(self._evidence_scores.items(), key=lambda x: -x[1])
            return ranked[:top_k]
        seeds = {n: 1.0 for n in seed_event_ids if n in self.g}
        try:
            pr = nx.pagerank(self.g, alpha=self.alpha, personalization=seeds or None, max_iter=200)
        except Exception:
            pr = {n: 1.0 / self.g.number_of_nodes() for n in self.g}
        ranked = [(n, s) for n, s in pr.items() if self.g.nodes[n].get("type") == "evidence"]
        ranked.sort(key=lambda x: -x[1])
        return ranked[:top_k]

    def important_entities(self, top_k: int = 5) -> list[tuple[str, float]]:
        if self.g.number_of_nodes() == 0:
            return []
        pr = nx.pagerank(self.g, alpha=self.alpha, max_iter=200)
        ranked = [(n, s) for n, s in pr.items() if self.g.nodes[n].get("type") == "entity"]
        ranked.sort(key=lambda x: -x[1])
        return ranked[:top_k]

    def neighbors(self, node: str) -> list[dict[str, Any]]:
        if node not in self.g:
            return []
        out = []
        for _, nbr, data in self.g.out_edges(node, data=True):
            out.append({"to": nbr, "rel": data.get("rel"), "type": self.g.nodes[nbr].get("type")})
        return out

    def export(self) -> dict[str, Any]:
        nodes = [{"id": n, **{k: v for k, v in d.items()}} for n, d in self.g.nodes(data=True)]
        edges = [{"source": a, "target": b, "rel": d.get("rel")} for a, b, d in self.g.edges(data=True)]
        return {"nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges)}
