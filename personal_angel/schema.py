"""Typed data contracts shared by perception, events, agent and UI.

Everything the agent reasons about is one of these records. They are plain
dataclasses (no pydantic dependency in the core) so the same code runs in a
bare Python environment, in a Colab runtime, and on a GPU workstation.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

@dataclass
class Detection:
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    track_id: int | None = None
    source: str = "detector"

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class PoseObservation:
    """17 COCO keypoints for one tracked person in one frame."""

    track_id: int
    keypoints: list[tuple[float, float, float]]
    box_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class FrameObservation:
    frame_index: int
    timestamp_s: float
    detections: list[Detection] = field(default_factory=list)
    poses: list[PoseObservation] = field(default_factory=list)
    motion_score: float = 0.0
    scene_change_score: float = 0.0
    triage_scores: dict[str, float] = field(default_factory=dict)
    image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 3),
            "detections": [d.to_dict() for d in self.detections],
            "poses": [p.to_dict() for p in self.poses],
            "motion_score": round(self.motion_score, 4),
            "scene_change_score": round(self.scene_change_score, 4),
            "triage_scores": {k: round(v, 4) for k, v in self.triage_scores.items()},
            "image_path": self.image_path,
        }

@dataclass
class AudioSegment:
    start_s: float
    end_s: float
    text: str
    language: str
    translation_en: str | None = None
    confidence: float = 0.0
    speaker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class AcousticEvent:
    start_s: float
    end_s: float
    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class TextRiskScores:
    """Output of the multilingual threat/hate classifiers on a transcript."""

    threat: float = 0.0
    hate: float = 0.0
    toxicity: float = 0.0
    self_harm: float = 0.0
    model: str = "none"
    matched_cues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

EventKind = Literal[
    "fall",
    "person_down",
    "slump_unresponsive",
    "distress_speech",
    "weapon_visible",
    "aggressive_interaction",
    "threatening_speech",
    "hateful_speech",
    "acoustic_alarm",
    "object_removed",
    "scene_change",
    "normal_activity",
]

@dataclass
class Evidence:
    evidence_id: str
    kind: str
    start_s: float
    end_s: float
    description: str
    path: str | None = None
    score: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class Event:
    event_id: str
    kind: str
    start_s: float
    end_s: float
    subject: str
    action: str
    obj: str | None
    location: str
    confidence: float
    severity: float
    summary: str
    evidence_ids: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ReasoningStep:
    """One visible ReAct step. `thought` is the agent's own reasoning summary
    (shown in the UI), `action` is the tool it chose, `observation` the result."""

    step: int
    agent: str
    thought: str
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str | None = None
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class Hypothesis:
    statement: str
    kind: str
    probability: float
    evidence_ids: list[str]
    alternatives: list[str] = field(default_factory=list)
    critic_verdict: str | None = None
    critic_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ActionDecision:
    action: str
    rationale: str
    risk_if_wrong: float
    expected_benefit: float
    requires_confirmation: bool
    executed: bool
    simulated: bool
    result: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class UserQuestion:
    question_id: str
    text: str
    language: str
    options: list[str]
    timeout_s: float
    default_if_silent: str
    asked_at: float = field(default_factory=time.time)
    answer: str | None = None
    answered_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class InvestigationState:
    """The master agent's working memory (blackboard)."""

    run_id: str
    objective: str
    scenario_hint: str | None
    media_path: str
    media_kind: str
    events: list[Event] = field(default_factory=list)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    steps: list[ReasoningStep] = field(default_factory=list)
    questions: list[UserQuestion] = field(default_factory=list)
    decisions: list[ActionDecision] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    compute_spent: float = 0.0
    compute_budget: float = 100.0
    uncertainty: float = 1.0
    risk: float = 0.0
    final_answer: str | None = None
    final_headline: str | None = None
    final_uncertainty: str | None = None
    status: str = "running"

    def add_step(self, step: ReasoningStep) -> ReasoningStep:
        step.step = len(self.steps) + 1
        self.steps.append(step)
        return step

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "scenario_hint": self.scenario_hint,
            "media_path": self.media_path,
            "media_kind": self.media_kind,
            "events": [e.to_dict() for e in self.events],
            "evidence": {k: v.to_dict() for k, v in self.evidence.items()},
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "steps": [s.to_dict() for s in self.steps],
            "questions": [q.to_dict() for q in self.questions],
            "decisions": [d.to_dict() for d in self.decisions],
            "retrieved": self.retrieved,
            "compute_spent": self.compute_spent,
            "compute_budget": self.compute_budget,
            "uncertainty": self.uncertainty,
            "risk": self.risk,
            "final_answer": self.final_answer,
            "final_headline": self.final_headline,
            "final_uncertainty": self.final_uncertainty,
            "status": self.status,
        }
