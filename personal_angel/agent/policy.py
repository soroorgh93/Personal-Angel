"""Risk-aware adaptive investigation policy.

The master agent (LLM) proposes; this module decides whether the proposal is
*justified* given belief, severity, harm asymmetry and compute budget, and
recommends what to do next when it is not. It is deliberately a transparent
decision rule (not RL) so every gate can be printed in the UI:

    expected_harm_if_ignored   = p · severity
    expected_harm_if_wrong     = (1 − p) · action_cost
    justified(a)  ⇔  p ≥ threshold(a)  and  p·severity ≥ (1 − p)·cost(a)

    value_of_information(tool) ≈ H(p) · availability · (1 − spent/budget)
    ask_user  ⇔  a human can answer, asking is not itself dangerous,
                 and p lies in the ambiguous band (or consent is required)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

KIND_PROFILE: dict[str, dict[str, Any]] = {
    "fall": {"severity": 0.75, "human_can_answer": True, "ask_safe": True, "consent_needed": False,
             "escalation": ["call_911", "notify_owner"], "check_question": {
                 "en": "I noticed you may have fallen. Are you okay? Please answer or move if you can hear me.",
                 "es": "Parece que se ha caído. ¿Está bien? Responda o muévase si me escucha."},
             "silent_default": "no_response"},
    "person_down": {"severity": 0.6, "human_can_answer": True, "ask_safe": True, "consent_needed": False,
                    "escalation": ["notify_owner"], "check_question": {
                        "en": "I see someone lying on the floor. Are you okay?", "es": "Veo a alguien en el suelo. ¿Está bien?"},
                    "silent_default": "no_response"},
    "slump_unresponsive": {"severity": 0.7, "human_can_answer": True, "ask_safe": True, "consent_needed": True,
                           "escalation": ["reroute_to_hospital", "call_911"], "check_question": {
                               "en": "You seem unwell. On a scale from 0 to 10, how bad is your pain? Do you want me to take you to the nearest emergency room?",
                               "es": "Parece que no se siente bien. Del 0 al 10, ¿cuánto le duele? ¿Quiere que le lleve a la sala de emergencias más cercana?"},
                           "silent_default": "no_response"},
    "distress_speech": {"severity": 0.65, "human_can_answer": True, "ask_safe": True, "consent_needed": True,
                        "escalation": ["reroute_to_hospital", "call_911"], "check_question": {
                            "en": "I heard you say you are not feeling well. On a scale from 0 to 10, how bad is your pain? Do you want me to drive you to the nearest emergency room?",
                            "es": "Le escuché decir que no se siente bien. Del 0 al 10, ¿cuánto le duele? ¿Quiere que le lleve a la sala de emergencias más cercana?"},
                        "silent_default": "no_response"},
    "weapon_visible": {"severity": 0.9, "human_can_answer": False, "ask_safe": False, "consent_needed": False,
                       "escalation": ["call_police_share_location", "notify_security"], "silent_default": None},
    "aggressive_interaction": {"severity": 0.85, "human_can_answer": False, "ask_safe": False, "consent_needed": False,
                               "escalation": ["notify_parents", "notify_security"], "silent_default": None},
    "threatening_speech": {"severity": 0.7, "human_can_answer": False, "ask_safe": False, "consent_needed": False,
                           "escalation": ["call_police_share_location", "notify_security"], "silent_default": None},
    "hateful_speech": {"severity": 0.45, "human_can_answer": False, "ask_safe": False, "consent_needed": False,
                       "escalation": ["notify_owner"], "silent_default": None},

    "abusive_speech": {"severity": 0.45, "human_can_answer": True, "ask_safe": True, "consent_needed": True,
                       "escalation": ["notify_owner"], "check_question": {
                           "en": "This recording contains abusive language directed at you. Do you want me to report it (to security / HR / the platform) with the transcript?",
                           "es": "Esta grabación contiene lenguaje abusivo dirigido a usted. ¿Quiere que lo reporte (a seguridad / recursos humanos / la plataforma) con la transcripción?"},
                       "silent_default": "no_response"},
    "infant_distress": {"severity": 0.6, "human_can_answer": False, "ask_safe": False, "consent_needed": False,
                        "escalation": ["notify_parents", "notify_owner"], "silent_default": None},
    "acoustic_alarm": {"severity": 0.6, "human_can_answer": True, "ask_safe": True, "consent_needed": False,
                       "escalation": ["notify_security"], "check_question": {
                           "en": "I heard an alarming sound. Is everyone okay?", "es": "Escuché un sonido alarmante. ¿Están todos bien?"},
                       "silent_default": "no_response"},
    "normal_activity": {"severity": 0.0, "human_can_answer": False, "ask_safe": True, "consent_needed": False,
                        "escalation": ["log_only"], "silent_default": None},
}
INFO_TOOLS = ("run_vlm", "inspect_frames", "run_pose_analysis", "run_audio_analysis", "query_memory", "retrieve_policy")

def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def entropy(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

def update_belief(p: float, support: float, weight: float = 1.0) -> float:
    """support ∈ [-1, 1]: -1 refutes, +1 confirms."""
    return sigmoid(logit(p) + weight * 2.0 * support)

@dataclass
class GateResult:
    allowed: bool
    reason: str
    requires_confirmation: bool = False
    suggestion: str | None = None

class InvestigationPolicy:
    def __init__(self, config: dict[str, Any], agent_config: dict[str, Any]) -> None:
        self.escalate_threshold = float(config.get("escalate_threshold", 0.7))
        self.ask_threshold = float(config.get("ask_threshold", 0.4))
        self.info_gain_min = float(config.get("info_gain_min", 0.08))
        self.action_cost = {k: float(v) for k, v in config.get("action_cost", {}).items()}
        self.allowlist = set(config.get("allowlist", []))
        self.simulate = bool(config.get("simulate_all_external_actions", True))
        self.ask_enabled = bool(agent_config.get("ask_user_enabled", True))
        self.require_critic = bool(config.get("require_critic", True)) and bool(agent_config.get("critic_enabled", True))
        self.review_threshold = float(config.get("review_threshold", 0.25))

    @staticmethod
    def profile(kind: str) -> dict[str, Any]:
        return KIND_PROFILE.get(kind, KIND_PROFILE["normal_activity"])

    def value_of_information(self, p: float, unused_tools: int, spent: float, budget: float) -> float:
        return entropy(p) * min(unused_tools, 3) / 3 * max(0.0, 1 - spent / max(budget, 1e-6))

    def should_ask_user(self, kind: str, p: float, asked_before: bool, user_answer: str | None,
                        user_present: bool = True, can_answer: bool = True) -> tuple[bool, str]:
        prof = self.profile(kind)
        if not self.ask_enabled:
            return False, "ask_user disabled in profile"
        if not can_answer:
            return False, "the person is an infant/toddler and cannot answer a question — a caregiver must be told instead"
        if not prof["ask_safe"]:
            return False, f"asking is unsafe for {kind} (could escalate the situation / tip off the subject)"
        if not prof["human_can_answer"] or not user_present:
            return False, "no human is expected to answer"
        if asked_before:
            return False, "already asked" + (f"; answer = {user_answer}" if user_answer else "; no answer yet")
        if prof["consent_needed"] and p >= self.ask_threshold:
            return True, "consent is required before a medical reroute unless the person is unresponsive"
        if self.ask_threshold <= p < self.escalate_threshold:
            return True, f"belief {p:.2f} is in the ambiguous band [{self.ask_threshold}, {self.escalate_threshold})"
        if p >= self.escalate_threshold and prof["human_can_answer"]:
            return True, "confirm responsiveness before escalating (cheap, respectful, high information)"
        return False, f"belief {p:.2f} below the ask threshold"

    def gate_action(self, action: str, kind: str, p: float, severity: float, critic_verdict: str | None,
                    user_answer: str | None, budget_left: float) -> GateResult:
        if action not in self.allowlist:
            return GateResult(False, f"'{action}' is not in the action allowlist")
        if action in {"log_only", "continue_route", "advise_user"}:
            return GateResult(True, "non-consequential action")
        if action == "ask_user":
            return GateResult(True, "question to a present human is always allowed when safe")
        cost = self.action_cost.get(action, 0.5)
        prof = self.profile(kind)
        if critic_verdict == "refuted" and cost >= 0.2:
            return GateResult(False, "the critic refuted the hypothesis; only a low-cost human-review notification is allowed",
                              suggestion="notify_security")
        if prof["consent_needed"] and action in {"reroute_to_hospital"}:
            if user_answer is None:
                return GateResult(False, "consent required: ask the person first (unless unresponsive)", suggestion="ask_user")
            if user_answer == "declines_help":
                return GateResult(False, "the person declined; respect their decision and keep monitoring",
                                  suggestion="continue_route")
        harm_ignored = p * severity
        harm_wrong = (1 - p) * cost
        threshold = self.escalate_threshold if cost >= 0.3 else (self.ask_threshold if cost >= 0.2 else self.review_threshold)
        if user_answer == "no_response":
            threshold -= 0.15
        if user_answer == "moderate" and action == "call_911":
            return GateResult(False, "the person rates the pain as moderate (4-6/10): reroute to the nearest hospital, no 911 call unless they worsen",
                              suggestion="reroute_to_hospital")
        if user_answer == "declines_help" and action in {"call_911", "call_police_share_location", "reroute_to_hospital"}:
            return GateResult(False, "the person says they are okay; an emergency call is not justified — notify a human to follow up",
                              suggestion="notify_owner")
        if p < threshold:
            return GateResult(False, f"belief {p:.2f} < threshold {threshold:.2f} for '{action}' (cost {cost:.2f})",
                              suggestion="gather more evidence (run_vlm / ask_user) or choose a lower-cost action")
        if harm_ignored < harm_wrong:
            return GateResult(False, f"expected harm if wrong ({harm_wrong:.2f}) exceeds harm if ignored ({harm_ignored:.2f})",
                              suggestion="choose a lower-cost action such as notify_security / notify_owner")
        if cost >= 0.5 and critic_verdict is None and self.require_critic:
            return GateResult(False, "high-cost action requires the critic to have reviewed the hypothesis first",
                              suggestion="run_critic")
        return GateResult(True, f"p={p:.2f} ≥ {threshold:.2f}; harm-if-ignored {harm_ignored:.2f} ≥ harm-if-wrong {harm_wrong:.2f}",
                          requires_confirmation=cost >= 0.5 and not self.simulate)

    def recommend(self, kind: str, p: float, user_answer: str | None) -> list[str]:
        prof = self.profile(kind)
        if kind == "normal_activity":
            return ["log_only"]
        if kind == "abusive_speech":
            if user_answer == "needs_help":
                return ["notify_owner"]
            if user_answer in {"declines_help", "no_response", "unclear"}:
                return ["advise_user"]
            return ["log_only"] if p < self.ask_threshold else ["notify_owner"]
        if user_answer == "declines_help":
            return ["continue_route"] if prof["consent_needed"] else ["notify_owner"]
        if user_answer == "moderate":

            return ["reroute_to_hospital", "notify_owner"] if prof["consent_needed"] else ["notify_owner"]
        if p < self.ask_threshold:

            if p >= self.review_threshold and prof["severity"] >= 0.6:
                return ["notify_security"] if kind != "infant_distress" else ["notify_owner"]
            return ["log_only"]
        actions = list(prof["escalation"])
        if kind == "aggressive_interaction" and p >= 0.95:
            actions.append("call_911")
        return actions
