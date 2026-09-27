"""Master agent: a visible ReAct loop (Thought → Action → Observation) over
the tool registry, with a heuristic planner that (a) drives the fixture
profile, (b) advises the LLM each turn, and (c) takes over if the model emits
unparsable output twice in a row.

`investigate()` is a generator that yields stream events for the UI/CLI.
"""
from __future__ import annotations

import json
import re
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from ..events.builder import build_events
from ..events.graph import EventGraph
from ..memory.store import MemoryStore
from ..perception.fixtures import load_fixture
from ..perception.pipeline import PerceptionResult, run_perception
from ..schema import Hypothesis, InvestigationState, ReasoningStep
from .actions import ActionExecutor
from .llm import LLMClient, extract_json_object
from .policy import InvestigationPolicy, update_belief
from .safety import SafetyGuard
from .tools import TOOL_SPECS, AnswerProvider, ToolBox, ToolResult, interpret_answer, normalize_answer

log = logging.getLogger(__name__)

MASTER_SYSTEM = """You are the MASTER agent of PersonalAngel, an edge-native autonomous safety investigator running fully
locally on this device. You investigate structured events extracted from video/audio by perception models and decide,
step by step, what to do. You think out loud in short, concrete sentences (these thoughts are shown to the operator),
then choose exactly ONE tool per turn.

Principles
1. Evidence first: cite evidence ids; never claim identity, ownership, intent, diagnosis or legality as fact.
2. Spend compute only when it changes the decision: cheap tools before the VLM; the VLM only to verify the primary hypothesis.
3. Separate observation → interpretation → recommendation → (simulated) execution.
4. Ask the person when it is safe and informative (medical distress, falls). Never ask when it could tip off or escalate
   (weapons, aggression toward a child, threats).
5. High-cost actions (police, 911, reroute) require the critic's review and a belief above the policy threshold; the
   policy gate will reject otherwise — read its reason and gather what is missing.
6. Speak to a person in their own language; translate foreign speech to English for the operator. Your own reasoning
   and the final report are always in English.
7. The environment (home, nursery, vehicle cabin, shop, street...) and who is present (baby, child, adult, elderly)
   were INFERRED from the pixels by scene understanding — reason with that context; do not assume a different one.
8. Never call a tool again with the same input: the result will not change. Each tool result is already in
   PREVIOUS STEPS. Memory hits are past, unrelated recordings — context only, never evidence.
9. Stop when the belief is settled: after the VLM and/or critic refute the hypothesis (belief < 0.15) or support it
   (belief > 0.85 with critic review), do not keep inspecting — propose the policy-justified action or finalize.
10. Finish with finalize: a specific, evidence-grounded answer (what happened, when, who, what was verified, what was
   decided), hypothesis, probability, uncertainty, recommended/executed action. Never end without a verdict.
11. A fight is judged as an incident, not by blame: several striking motions plus people going to the ground is a brawl
   even when tracker ids change every few frames or one strike shows "nobody within reach". Ask the vision model
   whether people are physically fighting in the scene, never whether two specific ids are; "the named person is the
   victim" confirms the fight rather than refuting it. Only play, dance, sport or greeting make it benign.

Output format (STRICT JSON, one object, no prose outside it):
{"thought": "<2-4 sentences of reasoning>", "action": "<tool name>", "action_input": {<arguments>}}
"""

FINAL_SYSTEM = """You write the closing report of an on-device safety investigation for a human operator (security desk,
fleet supervisor, parent). Plain, confident, specific English — ENGLISH ONLY in every field, never Chinese or any other
language, even when the evidence contains foreign speech (quote its English translation). Use PERSON_xx labels, seconds, and the names of the
checks that were run (detector, pose geometry, fallen-person detector, activity classifier, scene understanding,
audio/transcript, vision model, critic, the question asked and the answer). Say plainly when an initial alarm was a
false positive and what showed it. Never invent facts that are not in the material; never change the action.
Reply with STRICT JSON:
{"headline": "<max 10 words, e.g. 'Fall confirmed - 911 called' or 'Toy mistaken for gun - no incident'>",
 "what_happened": "<1-2 sentences: who, what, when>",
 "checks": ["<3-6 short items: check -> finding>"],
 "why": "<1-2 sentences: belief vs policy threshold, harm asymmetry, consent/answer>",
 "action": "<1 sentence: what was done (simulated) or 'no action needed'>",
 "uncertainty": "<1 sentence>"}"""

def compose_report(headline: str, what: str, checks: list[str], why: str, action: str, uncertainty: str = "") -> str:
    lines = [f"WHAT HAPPENED — {what.strip()}"]
    if checks:
        lines.append("WHAT WAS CHECKED — " + " · ".join(c.strip().rstrip(".") for c in checks if c and c.strip()))
    lines.append(f"WHY THIS DECISION — {why.strip()}")
    lines.append(f"ACTION — {action.strip()}")
    return "\n".join(lines)

HIGH_COST_ACTIONS = {"call_911", "call_police_share_location", "reroute_to_hospital", "notify_parents"}

def _tool_manual() -> str:
    lines = []
    for spec in TOOL_SPECS:
        lines.append(f"- {spec['name']}: {spec['description']} Input: {json.dumps(spec['input'])}")
    return "\n".join(lines)

def initial_belief(state: InvestigationState) -> tuple[Hypothesis | None, Any]:
    events = [e for e in state.events if e.kind != "normal_activity"]
    if not events:
        e = state.events[0] if state.events else None
        if e is None:
            return None, None
        return Hypothesis("Nothing safety-relevant happened.", "normal_activity", 0.75, e.evidence_ids), e
    primary = max(events, key=lambda e: e.severity * 0.6 + e.confidence * 0.4)
    p = min(0.9, max(0.05, 0.1 + 0.75 * primary.confidence))
    corroborating = {
        "slump_unresponsive": {"distress_speech", "acoustic_alarm"}, "distress_speech": {"slump_unresponsive"},
        "fall": {"acoustic_alarm", "person_down"}, "person_down": {"fall"},
        "weapon_visible": {"threatening_speech", "aggressive_interaction"},
        "aggressive_interaction": {"acoustic_alarm", "weapon_visible"},
        "infant_distress": {"acoustic_alarm", "person_down"},
        "threatening_speech": {"weapon_visible", "hateful_speech"},
    }
    ev_ids = list(primary.evidence_ids)
    for other in events:
        if other is primary:
            continue
        if other.kind in corroborating.get(primary.kind, set()):
            p = 1 - (1 - p) * (1 - 0.5 * other.confidence)
            ev_ids += other.evidence_ids
    age = primary.attributes.get("age_group") or primary.attributes.get("holder_age_group")
    statement = {
        "fall": f"{primary.subject} fell and may be injured or unable to get up",
        "person_down": (f"{primary.subject} (looks like a {age}) is on the floor — probably crawling or playing, verify"
                        if age == "baby" else f"{primary.subject} is lying on the ground and may need help"),
        "slump_unresponsive": f"{primary.subject} (passenger) is slumped and possibly unresponsive — medical distress",
        "distress_speech": f"{primary.subject} is reporting medical distress",
        "weapon_visible": (f"{primary.subject} is carrying a visible {primary.attributes.get('label', 'weapon')}"
                           + (f" (second opinion: looks like a {primary.attributes['second_opinion']})"
                              if primary.attributes.get("second_opinion") not in (None, "weapon", "uncertain") else "")),
        "aggressive_interaction": (f"Several people are physically fighting (brawl involving "
                                   f"{', '.join((primary.attributes.get('strikers') or [primary.subject])[:4])}) — roles unknown"
                                   if primary.attributes.get("group_fight") else
                                   f"{primary.subject} is physically aggressive toward {primary.obj}"),
        "threatening_speech": f"{primary.subject} made an explicit threat of violence",
        "hateful_speech": f"{primary.subject} used hateful language",
        "abusive_speech": f"{primary.subject} is verbally abusing the person on the recording (no threat to life)",
        "acoustic_alarm": f"An alarming sound ({primary.obj}) indicates an incident",
        "infant_distress": f"{primary.subject} (baby) is crying and needs a caregiver",
    }.get(primary.kind, primary.summary)
    return Hypothesis(statement, primary.kind, round(p, 3), list(dict.fromkeys(ev_ids))), primary

class HeuristicPlanner:
    """Deterministic ReAct policy: usable stand-alone (no LLM) and as an advisor."""

    def __init__(self, policy: InvestigationPolicy, config: dict[str, Any]) -> None:
        self.policy = policy
        self.critic_min_risk = float(config.get("agent", {}).get("critic_min_risk", 0.35))
        self.skip_tools = set(config.get("agent", {}).get("skip_tools", []))
        if not bool(config.get("agent", {}).get("critic_enabled", True)):
            self.skip_tools.add("run_critic")

    def next(self, state: InvestigationState, box: ToolBox, done: set[str]) -> dict[str, Any]:
        done = done | self.skip_tools
        primary = box.primary_event()
        hyp = state.hypotheses[-1] if state.hypotheses else None
        p = hyp.probability if hyp else 0.5
        kind = primary.kind if primary else "normal_activity"
        perception = box.perception
        has_frames = any(e.kind == "frame" for e in state.evidence.values())
        if "inspect_frames" not in done and has_frames:
            start = max(0.0, (primary.start_s if primary else 0.0) - 2.0)
            end = min(perception.duration_s, (primary.end_s if primary else perception.duration_s) + 2.0)
            if kind == "normal_activity":
                thought = (f"Perception found no fall, weapon, aggression or distress in {perception.duration_s:.0f}s of video "
                           f"({int(perception.counters.get('frames_scanned', 0))} sampled frames). I will still inspect the tracked activity once to make sure nothing was missed.")
            else:
                thought = (f"Perception flagged '{kind}' ({primary.subject if primary else '-'}) between {start:.1f}s and {end:.1f}s "
                           f"with belief {p:.2f}. Before spending on the VLM I will inspect the cheap detector/pose evidence in that window.")
            return {"thought": thought, "action": "inspect_frames", "action_input": {"start_s": round(start, 1), "end_s": round(end, 1)}}
        if "run_audio_analysis" not in done and (perception.audio_segments or perception.acoustic_events):
            return {"thought": "There is speech or a salient sound. I will read the transcript, its English translation and the text-risk scores before interpreting the visual evidence.",
                    "action": "run_audio_analysis", "action_input": {}}
        if kind == "normal_activity":
            return {"thought": "No fall, weapon, aggression, distress or threat was detected and nothing in the evidence contradicts that. Spending VLM compute would not change the decision; I will close with a log-only outcome.",
                    "action": "finalize", "action_input": {"answer": "Nothing safety-relevant happened in this recording.",
                                                           "hypothesis": "normal activity", "probability": 0.8,
                                                           "uncertainty": "Only sampled frames were inspected.", "recommended_action": "log_only"}}
        if "run_vlm" not in done and has_frames and kind not in {"threatening_speech", "hateful_speech", "abusive_speech"}:
            age = (primary.attributes.get("age_group") or primary.attributes.get("holder_age_group")) if primary else None
            question = {
                "fall": "Did this person fall to the ground involuntarily (vs. lying down on purpose)? Do they look injured or unresponsive?",
                "person_down": ("Is this baby/toddler crawling, sitting or playing normally, or does it look hurt, stuck or in distress? Is an adult present?"
                                if age == "baby" else "Is the person on the ground in distress or resting intentionally?"),
                "slump_unresponsive": "Is the seated passenger slumped, eyes closed or in visible pain, versus simply relaxed or sleeping?",
                "distress_speech": "Does the passenger look unwell (pain, pallor, holding abdomen/chest)?",
                "weapon_visible": "Is the highlighted object a real knife/firearm (not a phone, wallet or toy)? Is it held or partially concealed?",
                "aggressive_interaction": "Is the adult striking, shaking or roughly handling the smaller person, versus play or care?",
                "acoustic_alarm": "Is there any visible sign of the incident suggested by the sound?",
                "infant_distress": "Is the baby crying or in distress (red face, open mouth, thrashing)? Is any adult attending to it? Any sign of injury, choking or an unsafe position?",
            }.get(kind, "What is happening?")
            ids = [i for i in (hyp.evidence_ids if hyp else []) if i in state.evidence and state.evidence[i].kind == "frame"][:6]
            return {"thought": f"Belief is {p:.2f}; the detector/pose signal alone could be a false positive. A single VLM verification on the {len(ids)} PageRank-ranked frames is worth its cost here.",
                    "action": "run_vlm", "action_input": {"evidence_ids": ids, "question": question}}
        if "retrieve_policy" not in done:
            query = {
                "fall": "fall of resident on the ground ask if okay emergency services family contact",
                "person_down": "person lying on the ground ask if okay",
                "slump_unresponsive": "passenger medical distress unresponsive ask emergency room reroute call",
                "distress_speech": "passenger reports feeling unwell ask emergency room reroute",
                "weapon_visible": "visible firearm knife weapon police live location do not confront",
                "aggressive_interaction": "caregiver striking shaking child notify parents emergency",
                "threatening_speech": "recorded threat of violence translate report police advise target",
                "hateful_speech": "hateful speech owner manager review",
                "acoustic_alarm": "alarming sound scream gunshot security",
                "infant_distress": "baby crying unattended notify parents caregiver nursery",
                "abusive_speech": "harassment abusive language report consent advice hotline",
            }.get(kind, f"{kind} escalation")
            return {"thought": f"Before deciding I will check the operator's policy for '{kind}' at {primary.location if primary else 'this site'}.",
                    "action": "retrieve_policy", "action_input": {"query": query}}
        if "query_memory" not in done:
            return {"thought": "I will also check episodic memory for prior incidents at this location and strategies that worked.",
                    "action": "query_memory", "action_input": {"query": f"{kind} at {primary.location if primary else 'site'}"}}
        if "run_critic" not in done and primary and primary.severity >= self.critic_min_risk:
            return {"thought": f"Current belief {p:.2f}. Any escalation must survive the critic first; I will ask it to find the strongest benign explanation.",
                    "action": "run_critic", "action_input": {}}
        asked = bool(state.questions)
        answer = interpret_answer(kind, normalize_answer(state.questions[-1].answer)) if asked and state.questions[-1].answer else None
        infant = box.subject_is_infant(primary)
        should_ask, why = self.policy.should_ask_user(kind, p, asked, answer, can_answer=not infant)
        armed = [e for e in state.events if e.kind == "weapon_visible" and e.confidence >= 0.5]
        if should_ask and armed:
            should_ask, why = False, f"a weapon is in view ({armed[0].attributes.get('label', 'weapon')}); a question could tip off or escalate — no interaction"
        if infant and kind in {"person_down", "fall"} and not asked and "run_vlm" in done:

            recommended = ["notify_parents"] if p >= 0.35 else ["log_only"]
            for action in recommended:
                if action in done or any(d.action == action for d in state.decisions):
                    continue
                rationale = (f"The person on the floor is an infant (belief {p:.2f} that something is wrong). An infant cannot answer a "
                             f"question, so I will not ask one; {'the family/caregiver must be told now' if action == 'notify_parents' else 'nothing suggests distress, so I only log it'}.")
                return {"thought": rationale, "action": "propose_action",
                        "action_input": {"action": action, "rationale": rationale, "params": self._params(kind, action, primary, state)}}
        if should_ask:
            prof = self.policy.profile(kind)
            lang = box.language if box.language in {"en", "es"} else "en"
            q = prof.get("check_question", {}).get(lang) or prof.get("check_question", {}).get("en")
            return {"thought": f"Policy: {why}. Asking is safe and cheap here and the answer changes the action, so I will ask in {lang}.",
                    "action": "ask_user", "action_input": {"question": q, "language": lang, "options": ["yes", "no", "no_response"]}}
        recommended = self.policy.recommend(kind, p, answer)
        if kind == "aggressive_interaction":
            scene = getattr(box.perception, "scene", None)
            child_involved = (primary is not None and (primary.attributes.get("crying_heard") or "NURSERY" in (primary.location or "")))
            if scene and not child_involved:

                involved = {primary.subject, primary.obj, *(primary.attributes.get("strikers") or [])} if primary else set()
                for pp in scene.people:
                    tag = f"PERSON_{pp.track_id:02d}" if pp.track_id is not None else None
                    if pp.age_group in {"baby", "child"} and pp.confidence >= 0.6 and (tag in involved or (scene.location in {"NURSERY", "HOME_ROOM"})):
                        child_involved = True
                        break
            if not child_involved:

                recommended = [("call_police_share_location" if a == "notify_parents" else a) for a in recommended]
                recommended = [a for i, a in enumerate(recommended) if a not in recommended[:i]]
        for action in recommended:
            if action in done:
                continue
            if any(d.action == action for d in state.decisions):
                continue
            rationale = self._rationale(kind, action, p, answer, primary)
            return {"thought": rationale, "action": "propose_action",
                    "action_input": {"action": action, "rationale": rationale, "params": self._params(kind, action, primary, state)}}
        return self.finalize_now(state, box, "All justified actions have been taken (simulated) and the hypothesis has been reviewed. I will summarize for the operator.")

    def finalize_now(self, state: InvestigationState, box: ToolBox, thought: str | None = None) -> dict[str, Any]:
        """A complete, evidence-grounded finalize decision from the current state (used at the natural end of the
        investigation and when the step budget forces closure — the run never ends without a verdict)."""
        primary = box.primary_event()
        hyp = state.hypotheses[-1] if state.hypotheses else None
        p = hyp.probability if hyp else 0.5
        kind = primary.kind if primary else "normal_activity"
        executed = [d.action for d in state.decisions if d.executed]
        final_action = executed[-1] if executed else ("log_only" if p < self.policy.ask_threshold or kind == "normal_activity"
                                                       else (self.policy.recommend(kind, p, None) or ["log_only"])[0])
        thought = thought or ("The step budget is nearly exhausted. The evidence gathered so far is enough for a verdict; "
                              "I will close with what was verified, what was not, and the policy-justified action.")
        return {"thought": thought, "action": "finalize", "action_input": {
            "answer": self._answer(kind, p, primary, executed, state),
            "hypothesis": hyp.statement if hyp else "", "probability": round(p, 2),
            "uncertainty": self._uncertainty(kind, hyp), "recommended_action": final_action}}

    @staticmethod
    def _rationale(kind: str, action: str, p: float, answer: str | None, primary) -> str:
        if kind in {"slump_unresponsive", "distress_speech"}:
            if answer == "declines_help":
                return "The passenger said they are fine (pain 0-3/10 or 'I'm okay'). Respecting that: continue the route but keep monitoring at a higher rate."
            if answer == "moderate":
                return (f"The passenger rates the pain as moderate (4-6/10). Belief {p:.2f} in medical distress: {action.replace('_', ' ')} — "
                        "the nearest hospital is the right destination, an emergency call is not (yet) justified.")
            if answer in {"no_response", "needs_help", "unclear"}:
                who = "did not answer" if answer == "no_response" else "asked for help" if answer == "needs_help" else "gave an unclear answer"
                return f"The passenger {who}. With belief {p:.2f} in medical distress the harm of ignoring outweighs a false alarm: {action.replace('_', ' ')}."
        if kind in {"fall", "person_down", "acoustic_alarm"}:
            if answer == "no_response":
                return f"No response after the check. Belief {p:.2f}: {action.replace('_', ' ')} is justified (harm of a missed injury ≫ cost of a false call)."
            if answer == "declines_help":
                return "The person responded that they are okay; notify the owner/caregiver so a human follows up, no emergency call."
            if answer in {"needs_help", "unclear"}:
                return f"The person {'asked for help' if answer == 'needs_help' else 'answered unclearly'} after the fall: {action.replace('_', ' ')}."
        if kind == "weapon_visible":
            return (f"A {primary.attributes.get('label', 'weapon') if primary else 'weapon'} is visible with belief {p:.2f}. Asking would tip the person off, "
                    f"so no interaction: {action.replace('_', ' ')} with the evidence frames and do not confront.")
        if kind == "aggressive_interaction":
            if primary and primary.attributes.get("group_fight"):
                who = (f"several people are physically fighting ({primary.attributes.get('strike_events', 0)} striking motions"
                       + (f", {primary.attributes.get('falls_during')} person(s) knocked down" if primary.attributes.get("falls_during") else "") + ")")
            else:
                who = "a caregiver is physically aggressive toward a child" if (primary and (primary.attributes.get("crying_heard") or "NURSERY" in (primary.location or ""))) else "people are physically fighting"
            return f"Belief {p:.2f} that {who}. This is not a situation to negotiate with: {action.replace('_', ' ')} with the evidence clip."
        if kind == "threatening_speech":
            return f"An explicit threat was made (belief {p:.2f}); the translated transcript and audio are evidence. {action.replace('_', ' ')} and advise the target not to engage."
        if kind == "hateful_speech":
            return "Hateful language is a moderation/HR matter, not an emergency: notify the owner with the transcript."
        if kind == "abusive_speech":
            if answer == "needs_help":
                return "The person asked me to report it: notify the owner/HR/security with the transcript and the recording as evidence."
            if answer in {"declines_help", "no_response", "unclear"}:
                return ("The person does not want a report (or did not answer). No threat to life was made, so I will not escalate; "
                        "instead I give practical advice: keep the recording, do not engage, block the contact, tell a trusted person, "
                        "and call 911 if it turns into a threat.")
            return f"Abusive language without a threat to life (belief {p:.2f}): the person decides whether to report it."
        if kind == "threatening_speech" and action in {"call_police_share_location", "call_911"}:
            return (f"An explicit threat to someone's life or safety was made (belief {p:.2f}). No communication with the caller: "
                    f"{action.replace('_', ' ')} with the transcript, the translation and the audio as evidence, and advise the target not to respond.")
        if kind == "infant_distress":
            return (f"A baby is crying (belief {p:.2f}) and cannot answer a question. This is not an emergency call, but a caregiver "
                    f"must know now: {action.replace('_', ' ')} with the clip.")
        if action == "notify_security" and p < 0.5:
            return (f"Belief {p:.2f} for {kind.replace('_', ' ')} is too low for an emergency call but too high to ignore: "
                    f"ask on-site security / a human operator to look at the evidence clip now (low-cost review).")
        return f"Belief {p:.2f} for {kind}: {action.replace('_', ' ')}."

    @staticmethod
    def _params(kind: str, action: str, primary, state: InvestigationState) -> dict[str, Any]:
        summary = primary.summary if primary else ""
        if action == "call_911":
            return {"nature": f"{kind.replace('_', ' ')}: {summary[:120]}"}
        if action == "call_police_share_location":
            return {"report": summary[:240]}
        if action == "notify_parents":
            return {"contacts": ["parent_primary", "parent_secondary"], "message": summary[:200]}
        return {}

    @staticmethod
    def _answer(kind: str, p: float, primary, executed: list[str], state: InvestigationState) -> str:
        if primary is None:
            return "Nothing to report."
        answered = interpret_answer(kind, normalize_answer(state.questions[-1].answer)) if state.questions and state.questions[-1].answer else None
        slump_headline = ("The passenger showed signs of medical distress but responded that they are fine." if answered == "declines_help"
                          else "The passenger is in probable medical distress and did not respond." if answered == "no_response"
                          else "The passenger is in probable medical distress and accepted help (severe pain reported)." if answered == "needs_help"
                          else "The passenger reports moderate pain and is being taken to the nearest hospital." if answered == "moderate"
                          else "The passenger is in probable medical distress.")
        fall_headline = ("A person fell, stayed on the ground and did not respond." if answered == "no_response"
                         else "A person fell but responded that they are okay." if answered == "declines_help"
                         else "A person fell and asked for help." if answered == "needs_help" else "A person fell and stayed on the ground.")
        headline = {
            "fall": fall_headline, "person_down": "A person is lying on the ground.",
            "slump_unresponsive": slump_headline, "distress_speech": slump_headline,
            "weapon_visible": f"A visible {primary.attributes.get('label', 'weapon')} is being carried by {primary.subject}.",
            "aggressive_interaction": (f"Several people were physically fighting between {primary.start_s:.1f}s and {primary.end_s:.1f}s "
                                       f"({primary.attributes.get('strike_events', 0)} striking motions, {primary.attributes.get('falls_during', 0)} person(s) went down); who started it is not determinable."
                                       if primary.attributes.get("group_fight") else f"{primary.subject} was physically aggressive toward {primary.obj}."),
            "threatening_speech": "The recording contains an explicit threat of violence.",
            "hateful_speech": "The recording contains hateful speech.", "acoustic_alarm": "An alarming sound was detected.",
            "infant_distress": "A baby is crying and needs a caregiver.",
            "abusive_speech": ("The recording contains abusive language; the person asked for it to be reported." if answered == "needs_help"
                               else "The recording contains abusive language; the person chose not to report it, so advice was given."),
        }.get(kind, primary.summary)
        others = [e for e in state.events if e.kind not in {kind, "normal_activity"}]
        corroboration = ("; corroborated by " + ", ".join(f"{e.kind.replace('_', ' ')} ({e.start_s:.0f}s)" for e in others[:3])) if others else ""
        hyp = state.hypotheses[-1] if state.hypotheses else None
        verification = []
        for s in state.steps:
            if s.action == "run_vlm":
                verification.append("VLM verified the frames")
            if s.action == "run_critic" and hyp and hyp.critic_verdict:
                verification.append(f"critic: {hyp.critic_verdict}")
        q = ""
        if state.questions:
            qq = state.questions[-1]
            q = f" I asked ({qq.language}): \"{qq.text}\" → answer: \"{qq.answer}\"."
        from .actions import ACTION_CATALOG

        acts = ", ".join(ACTION_CATALOG.get(a, {}).get("label", a.replace("_", " ")) for a in executed) or "no external action needed"
        if "advise_user" in executed:
            advice = next((d.params.get("dispatch", {}).get("advice") for d in state.decisions if d.action == "advise_user" and d.executed), None)
            if advice:
                acts += " — advice given: " + " ".join(f"({i + 1}) {x}" for i, x in enumerate(advice[:4]))
        when = f" ({primary.start_s:.0f}-{primary.end_s:.0f}s, {primary.location.replace('_', ' ').lower()})"
        import re as _re

        def clean(text: str) -> str:
            return _re.sub(r"</?untrusted[^>]*>", "", text or "").strip()

        checks = [f"perception: {clean(primary.summary)[:160]}"]
        for s in state.steps:
            if s.action == "run_vlm" and s.observation:
                checks.append("vision model: " + clean(s.observation.split("| answer:")[-1].split("| supports")[0])[:160])
            if s.action == "run_critic" and hyp and hyp.critic_verdict:
                checks.append(f"critic: {hyp.critic_verdict}" + (f" ({hyp.critic_notes[:80]})" if hyp.critic_notes else ""))
            if s.action == "run_audio_analysis" and s.observation and "No speech" not in s.observation:
                checks.append("audio: " + clean(s.observation.splitlines()[0])[:140])
        if state.questions:
            qq = state.questions[-1]
            checks.append(f"asked ({qq.language}) \"{qq.text[:70]}\" -> answer: \"{qq.answer}\"")
        why = f"Belief {p:.0%} after {', '.join(verification) or 'perception only'}{corroboration}."
        if kind == "abusive_speech":
            why += (" Abusive language without a threat to life is not an emergency: the person on the receiving end decides. "
                    + ("They asked for a report, so it goes to the owner/HR/security with the transcript." if answered == "needs_help"
                       else "They declined a report (or did not answer), so the system gives safety advice instead of escalating; it would call 911 automatically if a threat to life appeared."))
        elif answered == "no_response":
            why += " Silence after a direct question is treated as unresponsive; missing a real emergency is far worse than a false call."
        elif answered == "declines_help":
            why += " The person answered that they are fine; an emergency call is not justified, a human follows up."
        elif answered == "moderate":
            why += " Moderate pain (4-6/10): hospital, not 911."
        elif kind in {"weapon_visible", "aggressive_interaction", "threatening_speech"} and executed:
            why += " Asking would tip the person off, so the policy escalates without any interaction."
        elif not executed:
            why += " Below the policy threshold for any external action."
        return compose_report(headline, f"{headline}{when}", checks, why, f"{acts}" + (" (simulated)" if executed else "") + ".")

    @staticmethod
    def _uncertainty(kind: str, hyp: Hypothesis | None) -> str:
        base = {
            "weapon_visible": "Only the visible part of the object was detected; concealed objects cannot be seen by an RGB camera; toy/replica cannot be excluded.",
            "fall": "Pose geometry cannot diagnose injury or consciousness.",
            "slump_unresponsive": "Sleep and fainting look alike to a camera; the question/answer is the discriminating evidence.",
            "distress_speech": "Self-report was transcribed by ASR; wording may be imperfect.",
            "aggressive_interaction": "Motion primitives cannot establish intent; play vs. abuse needs human review of the clip.",
            "threatening_speech": "A transcript cannot establish credibility or capability; sarcasm/quotation were checked by the critic.",
            "infant_distress": "Crying alone does not indicate injury; the caregiver decides. No audio = camera-only judgement.",
            "abusive_speech": "Tone and context are judged from a transcript; sarcasm or a joke between friends can look abusive.",
        }.get(kind, "Sampled frames only.")
        if hyp and hyp.critic_verdict:
            base += f" Critic: {hyp.critic_verdict}."
        return base

def investigate(media_path: str | Path, objective: str, config: dict[str, Any], run_dir: Path,
                llm: LLMClient, memory: MemoryStore, telemetry: Any, scenario_hint: str | None = None,
                answer_provider: AnswerProvider | None = None, edge_cloud=None) -> Iterator[dict[str, Any]]:
    """Generator: yields UI stream events and finally {'type': 'final', 'report': ...}."""
    media_path = Path(media_path)
    fixture = load_fixture(media_path)
    guard = SafetyGuard(config.get("security", {}), run_dir, Path(config.get("_project_root", ".")))
    guard.validate_media(media_path, config.get("security", {}))
    policy = InvestigationPolicy(config.get("policy", {}), config.get("agent", {}))
    executor = ActionExecutor(run_dir, bool(config.get("policy", {}).get("simulate_all_external_actions", True)))
    state = InvestigationState(run_id=run_dir.name, objective=objective, scenario_hint=scenario_hint,
                               media_path=str(media_path), media_kind="", compute_budget=float(config.get("agent", {}).get("compute_budget", 100)))
    stream: list[dict[str, Any]] = []

    def progress(stage: str, payload: dict[str, Any]) -> None:
        stream.append({"type": "stage", "stage": stage, **payload})

    yield {"type": "stage", "stage": "INGESTED", "detail": f"{media_path.name} accepted; provenance hash computed."}
    if edge_cloud is not None:
        edge_cloud.decide("perception(detector+pose+asr)", ["video", "audio"], latency_budget_ms=1000)
        edge_cloud.decide("vlm_verification", ["image"], latency_budget_ms=4000)
        edge_cloud.decide("llm_reasoning", ["transcript", "location"], latency_budget_ms=4000)
        edge_cloud.decide("memory_retrieval", ["transcript"], latency_budget_ms=200)
        edge_cloud.decide("model_download_and_updates", [], latency_budget_ms=None)
    with telemetry.span("perception"):
        perception = run_perception(media_path, config, run_dir, progress, scenario_hint, telemetry)
    for item in stream:
        yield item
    stream.clear()
    for k, v in perception.counters.items():
        telemetry.set(k, v)
    state.media_kind = perception.media_kind
    events, evidence = build_events(perception, run_dir)
    state.events, state.evidence = events, evidence
    graph = EventGraph(float(config.get("graph", {}).get("pagerank_alpha", 0.85)),
                       bool(config.get("graph", {}).get("pagerank_enabled", True))).build(events, evidence)
    yield {"type": "stage", "stage": "EVENTS", "detail": f"{len(events)} structured event(s), {len(evidence)} evidence record(s), graph {graph.g.number_of_nodes()} nodes"}
    yield {"type": "events", "events": [e.to_dict() for e in events], "evidence": {k: v.to_dict() for k, v in evidence.items()},
           "backends": perception.backends, "location": perception.location, "duration_s": perception.duration_s,
           "windows": [w.__dict__ for w in perception.windows], "scene": perception.scene.to_dict(),
           "clips": perception.clips, "language": None}
    hyp, primary = initial_belief(state)
    if hyp:
        state.hypotheses.append(hyp)
        state.risk = primary.severity if primary else 0.0
        yield {"type": "hypothesis", "hypothesis": hyp.to_dict(), "primary_event": primary.to_dict() if primary else None}

    box = ToolBox(state, perception, graph, memory, llm, policy, executor, guard, config, answer_provider, telemetry, fixture)
    planner = HeuristicPlanner(policy, config)
    agent_cfg = config.get("agent", {})
    max_steps = int(agent_cfg.get("max_steps", 12))
    mode = str(agent_cfg.get("planner", "auto"))
    if mode == "auto":
        mode = "llm" if llm.is_real else "heuristic"
    if not llm.is_real:
        mode = "heuristic"
    use_llm_planner = mode == "llm"
    llm_synthesis = llm.is_real and mode in {"llm", "hybrid"}
    done: set[str] = set()
    history: list[dict[str, Any]] = []
    signatures: dict[str, int] = {}
    failures = repeats = llm_calls = 0
    steps_taken = 0
    yield {"type": "stage", "stage": "INVESTIGATING",
           "detail": f"Master agent ({llm.name}, planner={mode}) starts the ReAct loop; {max_steps} steps, budget {state.compute_budget:.0f} units."}
    while steps_taken < max_steps:
        steps_left = max_steps - steps_taken
        t0 = time.perf_counter()
        decision: dict[str, Any] | None = None
        reasoning_text = ""
        tokens = (0, 0)
        advisor = planner.next(state, box, done)
        primary_now = box.primary_event()
        critic_pending = ("run_critic" not in done and "run_critic" not in planner.skip_tools and primary_now is not None
                          and primary_now.severity >= planner.critic_min_risk and primary_now.kind != "normal_activity")

        if steps_left <= 1 and advisor["action"] != "finalize":
            decision = planner.finalize_now(state, box)
        elif steps_left <= 2 and critic_pending and advisor["action"] not in {"finalize", "run_critic"}:
            decision = {"thought": "Only two steps remain and no critic review has happened yet. Any verdict must survive "
                                   "the adversarial critic first, so I run it now and finalize next.",
                        "action": "run_critic", "action_input": {}}
        if decision is None and use_llm_planner:
            messages = _build_messages(state, box, history, advisor, config, steps_left, done)
            queue: list[dict[str, Any]] = []

            def on_delta(channel: str, delta: str) -> None:
                if channel == "reasoning":
                    queue.append({"type": "thinking", "delta": delta})

            try:
                llm_calls += 1
                resp = llm.chat(messages, stream=on_delta if config.get("llm", {}).get("show_thinking", True) else None,
                                json_mode=True, max_tokens=int(config.get("llm", {}).get("max_tokens", 1200)))
                for q in queue:
                    yield q
                telemetry.record_model_call("master", resp.model, resp.tokens_in, resp.tokens_out, resp.latency_ms)
                tokens = (resp.tokens_in, resp.tokens_out)
                reasoning_text = resp.reasoning
                decision = extract_json_object(resp.content)
                if decision and "action" in decision:
                    failures = 0
                    decision["_llm_written"] = True
                else:
                    failures += 1
                    decision = None
            except Exception as error:
                failures += 1
                yield {"type": "warning", "detail": f"LLM call failed ({type(error).__name__}: {str(error)[:160]}); using the heuristic planner for this step."}
            if decision is None and failures >= 2:
                use_llm_planner = False
                yield {"type": "warning", "detail": "Model output was not valid JSON twice; the heuristic planner takes over."}
            if decision is not None:
                sig = _signature(str(decision.get("action", "")), decision.get("action_input") or {})
                if sig in signatures and str(decision.get("action")) not in {"finalize", "propose_action"}:
                    repeats += 1
                    note = (f"REJECTED: '{decision.get('action')}' with identical input already ran at step {signatures[sig]}; "
                            "its result is unchanged. Choose a different tool, propose an action, or finalize.")
                    history.append({"thought": str(decision.get("thought", ""))[:200], "action": decision.get("action"),
                                    "action_input": decision.get("action_input") or {}, "observation": note})
                    yield {"type": "warning", "detail": note}
                    if repeats >= 2 or llm_calls >= max_steps + 4:
                        decision = advisor
                        repeats = 0
                    else:
                        continue
        if decision is None:
            decision = advisor
        action = str(decision.get("action", "finalize"))
        args = decision.get("action_input") or {}
        if not isinstance(args, dict):
            args = {}
        thought = str(decision.get("thought", "")).strip()
        if action == "finalize":
            draft = planner.finalize_now(state, box)["action_input"]
            for key in ("hypothesis", "probability", "recommended_action"):
                args.setdefault(key, draft[key])
            if llm_synthesis:
                try:
                    synthesized = _llm_final_synthesis(state, box, llm, telemetry, config, draft)
                except Exception as error:
                    synthesized = None
                    yield {"type": "warning", "detail": f"Final synthesis call failed ({type(error).__name__}); using the structured summary."}
                if synthesized:
                    args["answer"] = synthesized.get("answer") or args.get("answer") or draft["answer"]
                    args["uncertainty"] = synthesized.get("uncertainty") or args.get("uncertainty") or draft["uncertainty"]
                    if synthesized.get("headline") and _headline_consistent(synthesized["headline"], state):
                        args["headline"] = synthesized["headline"]
            if not args.get("answer"):
                args["answer"] = draft["answer"]
            args["answer"] = str(args["answer"]) if not isinstance(args["answer"], str) else args["answer"]
            if "ACTION —" not in args["answer"] and "Decision:" not in args["answer"]:
                tail = draft["answer"].split("ACTION —")[-1].strip() if "ACTION —" in draft["answer"] else ""
                args["answer"] = f"{args['answer'].rstrip()}\nACTION — {tail}" if tail else args["answer"]
            try:
                args["probability"] = float(args.get("probability", draft["probability"]))
            except (TypeError, ValueError):
                args["probability"] = draft["probability"]
            for key in ("hypothesis", "uncertainty", "recommended_action", "headline"):
                if key in args and not isinstance(args[key], str):
                    args[key] = str(args[key])
            args.setdefault("uncertainty", draft["uncertainty"])
        step = state.add_step(ReasoningStep(0, "master", thought, action, args))
        if reasoning_text:
            step.action_input = {**args}
            yield {"type": "reasoning", "step": step.step, "text": reasoning_text[:4000]}
        yield {"type": "thought", "step": step.step, "thought": thought, "action": action, "action_input": args}
        result: ToolResult = box.call(action, {k: v for k, v in args.items() if not k.startswith("_")})
        step.observation = result.observation
        step.latency_ms = (time.perf_counter() - t0) * 1000
        step.tokens_in, step.tokens_out = tokens
        history.append({"thought": thought, "action": action, "action_input": args, "observation": result.observation[:1200]})
        signatures.setdefault(_signature(action, args), step.step)
        steps_taken += 1
        telemetry.increment("agent_steps")
        if not result.payload.get("denied") and not result.payload.get("budget_exhausted"):
            done.add(action)
        if result.support is not None and state.hypotheses:
            before = state.hypotheses[-1].probability
            weight = {"run_vlm": 1.0, "run_critic": 1.0, "ask_user": 0.9, "run_audio_analysis": 0.6, "run_pose_analysis": 0.4}.get(action, 0.5)
            support = result.support
            strong = _perception_strength(box.primary_event())
            if action == "run_vlm":
                vlm_conf = float((result.payload.get("vlm") or {}).get("confidence", 0.6) or 0.6)
                weight *= max(0.4, min(1.0, vlm_conf))
                if support < 0 and strong >= 0.7:
                    support = max(support, -0.6)
            if action == "run_critic" and support < 0 and strong >= 0.7:
                support = max(support, -0.6)
            after = update_belief(before, support, weight)
            floor = _belief_floor(box.primary_event(), state)
            if after < floor:
                after = floor
                yield {"type": "warning", "detail": f"belief floor {floor:.2f}: the detector/second-opinion evidence for "
                                                     f"{box.primary_event().kind.replace('_', ' ')} is too strong to dismiss on one contrary opinion"}
            state.hypotheses[-1].probability = round(after, 3)
            state.uncertainty = round(1 - abs(2 * state.hypotheses[-1].probability - 1), 3)
            yield {"type": "belief", "step": step.step, "before": before, "after": state.hypotheses[-1].probability,
                   "support": support, "source": action}
        if action == "run_vlm" and result.payload.get("vlm"):
            revised = _revise_after_vlm(state, box, result.payload["vlm"])
            if revised:
                yield {"type": "hypothesis", "hypothesis": state.hypotheses[-1].to_dict(),
                       "primary_event": box.primary_event().to_dict() if box.primary_event() else None, "revised": revised}
                yield {"type": "warning", "detail": f"hypothesis revised: {revised}"}
                done.discard("run_critic")
        yield {"type": "observation", "step": step.step, "observation": result.observation, "payload": _safe(result.payload),
               "images": result.images, "cost": result.cost, "compute_spent": state.compute_spent,
               "compute_budget": state.compute_budget, "latency_ms": round(step.latency_ms, 1)}
        if action == "ask_user" and result.payload.get("question"):
            yield {"type": "question", "question": result.payload["question"], "normalized": result.payload.get("normalized")}
        if action == "propose_action" and result.payload.get("decision"):
            yield {"type": "decision", "decision": result.payload["decision"]}
        if result.stop or state.status == "done":
            break
    if state.status != "done":

        closing = planner.finalize_now(state, box)["action_input"]
        box.tool_finalize(**closing)
        state.add_step(ReasoningStep(0, "master", "Closing with the evidence gathered so far.", "finalize", closing)).observation = "Investigation finalized."
        state.status = "done"

    written = {}
    if bool(config.get("memory", {}).get("persist_episodes", True)):
        written = memory.reflect(state.run_id, perception.scene.label or "auto", [e.to_dict() for e in events],
                                 [d.to_dict() for d in state.decisions], state.final_answer, perception.location)
    telemetry.set("cache_hits", 0)
    report = _report(state, perception, graph, config, telemetry, guard, written, edge_cloud)
    _write_report(run_dir, report)
    yield {"type": "final", "report": report}

def _perception_strength(primary) -> float:
    """How many independent perception signals back the primary event (0..1)."""
    if primary is None:
        return 0.0
    a = primary.attributes or {}
    votes = 0.0
    if primary.kind == "weapon_visible":
        votes += 1.0 if float(a.get("max_conf", 0)) >= 0.6 else 0.5
        if float(a.get("persistence", 0)) >= 0.4:
            votes += 0.5
        if a.get("second_opinion") == "weapon":
            votes += 1.0
        elif a.get("second_opinion") in {"toy", "phone", "hand", "household"}:
            votes -= 1.0
        if float(a.get("gunshot_heard", 0)) >= 0.6:
            votes += 1.0
    elif primary.kind == "aggressive_interaction":
        votes += 0.8 if a.get("fast_wrist_frames") else 0.0
        votes += 1.0 if float(a.get("activity_zero_shot", 0)) >= 0.5 else (0.5 if float(a.get("activity_zero_shot", 0)) >= 0.35 else 0.0)
        votes += 0.4 if a.get("crying_heard") else 0.0
        if a.get("group_fight"):
            votes += 1.0 if int(a.get("strike_events", 0)) >= 2 else 0.5
            votes += 0.5 if int(a.get("falls_during", 0)) >= 1 else 0.0
    elif primary.kind in {"fall", "person_down"}:
        votes += 1.0 if a.get("fall_observed", True) else 0.3
        votes += 0.5 if float(a.get("tcn_fall_probability", 0)) >= 0.7 else 0.0
    elif primary.kind == "threatening_speech":
        votes += 1.0 if a.get("cues") else 0.6
    elif primary.kind == "infant_distress":
        votes += 1.0
    return max(0.0, min(1.0, votes / 2.0))

def _belief_floor(primary, state: InvestigationState) -> float:
    """Detector + zero-shot agreement on a weapon, or pose + activity agreement on a fight, keeps the belief high
    enough for a human-review notification even if one model call disagrees."""
    if primary is None:
        return 0.0
    a = primary.attributes or {}
    if primary.kind == "weapon_visible" and a.get("is_firearm"):
        if a.get("second_opinion") == "weapon" and float(a.get("gunshot_heard", 0)) >= 0.6 and float(a.get("max_conf", 0)) >= 0.6:
            return 0.75
        if a.get("second_opinion") == "weapon" and float(a.get("max_conf", 0)) >= 0.6:
            return 0.45
        if a.get("second_opinion") in (None, "uncertain") and float(a.get("max_conf", 0)) >= 0.75 and float(a.get("persistence", 0)) >= 0.5:
            return 0.3
    if primary.kind == "aggressive_interaction":
        if a.get("group_fight") and int(a.get("strike_events", 0)) >= 2 and (int(a.get("falls_during", 0)) >= 1 or float(a.get("activity_zero_shot", 0)) >= 0.35):
            return 0.5
        if a.get("group_fight"):
            return 0.4
        if float(a.get("activity_zero_shot", 0)) >= 0.5 and a.get("fast_wrist_frames"):
            return 0.35
        if float(a.get("activity_zero_shot", 0)) >= 0.6:
            return 0.3
    if primary.kind in {"fall", "person_down"} and a.get("fallen_detector_conf") and a.get("fall_observed", False):
        return 0.3
    return 0.0

def _revise_after_vlm(state: InvestigationState, box: ToolBox, vlm: dict[str, Any]) -> str | None:
    """The vision model can reveal a *different* incident than the one perception flagged (e.g. a 'person down'
    that is a crying baby, or a 'fall' that is a fight). Promote it to the working hypothesis."""
    text = " ".join(str(vlm.get(k, "")) for k in ("description", "answer", "scene")).lower()
    alts = " ".join(str(x) for x in (vlm.get("alternatives") or [])).lower()
    primary = box.primary_event()
    if primary is None:
        return None
    scene = getattr(box.perception, "scene", None)
    if scene is not None:
        scene.update_from_text(str(vlm.get("scene") or ""))
    baby_words = ("baby", "infant", "toddler", "newborn")
    distress_words = ("crying", "cries", "screaming", "distress", "wailing", "in pain")
    if primary.kind in {"person_down", "fall", "acoustic_alarm"} and any(w in text for w in baby_words) and any(w in text for w in distress_words):
        if not any(e.kind == "infant_distress" for e in state.events):
            from ..schema import Event, new_id

            ev = Event(new_id("evt"), "infant_distress", primary.start_s, primary.end_s, primary.subject, "CRYING", None,
                       primary.location, 0.7, 0.6,
                       f"The vision model reports a crying/distressed baby ({primary.start_s:.0f}-{primary.end_s:.0f}s); "
                       f"the initial '{primary.kind}' alarm was the infant's posture, not a fall. A caregiver should be told.",
                       list(primary.evidence_ids), {"age_group": "baby", "source": "vlm_revision"})
            state.events.append(ev)
            primary.severity = min(primary.severity, 0.2)
            state.hypotheses.append(Hypothesis(f"{primary.subject} (baby) is crying and needs a caregiver", "infant_distress", 0.7, list(ev.evidence_ids)))
            return "infant crying (from the vision model's description)"
    fight_words = ("fighting", "punch", "kick", "hitting", "attack", "assault", "brawl", "wrestl", "struggl")
    if primary.kind in {"fall", "person_down", "acoustic_alarm"} and any(w in text for w in fight_words) and "play" not in text:
        if not any(e.kind == "aggressive_interaction" for e in state.events):
            from ..schema import Event, new_id

            ev = Event(new_id("evt"), "aggressive_interaction", primary.start_s, primary.end_s, primary.subject, "FIGHTS_WITH", "PERSON_UNKNOWN",
                       primary.location, 0.65, 0.75, "The vision model describes people fighting/attacking in these frames.",
                       list(primary.evidence_ids), {"source": "vlm_revision"})
            state.events.append(ev)
            state.hypotheses.append(Hypothesis(f"{primary.subject} is physically aggressive toward another person", "aggressive_interaction", 0.65, list(ev.evidence_ids)))
            return "physical aggression (from the vision model's description)"
    return None

def _signature(action: str, args: dict[str, Any]) -> str:
    try:
        canon = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        canon = str(args)
    return f"{action}:{canon[:300]}"

def _llm_final_synthesis(state: InvestigationState, box: ToolBox, llm: LLMClient, telemetry: Any,
                         config: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any] | None:
    """One LLM call that turns the structured trace into the operator-facing closing report."""
    hyp = state.hypotheses[-1] if state.hypotheses else None
    scene = box.perception.scene.description() if getattr(box.perception, "scene", None) else "unknown"
    events = "\n".join(f"- {e.kind} {e.subject} {e.start_s:.1f}-{e.end_s:.1f}s conf={e.confidence:.2f} sev={e.severity:.2f}: {e.summary[:220]}"
                       for e in state.events)
    steps = "\n".join(f"{s.step}. {s.action}({json.dumps(s.action_input, default=str)[:80]}) -> {(s.observation or '')[:360]}"
                      for s in state.steps if s.action != "finalize")
    qa = ""
    if state.questions:
        q = state.questions[-1]
        qa = f"QUESTION ASKED ({q.language}): {q.text} -> ANSWER: {q.answer}"
    decisions = "; ".join(f"{d.action}: {'executed (simulated)' if d.executed else d.result}" for d in state.decisions) or "none"
    user = f"""SCENE: {scene}
OBJECTIVE: {state.objective}
EVENTS FROM PERCEPTION:
{events or '- none'}
INVESTIGATION STEPS:
{steps or '- none'}
{qa}
FINAL BELIEF in "{hyp.statement if hyp else 'n/a'}": {hyp.probability if hyp else 0:.2f}; critic: {hyp.critic_verdict if hyp else 'not run'}
DECISIONS: {decisions}
RECOMMENDED/EXECUTED ACTION (policy, do not change): {draft.get('recommended_action')}
STRUCTURED DRAFT (facts to keep): {draft.get('answer')}
Write the closing report JSON now."""
    resp = llm.chat([{"role": "system", "content": FINAL_SYSTEM}, {"role": "user", "content": user}], json_mode=True,
                    max_tokens=min(700, int(config.get("llm", {}).get("max_tokens", 1200))))
    telemetry.record_model_call("final_report", resp.model, resp.tokens_in, resp.tokens_out, resp.latency_ms)
    data = extract_json_object(resp.content)
    if not data:
        return None
    data = _english_only(data)
    if data.get("what_happened") or data.get("checks"):
        checks = data.get("checks") or []
        if isinstance(checks, str):
            checks = [checks]
        answer = compose_report(str(data.get("headline", "")), str(data.get("what_happened", "")), [str(c) for c in checks],
                                str(data.get("why", "")), str(data.get("action", "")) or draft.get("recommended_action", ""))
    else:
        answer = str(data.get("answer", "")).strip()
    if not answer.strip():
        return None
    return {"answer": answer, "uncertainty": str(data.get("uncertainty", "")).strip(), "headline": str(data.get("headline", "")).strip()[:90]}

_NON_LATIN = re.compile(r"[\u0400-\u04ff\u0590-\u06ff\u0900-\u097f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")

def _headline_consistent(headline: str, state: InvestigationState) -> bool:
    """'Visible gun detected - call police' over a run that only logged the case would mislead the operator."""
    h = headline.lower()
    executed = {d.action for d in state.decisions if d.executed}
    claims = {"call_911": ("911",), "call_police_share_location": ("police",),
              "reroute_to_hospital": ("hospital", "reroute"), "notify_parents": ("parent", "family"),
              "notify_security": ("security",), "notify_owner": ("owner", "caregiver", "hr")}
    for action, words in claims.items():
        if any(w in h for w in words) and action not in executed:
            return False
    return True

def _english_only(data: dict[str, Any]) -> dict[str, Any]:
    """Small local models sometimes drift into Chinese mid-report. Drop any field/item that is not Latin-script
    text; the structured draft fills the gap."""
    clean: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            if not _NON_LATIN.search(value):
                clean[key] = value
        elif isinstance(value, list):
            items = [v for v in value if isinstance(v, str) and not _NON_LATIN.search(v)]
            if items:
                clean[key] = items
        else:
            clean[key] = value
    return clean

def verdict_for(state: InvestigationState, primary) -> dict[str, Any]:
    """Operator-facing verdict derived from the policy state (never from free text)."""
    hyp = state.hypotheses[-1] if state.hypotheses else None
    p = hyp.probability if hyp else 0.0
    severity = primary.severity if primary else 0.0
    executed = [d.action for d in state.decisions if d.executed and d.action != "log_only"]
    kind = primary.kind if primary else "normal_activity"
    if kind == "normal_activity" or p < 0.2 or severity < 0.2:
        level, headline = "clear", "No safety incident found"
    elif any(a in HIGH_COST_ACTIONS for a in executed) or (p >= 0.7 and severity >= 0.6):
        level, headline = "alert", f"{kind.replace('_', ' ').capitalize()} — escalated"
    elif any(a in {"notify_owner", "notify_security", "continue_route", "advise_user"} for a in executed) or p >= 0.35:
        level, headline = "watch", f"Possible {kind.replace('_', ' ')} — human follow-up"
    else:
        level, headline = "clear", "No action required"
    if executed and level == "clear":
        level, headline = "watch", f"{kind.replace('_', ' ').capitalize()} — {executed[-1].replace('_', ' ')}"
    return {"level": level, "headline": headline, "belief": round(p, 3), "severity": round(severity, 3),
            "kind": kind, "executed_actions": executed}

def _build_messages(state: InvestigationState, box: ToolBox, history: list[dict[str, Any]], advisor: dict[str, Any],
                    config: dict[str, Any], steps_left: int = 12, done: set[str] | None = None) -> list[dict[str, Any]]:
    hyp = state.hypotheses[-1] if state.hypotheses else None
    events = "\n".join(f"- {e.event_id} {e.kind} {e.subject} {e.action} {e.obj or ''} @ {e.start_s:.1f}-{e.end_s:.1f}s "
                       f"conf={e.confidence:.2f} severity={e.severity:.2f}: {e.summary[:220]}" for e in state.events)
    ev = "\n".join(f"- {k} [{v.kind} @ {v.start_s:.1f}s]: {v.description[:100]}" for k, v in list(state.evidence.items())[:14])
    hist = "\n".join(f"Step {len(history) - len(history[-5:]) + i + 1}: action={h['action']} input={json.dumps(h['action_input'], default=str)[:120]}\n"
                     f"  observation={h['observation'][:520]}" for i, h in enumerate(history[-5:]))
    asked = state.questions[-1].to_dict() if state.questions else None
    decisions = [f"{d.action}: {'executed' if d.executed else d.result}" for d in state.decisions]
    scene = box.perception.scene.description() if getattr(box.perception, "scene", None) else box.perception.location
    used = sorted(done or set())
    note = f"\nOPERATOR NOTE (free text, may be wrong): {box.perception.operator_note}" if getattr(box.perception, "operator_note", None) else ""
    user = f"""OBJECTIVE: {state.objective}
MEDIA: {state.media_kind}, duration {box.perception.duration_s:.1f}s, spoken language {box.language}
SCENE (inferred from pixels): {scene}{note}
STRUCTURED EVENTS:
{events or '- none'}
EVIDENCE IDS:
{ev or '- none'}
CURRENT HYPOTHESIS: {hyp.statement if hyp else 'none'} | belief p={hyp.probability if hyp else 0:.2f} | critic={hyp.critic_verdict if hyp else 'not run'}
QUESTION ASKED: {json.dumps(asked)[:300] if asked else 'none'}
DECISIONS SO FAR: {decisions or 'none'}
BUDGET: {steps_left} step(s) left of which the last is reserved for finalize; compute spent {state.compute_spent:.0f} / {state.compute_budget:.0f}
TOOLS ALREADY USED (do not repeat with the same input): {used or 'none'}
ALLOWLISTED ACTIONS: {sorted(box.policy.allowlist)}
POLICY ADVISOR SUGGESTS NEXT: {advisor['action']} — {advisor['thought'][:300]}
PREVIOUS STEPS:
{hist or '(none yet)'}

Choose the next single tool (a different one from what was already used, unless the input is new). Respond with STRICT JSON: {{"thought": ..., "action": ..., "action_input": {{...}}}}"""

    return [{"role": "system", "content": MASTER_SYSTEM + "\nTOOLS:\n" + _tool_manual()}, {"role": "user", "content": user}]

def _safe(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        json.dumps(payload, default=str)
        return payload
    except Exception:
        return {"repr": str(payload)[:2000]}

def _report(state: InvestigationState, perception: PerceptionResult, graph: EventGraph, config: dict[str, Any],
            telemetry: Any, guard: SafetyGuard, memory_written: dict[str, int], edge_cloud) -> dict[str, Any]:
    hyp = state.hypotheses[-1] if state.hypotheses else None
    executed = [d for d in state.decisions if d.executed]
    candidates = [e for e in state.events if e.kind != "normal_activity"]
    primary = max(candidates, key=lambda e: e.severity * 0.6 + e.confidence * 0.4) if candidates else (state.events[0] if state.events else None)
    verdict = verdict_for(state, primary)
    if state.final_headline:
        verdict["headline"] = state.final_headline
    return {
        "schema_version": "2.1",
        "run_id": state.run_id, "created_at": time.time(), "profile": config.get("project", {}).get("mode"),
        "objective": state.objective, "scenario_hint": state.scenario_hint, "operator_note": perception.operator_note,
        "media": {"path": state.media_path, "kind": state.media_kind, "duration_s": perception.duration_s,
                  "location": perception.location, "backends": perception.backends},
        "scene": perception.scene.to_dict(),
        "clips": perception.clips,
        "verdict": verdict,
        "final_answer": state.final_answer,
        "final_uncertainty": state.final_uncertainty,
        "hypothesis": hyp.to_dict() if hyp else None,
        "belief": hyp.probability if hyp else None,
        "uncertainty": state.uncertainty,
        "events": [e.to_dict() for e in state.events],
        "evidence": {k: v.to_dict() for k, v in state.evidence.items()},
        "windows": [w.__dict__ for w in perception.windows],
        "audio": {"segments": [s.to_dict() for s in perception.audio_segments],
                  "acoustic_events": [a.to_dict() for a in perception.acoustic_events],
                  "text_risk": perception.text_risk.to_dict()},
        "steps": [s.to_dict() for s in state.steps],
        "questions": [q.to_dict() for q in state.questions],
        "decisions": [d.to_dict() for d in state.decisions],
        "executed_actions": [d.action for d in executed],
        "retrieved": state.retrieved[:12],
        "graph": graph.export(),
        "important_entities": graph.important_entities(),
        "compute": {"spent": state.compute_spent, "budget": state.compute_budget},
        "telemetry": telemetry.to_dict(config.get("telemetry", {}), perception.duration_s),
        "security": guard.export(),
        "memory_written": memory_written,
        "edge_cloud": edge_cloud.export() if edge_cloud is not None else None,
        "disclosure": {
            "external_actions_executed_for_real": False,
            "all_consequential_actions_simulated": True,
            "semantic_model": config.get("llm", {}).get("model") if getattr(state, "media_kind", "") else None,
            "visible_reasoning": "Thoughts shown are the agent's own step summaries (ReAct), plus the model's exposed reasoning stream when the server provides it.",
        },
    }

def _write_report(run_dir: Path, report: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    with open(run_dir / "events.jsonl", "w", encoding="utf-8") as handle:
        for e in report["events"]:
            handle.write(json.dumps(e, default=str) + "\n")
    with open(run_dir / "reasoning_trace.jsonl", "w", encoding="utf-8") as handle:
        for s in report["steps"]:
            handle.write(json.dumps(s, default=str) + "\n")
    with open(run_dir / "telemetry.json", "w", encoding="utf-8") as handle:
        json.dump(report["telemetry"], handle, indent=2, default=str)
