"""Adversarial critic: tries to disprove the master's hypothesis before any
consequential action. Same local model, different role prompt, evidence-only
context (no access to the master's reasoning) so it cannot simply agree.
"""
from __future__ import annotations

from typing import Any

from ..schema import Event, Hypothesis, InvestigationState
from .llm import LLMClient, extract_json_object

CRITIC_SYSTEM = (
    "You are the CRITIC agent of a safety-investigation system. Your job is to attack the hypothesis: "
    "find the strongest benign explanation, point out evidence that is weak (single frame, low confidence, "
    "ambiguous posture, sarcasm/quotation in speech, toy vs. real weapon, phone vs. gun, intentional lying "
    "down vs. fall, play vs. aggression). Be concrete and cite evidence ids. Never invent facts. "
    "Verdict rules: 'refuted' ONLY when a cited observation directly contradicts the hypothesis (e.g. the vision "
    "model explicitly identified the object as a toy/phone, the person stood up and walked normally, the speech is "
    "clearly a quotation or a joke). Low confidence, few frames, or a merely possible benign story is 'weakened', "
    "not 'refuted'. Independent signals that agree (detector + zero-shot second opinion + vision model, or pose + "
    "activity classifier) make the hypothesis 'supported' even if one of them is weak. "
    "The hypothesis is about whether a SAFETY INCIDENT of that kind is happening, not about who is to blame: an "
    "alternative that only changes the roles (the named person is the victim, several people attack each other, a "
    "crowd brawl, self-defence) is NOT a benign alternative — it confirms the incident and the verdict is 'supported'. "
    "Only play-fighting, dancing, sport, celebration, greeting/hugging or clearly staged acting can refute a fight. "
    "The age-group label (child/adult) is a weak zero-shot guess that is often wrong for helmets, hats and small figures: "
    "never build a 'toy' alternative on it alone. If a gunshot was heard and the zero-shot second opinion says 'weapon', a "
    "toy is not a credible alternative — a toy does not fire. "
    "Reply with strict JSON: {\"verdict\": \"supported\"|\"weakened\"|\"refuted\", \"strongest_alternative\": str, "
    "\"contradicting_evidence\": str, \"missing_evidence\": [str], \"confidence\": number in [0,1], \"notes\": str}"
)

def run_critic(llm: LLMClient, state: InvestigationState, hypothesis: Hypothesis, primary: Event | None,
               telemetry: Any, fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    fixture = fixture or {}
    if not llm.is_real:
        spec = fixture.get("critic", {})
        kind = primary.kind if primary else "normal_activity"
        verdict = spec.get("verdict", {}).get(kind)
        if verdict is None:
            verdict = "supported" if hypothesis.probability >= 0.6 else "weakened"
        return {"verdict": verdict,
                "strongest_alternative": spec.get("alternative", {}).get(kind, "A benign explanation remains possible (fixture critic)."),
                "missing_evidence": spec.get("missing", {}).get(kind, ["independent second sensor", "longer temporal context"]),
                "confidence": 0.65, "notes": "fixture critic"}
    evidence_lines = []
    for ev_id in hypothesis.evidence_ids[:12]:
        ev = state.evidence.get(ev_id)
        if ev:
            evidence_lines.append(f"{ev_id} [{ev.kind} @ {ev.start_s:.1f}s score {ev.score:.2f}]: {ev.description[:220]}")
    other_events = [f"{e.kind} {e.start_s:.1f}-{e.end_s:.1f}s conf {e.confidence:.2f}: {e.summary[:160]}" for e in state.events]
    prompt = (
        f"Hypothesis under test: {hypothesis.statement}\nCurrent belief: {hypothesis.probability:.2f}\n"
        f"Location: {primary.location if primary else 'unknown'}\n"
        f"Structured events:\n- " + "\n- ".join(other_events) + "\n"
        f"Cited evidence:\n- " + "\n- ".join(evidence_lines or ["(none)"]) + "\n"
        "Attack the hypothesis. Output JSON only."
    )
    images = [state.evidence[i].path for i in hypothesis.evidence_ids if i in state.evidence and state.evidence[i].kind == "frame" and state.evidence[i].path][:4]
    resp = llm.chat([{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": prompt}],
                    json_mode=True, images=images or None, max_tokens=600)
    telemetry.record_model_call("critic", resp.model, resp.tokens_in, resp.tokens_out, resp.latency_ms, len(images))
    data = extract_json_object(resp.content) or {}
    verdict = str(data.get("verdict", "weakened")).lower()
    if verdict not in {"supported", "weakened", "refuted"}:
        verdict = "weakened"
    contradiction = str(data.get("contradicting_evidence", "") or "").strip()
    if verdict == "refuted" and (len(contradiction) < 12 or contradiction.lower() in {"none", "n/a", "null"}):
        verdict = "weakened"
    notes = str(data.get("notes", "") or "")
    alternative = str(data.get("strongest_alternative", resp.content[:200]) or "")
    verdict, reconciled = reconcile_verdict(primary.kind if primary else "", verdict, alternative, contradiction, notes)
    if (primary is not None and primary.kind == "weapon_visible" and verdict != "supported"
            and float(primary.attributes.get("gunshot_heard", 0)) >= 0.6 and primary.attributes.get("second_opinion") == "weapon"
            and any(w in f"{alternative} {notes}".lower() for w in ("toy", "prop", "replica", "airsoft", "fake"))):
        verdict, reconciled = "supported", ("[reconciled: the 'toy' alternative is contradicted by the gunshot heard "
                                            f"({primary.attributes.get('gunshot_heard'):.2f}) and the zero-shot second opinion 'weapon' — toys do not fire]")
    if reconciled:
        notes = (notes + " " if notes else "") + reconciled
    return {"verdict": verdict, "strongest_alternative": alternative,
            "contradicting_evidence": contradiction, "reconciled": reconciled or None,
            "missing_evidence": data.get("missing_evidence", []), "confidence": float(data.get("confidence", 0.5) or 0.5),
            "notes": notes, "reasoning": resp.reasoning[:1200]}

SAME_INCIDENT_WORDS = {
    "aggressive_interaction": ("brawl", "fight", "fighting", "attack", "assault", "struck", "strike", "striking", "punch",
                               "kick", "shov", "grappl", "wrestl", "victim", "aggressor", "self-defen", "self defen",
                               "defending", "mutual", "group", "several people", "multiple people", "melee", "scuffle"),
    "weapon_visible": ("real gun", "real firearm", "actual gun", "loaded", "armed", "pointing the gun", "brandish"),
    "fall": ("collapsed", "knocked down", "pushed", "tripped", "fell", "unconscious", "injur"),
    "person_down": ("collapsed", "unconscious", "injur", "not moving", "motionless"),
    "threatening_speech": ("threat", "intimidat", "extort", "warning of violence"),
}
BENIGN_PATTERNS = (r"\bplay(ing|ful|fight|-fight|ed)?\b", r"\bdanc", r"\bsport", r"\bsparring\b", r"\bcelebrat", r"\bhug(s|ging|ged)?\b",
                   r"\bgreet", r"\bjok(e|ing)", r"\blaugh", r"\brehears", r"\bstaged\b", r"\bact(ing|ors?)\b", r"\btoy\b",
                   r"\bphone\b", r"\bremote control\b", r"\bprop\b", r"\bsleep", r"\bresting\b", r"\bnap(ping)?\b",
                   r"\byoga\b", r"\bexercis", r"\bstretch", r"\bsarcas", r"\bquot(e|ation|ing)", r"\blyric", r"\bsong\b", r"\bsing(ing)?\b",
                   r"\bhorseplay\b", r"\broughhous", r"\bfriendly\b", r"\bmock\b")

def reconcile_verdict(kind: str, verdict: str, alternative: str, contradiction: str, notes: str) -> tuple[str, str]:
    """A critic that 'refutes' a fight by saying the named person is the victim of a brawl has confirmed the fight.
    The belief tracks the incident, not the blame; only a benign alternative may lower it."""
    if verdict == "supported" or not kind:
        return verdict, ""
    import re

    text = f"{alternative} {contradiction} {notes}".lower()
    same = [w for w in SAME_INCIDENT_WORDS.get(kind, ()) if w in text]
    benign = [pat for pat in BENIGN_PATTERNS if re.search(pat, text)]
    if same and not benign:
        return "supported", (f"[reconciled: the critic's alternative ('{alternative[:90]}') is itself a {kind.replace('_', ' ')} "
                             f"— only the roles differ, so the incident stands and the verdict counts as SUPPORTED]")
    return verdict, ""
