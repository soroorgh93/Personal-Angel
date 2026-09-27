"""Turn raw perception into structured Events + Evidence.

An Event is the contract between computer vision and the agent:
  subject / action / object / location / time / confidence / severity / evidence
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..perception.pipeline import PerceptionResult
from ..schema import Event, Evidence, new_id

SEVERITY_PRIOR = {
    "fall": 0.70, "person_down": 0.55, "slump_unresponsive": 0.65, "weapon_visible": 0.80,
    "aggressive_interaction": 0.75, "threatening_speech": 0.70, "hateful_speech": 0.50,
    "distress_speech": 0.60, "acoustic_alarm": 0.50, "infant_distress": 0.60, "abusive_speech": 0.45, "object_removed": 0.40,
    "scene_change": 0.15, "normal_activity": 0.0,
}
ACOUSTIC_SEVERITY = {"gunshot": 0.9, "scream": 0.65, "glass_breaking": 0.5, "baby_crying": 0.35,
                     "crying": 0.35, "moaning_in_pain": 0.55, "shouting": 0.4, "angry_argument": 0.35,
                     "music": 0.0, "singing": 0.0}

def people_by_motion(result: PerceptionResult, start_s: float, end_s: float) -> list[int]:
    """Person tracks ordered by how much they moved inside a window (box-centre displacement per second,
    normalised by box height). Fighters move; bystanders watching do not."""
    pool = [o for o in (result.dense or result.scan) if start_s - 0.5 <= o.timestamp_s <= end_s + 0.5]
    per: dict[int, list[tuple[float, float, float, float]]] = {}
    for o in pool:
        for d in o.detections:
            if d.label == "person" and d.track_id is not None:
                x1, y1, x2, y2 = d.box_xyxy
                per.setdefault(d.track_id, []).append((o.timestamp_s, (x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1.0)))
    motion: dict[int, float] = {}
    for tid, rows in per.items():
        rows.sort()
        if len(rows) < 2:
            continue
        total = 0.0
        for (t0, x0, y0, h0), (t1, x1, y1, h1) in zip(rows, rows[1:]):
            dt = max(t1 - t0, 1e-3)
            total += (((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / ((h0 + h1) / 2)) / dt
        motion[tid] = total / (len(rows) - 1)
    return [t for t, _ in sorted(motion.items(), key=lambda kv: -kv[1])]

def nearest_person_distance(result: PerceptionResult, track_id: int, t: float) -> tuple[int | None, float]:
    """(other track id, distance in box-widths) of the closest other person around time t."""
    obs = result.observation_at(t)
    if obs is None:
        return None, 9.9
    me = next((d for d in obs.detections if d.label == "person" and d.track_id == track_id), None)
    if me is None:
        return None, 9.9
    mx, my = (me.box_xyxy[0] + me.box_xyxy[2]) / 2, (me.box_xyxy[1] + me.box_xyxy[3]) / 2
    mw = max(me.box_xyxy[2] - me.box_xyxy[0], 1.0)
    best: tuple[int | None, float] = (None, 9.9)
    for d in obs.detections:
        if d.label != "person" or d.track_id in (None, track_id):
            continue
        ox, oy = (d.box_xyxy[0] + d.box_xyxy[2]) / 2, (d.box_xyxy[1] + d.box_xyxy[3]) / 2
        dist = ((ox - mx) ** 2 + (oy - my) ** 2) ** 0.5 / mw
        if dist < best[1]:
            best = (d.track_id, dist)
    return best

def people_by_size_first(result: PerceptionResult) -> int | None:
    sizes: dict[int, float] = {}
    for o in (result.dense or result.scan):
        for d in o.detections:
            if d.label == "person" and d.track_id is not None:
                a = (d.box_xyxy[2] - d.box_xyxy[0]) * (d.box_xyxy[3] - d.box_xyxy[1])
                sizes[d.track_id] = max(sizes.get(d.track_id, 0.0), a)
    return max(sizes, key=sizes.get) if sizes else None

def _person(track_id: int | None) -> str:
    return f"PERSON_{track_id:02d}" if track_id is not None and track_id >= 0 else "PERSON_UNKNOWN"

def _consolidate_group_fight(events: list[Event], result: PerceptionResult) -> Event | None:
    """Crowd fights fragment into many small events (one 'striking' event per person, one 'fall' per person
    knocked down, tracker ids that change every few frames). Individually each looks weak — 'nobody within reach',
    'fall outcome unknown' — and a critic can wave each one away. Together they are one incident: a brawl.
    Merge them into a single high-confidence aggressive_interaction whose roles are explicitly unknown."""
    strikes = [e for e in events if e.kind == "aggressive_interaction"]
    downs = [e for e in events if e.kind in {"fall", "person_down"}]
    if not strikes:
        return None
    start = min(e.start_s for e in strikes)
    end = max(e.end_s for e in strikes)
    falls_during = [e for e in downs if e.end_s >= start - 2.0 and e.start_s <= end + 2.0]
    activity = float((result.scene.activity or {}).get("fight", 0.0)) if result.scene else 0.0
    activity = max(activity, max((float(e.attributes.get("activity_zero_shot", 0.0)) for e in strikes), default=0.0))
    strikers = list(dict.fromkeys(e.subject for e in strikes if e.subject and e.subject != "PERSON_UNKNOWN"))
    for e in strikes:
        for p in e.attributes.get("people", []) or []:
            if p not in strikers and p != "PERSON_UNKNOWN":
                strikers.append(p)
    group = (len(strikes) >= 2) or (len(strikes) >= 1 and len(falls_during) >= 2) or (len(strikes) >= 1 and falls_during and activity >= 0.3)
    if not group:
        return None
    conf = 1.0
    for e in strikes:
        conf *= 1.0 - min(0.9, e.confidence)
    conf = 1.0 - conf
    conf = min(0.95, conf + 0.05 * len(falls_during) + (0.1 if activity >= 0.4 else 0.0))
    within_reach = sum(1 for e in strikes if e.attributes.get("within_reach"))
    wrist_frames = sum(int(e.attributes.get("fast_wrist_frames", 0) or 0) for e in strikes)
    ids: list[str] = []
    for e in strikes + falls_during:
        for i in e.evidence_ids:
            if i not in ids:
                ids.append(i)
    lead = max(strikes, key=lambda e: e.confidence)
    people_txt = ", ".join(strikers[:5]) + (" and others" if len(strikers) > 5 else "")
    summary = (f"A multi-person fight (brawl) is likely between {start:.1f}s and {end:.1f}s: {len(strikes)} rapid striking motion(s) by "
               f"{people_txt or 'several people'}" + (f" ({within_reach} within reach of another person)" if within_reach else "")
               + (f", and {len(falls_during)} person(s) went to the ground during it" if falls_during else "")
               + (f"; the zero-shot activity classifier scores 'fight' at {activity:.2f}" if activity >= 0.3 else "")
               + ". Who started it cannot be determined from this camera — the safety-relevant fact is that several people are "
                 "physically fighting, and each fall is a possible injury.")
    merged = Event(new_id("evt"), "aggressive_interaction", start, end, lead.subject, "GROUP_FIGHT", "SEVERAL_PEOPLE",
                   result.location, round(conf, 3), max(SEVERITY_PRIOR["aggressive_interaction"], 0.8), summary, ids,
                   {"group_fight": True, "strikers": strikers, "strike_events": len(strikes), "falls_during": len(falls_during),
                    "within_reach_events": within_reach, "fast_wrist_frames": wrist_frames or None,
                    "activity_zero_shot": round(activity, 3), "roles_unknown": True,
                    "member_summaries": [e.summary[:160] for e in strikes[:6]]})
    for e in strikes:
        events.remove(e)
    for e in falls_during:
        e.attributes["during_fight"] = True
        if "during the fight" not in e.summary:
            e.summary += " This happened during the fight — a possible injury rather than a medical collapse."
    events.append(merged)
    return merged

def _fold_gunshot_into_weapon(events: list[Event]) -> None:
    """A gunshot heard while a firearm is in view is corroboration of the weapon, not a separate 'alarming sound'
    incident that would invite an 'are you okay?' question. Keep the sound as evidence, make the weapon primary."""
    weapons = [e for e in events if e.kind == "weapon_visible" and e.attributes.get("is_firearm")]
    shots = [e for e in events if e.kind == "acoustic_alarm" and str(e.obj or "").upper() in {"GUNSHOT", "GUNSHOT_OR_GUNFIRE", "EXPLOSION"}]
    if not weapons or not shots:
        return
    w = max(weapons, key=lambda e: e.confidence)
    for shot in shots:
        w.confidence = round(min(0.98, w.confidence + 0.1), 3)
        w.severity = max(w.severity, 0.92)
        w.attributes["gunshot_heard"] = max(float(w.attributes.get("gunshot_heard", 0.0)), float(shot.confidence))
        w.evidence_ids.extend(i for i in shot.evidence_ids if i not in w.evidence_ids)
        w.summary += f" A gunshot was heard at {shot.start_s:.1f}s (confidence {shot.confidence:.2f}) — a toy does not fire."
        shot.severity = 0.3
        shot.attributes["folded_into"] = w.event_id
        shot.summary += " (corroborates the visible firearm; handled under the weapon incident)"

def build_events(result: PerceptionResult, run_dir: Path) -> tuple[list[Event], dict[str, Evidence]]:
    events: list[Event] = []
    evidence: dict[str, Evidence] = {}

    def frame_evidence(ts: float, description: str, score: float, payload: dict[str, Any] | None = None) -> str | None:
        obs = result.observation_at(ts)
        if obs is None:
            return None
        path = obs.image_path or result.evidence_frames.get(obs.frame_index)
        if path is None:

            best = None
            for fi, p in result.evidence_frames.items():
                o = next((x for x in (result.dense or result.scan) if x.frame_index == fi), None)
                if o and abs(o.timestamp_s - ts) < 1.5 and (best is None or abs(o.timestamp_s - ts) < best[0]):
                    best = (abs(o.timestamp_s - ts), p, o)
            if best is None:
                return None
            path, obs = best[1], best[2]
        ev_id = new_id("ev")
        evidence[ev_id] = Evidence(ev_id, "frame", obs.timestamp_s, obs.timestamp_s, description, path, score,
                                   {"frame_index": obs.frame_index, "detections": [d.to_dict() for d in obs.detections],
                                    **(payload or {})})
        return ev_id

    people_tracks = {p.track_id for o in (result.dense or result.scan) for p in o.poses}
    for hev in result.human_events:
        if hev.kind == "recovered":
            continue
        ids = [i for i in (
            frame_evidence(hev.start_s, f"Onset: {hev.explanation}", hev.confidence, hev.features),
            frame_evidence((hev.start_s + hev.end_s) / 2, f"Mid: {hev.kind} in progress", hev.confidence),
            frame_evidence(hev.end_s, f"End: state at {hev.end_s:.1f}s", hev.confidence),
        ) if i]
        pose_id = new_id("ev")
        evidence[pose_id] = Evidence(pose_id, "pose", hev.start_s, hev.end_s, hev.explanation, None,
                                     hev.confidence, dict(hev.features))
        ids.append(pose_id)
        if hev.kind == "striking_motion":
            others = [t for t in people_tracks if t != hev.track_id]
            crying = any(a.label in {"baby_crying", "crying", "scream"} for a in result.acoustic_events)
            if others or crying:
                near_id, dist = nearest_person_distance(result, hev.track_id, (hev.start_s + hev.end_s) / 2)
                target = _person(near_id) if near_id is not None else (_person(others[0]) if others else "PERSON_UNKNOWN")
                within_reach = dist <= 1.6
                conf = min(0.9, hev.confidence + (0.15 if crying else 0.0) + (0.1 if within_reach else -0.25))
                note = (f" {target} is within reach ({dist:.1f} body-widths)." if within_reach and near_id is not None
                        else f" Nobody is within reach (nearest person {dist:.1f} body-widths away) — this may be gesturing, not striking.")
                events.append(Event(new_id("evt"), "aggressive_interaction", hev.start_s, hev.end_s,
                                    _person(hev.track_id), "STRUCK_TOWARD", target, result.location, round(max(conf, 0.2), 3),
                                    SEVERITY_PRIOR["aggressive_interaction"],
                                    f"{_person(hev.track_id)} made {hev.features.get('fast_wrist_frames')} rapid arm "
                                    f"movements toward {target}" + (" while crying was heard" if crying else "") + "." + note,
                                    ids, {"other_people": [_person(t) for t in others], "crying_heard": crying,
                                          "target_distance_body_widths": round(dist, 2), "within_reach": within_reach, **hev.features}))
            continue
        kind = hev.kind
        action = {"fall": "FELL", "person_down": "LYING_ON_GROUND", "slump_unresponsive": "SLUMPED_UNRESPONSIVE"}[kind]
        age = result.scene.age_of(hev.track_id) if result.scene else None
        severity, confidence, explanation = SEVERITY_PRIOR[kind], hev.confidence, hev.explanation
        attrs = dict(hev.features)
        if age:
            attrs["age_group"] = age
        if age == "baby" and kind in {"fall", "person_down"}:

            kind, action, severity = "person_down", "ON_THE_FLOOR", 0.2
            confidence = min(confidence, 0.5)
            explanation += " Scene understanding: this person looks like a baby/toddler — being low or horizontal is normal; verify with the VLM rather than escalating."
        elif age == "elderly" and kind == "fall":
            severity = min(0.95, severity + 0.1)
            explanation += " Scene understanding: the person looks elderly — falls carry a higher injury risk."
        events.append(Event(new_id("evt"), kind, hev.start_s, hev.end_s, _person(hev.track_id), action, None,
                            result.location, confidence, severity, explanation, ids, attrs))

    for w in result.weapons:
        ids = [i for i in (
            frame_evidence(w.first_s, f"First sighting of {w.label} (conf {w.max_conf:.2f})", w.max_conf),
            frame_evidence((w.first_s + w.last_s) / 2, f"{w.label} persisted {w.frames_seen}/{w.frames_possible} frames", w.max_conf),
        ) if i]
        track_id = new_id("ev")
        evidence[track_id] = Evidence(track_id, "track", w.first_s, w.last_s,
                                      f"{w.label} seen in {w.frames_seen} of {w.frames_possible} inspected frames "
                                      f"({w.persistence:.0%} persistence), max confidence {w.max_conf:.2f}",
                                      None, w.persistence, {"label": w.label, "holder": w.holder_track,
                                                           "frames": w.evidence_frames})
        ids.append(track_id)
        is_gun = any(k in w.label for k in ("gun", "pistol", "rifle", "firearm"))
        conf = min(0.95, 0.4 * w.max_conf + 0.4 * w.persistence + 0.2)
        severity = 0.9 if is_gun else 0.7
        summary = (f"A visible {w.label} was detected {w.frames_seen} times between {w.first_s:.1f}s and "
                   f"{w.last_s:.1f}s, associated with {_person(w.holder_track)}.")
        attrs: dict[str, Any] = {"label": w.label, "max_conf": w.max_conf, "persistence": w.persistence,
                                 "is_firearm": is_gun, "visible_part_only": True}
        holder_age = result.scene.age_of(w.holder_track) if result.scene else None
        if holder_age:
            attrs["holder_age_group"] = holder_age
        if w.second_opinion is not None:
            attrs["second_opinion"] = w.second_opinion
            attrs["second_opinion_weapon_probability"] = w.weapon_probability
            if w.second_opinion == "weapon":
                conf = min(0.97, conf + 0.15)
                summary += f" Zero-shot second opinion agrees it is a real weapon (p={w.weapon_probability:.2f})."
            elif w.second_opinion == "uncertain":
                summary += f" Zero-shot second opinion is undecided (weapon p={w.weapon_probability:.2f}); the VLM must verify."
            else:
                conf = max(0.15, conf * 0.5)
                severity = 0.45
                summary += (f" Zero-shot second opinion says the object looks like a {w.second_opinion} "
                            f"(weapon p={w.weapon_probability:.2f}) — likely a false positive; verify before acting.")
        if holder_age in {"baby", "child"} and w.second_opinion != "weapon":
            severity = min(severity, 0.4)
            summary += f" The holder looks like a {holder_age}; a toy is the more likely explanation."
        events.append(Event(new_id("evt"), "weapon_visible", w.first_s, w.last_s, _person(w.holder_track),
                            "HOLDS_OR_CARRIES", f"{w.label.upper()}_{(w.holder_track or 0):02d}", result.location, conf,
                            severity, summary, ids, attrs))

    def covered(start: float, end: float, kinds: set[str]) -> Event | None:
        return next((e for e in events if e.kind in kinds and e.end_s >= start - 1.5 and e.start_s <= end + 1.5), None)

    for fs in getattr(result, "fallen_sightings", []) or []:
        hit = covered(fs["first_s"], fs["last_s"], {"fall", "person_down"})
        ids = [i for i in (frame_evidence(fs["first_s"], f"Fallen-person detector: first sighting (conf {fs['max_conf']:.2f})", fs["max_conf"]),
                           frame_evidence(fs["last_s"], f"Fallen-person detector: {fs['frames_seen']}/{fs['frames_possible']} frames", fs["max_conf"])) if i]
        did = new_id("ev")
        evidence[did] = Evidence(did, "track", fs["first_s"], fs["last_s"],
                                 f"Fallen-person detector saw {_person(fs['track_id'])} on the ground in {fs['frames_seen']} of {fs['frames_possible']} "
                                 f"frames ({fs['persistence']:.0%}), max confidence {fs['max_conf']:.2f}" + ("; the same person was upright earlier" if fs.get("upright_before") else ""),
                                 None, fs["max_conf"], dict(fs))
        ids.append(did)
        if hit is not None:
            hit.confidence = round(min(0.97, hit.confidence + 0.15), 3)
            hit.evidence_ids.extend(ids)
            hit.attributes["fallen_detector_conf"] = fs["max_conf"]
            hit.summary += f" An independent fallen-person detector agrees ({fs['frames_seen']} frames, conf {fs['max_conf']:.2f})."
            continue
        kind = "fall" if fs.get("upright_before") else "person_down"
        conf = min(0.85, 0.35 + 0.35 * fs["max_conf"] + 0.2 * fs["persistence"])
        events.append(Event(new_id("evt"), kind, fs["first_s"], fs["last_s"], _person(fs["track_id"]),
                            "FELL" if kind == "fall" else "LYING_ON_GROUND", None, result.location, round(conf, 3), SEVERITY_PRIOR[kind],
                            f"The fallen-person detector saw {_person(fs['track_id'])} on the ground in {fs['frames_seen']} of {fs['frames_possible']} frames "
                            f"({fs['first_s']:.1f}-{fs['last_s']:.1f}s, conf {fs['max_conf']:.2f})" + (", after being upright earlier" if fs.get("upright_before") else "")
                            + ". Pose geometry alone did not confirm a fall (short clip or partial keypoints) — verify with the vision model.",
                            ids, {"fallen_detector_conf": fs["max_conf"], "persistence": fs["persistence"], "source": "fallen_detector",
                                  "fall_observed": bool(fs.get("upright_before"))}))
    for ae in [a for a in (getattr(result, "activity_events", []) or []) if a.get("kind") == "fallen_activity"]:
        hit = covered(ae["start_s"], ae["end_s"], {"fall", "person_down"})
        aid = new_id("ev")
        evidence[aid] = Evidence(aid, "activity", ae["start_s"], ae["end_s"],
                                 f"Zero-shot activity classifier: person collapsed/lying in {ae['frames_hit']}/{ae['frames_checked']} sampled frames (peak {ae['peak_score']:.2f})",
                                 None, ae["confidence"], dict(ae))
        if hit is not None:
            hit.confidence = round(min(0.97, hit.confidence + 0.1), 3)
            hit.evidence_ids.append(aid)
            hit.attributes["fallen_activity"] = ae["peak_score"]
            continue
        subj = _person(people_by_size_first(result))
        events.append(Event(new_id("evt"), "person_down", ae["start_s"], ae["end_s"], subj, "LYING_ON_GROUND", None, result.location,
                            ae["confidence"], SEVERITY_PRIOR["person_down"],
                            f"A zero-shot activity classifier sees a person collapsed or lying on the floor in {ae['frames_hit']} of {ae['frames_checked']} "
                            f"sampled frames ({ae['start_s']:.1f}-{ae['end_s']:.1f}s). Not confirmed by pose or the fallen detector — verify with the vision model.",
                            [aid], {"fallen_activity": ae["peak_score"], "source": "activity_zero_shot"}))

    people_by_size: list[int] = []
    sizes: dict[int, float] = {}
    for o in (result.dense or result.scan):
        for d in o.detections:
            if d.label == "person" and d.track_id is not None:
                a = (d.box_xyxy[2] - d.box_xyxy[0]) * (d.box_xyxy[3] - d.box_xyxy[1])
                sizes[d.track_id] = max(sizes.get(d.track_id, 0.0), a)
    people_by_size = [t for t, _ in sorted(sizes.items(), key=lambda kv: -kv[1])]
    for ae in [a for a in (getattr(result, "activity_events", []) or []) if a.get("kind") != "fallen_activity"]:
        existing = next((e for e in events if e.kind == "aggressive_interaction"
                         and e.end_s >= ae["start_s"] - 1 and e.start_s <= ae["end_s"] + 1), None)
        ids = [i for i in (
            frame_evidence(ae["start_s"], f"Activity zero-shot: violent activity onset (score {ae['peak_score']:.2f})", ae["confidence"], {"activity": ae["scores"]}),
            frame_evidence((ae["start_s"] + ae["end_s"]) / 2, "Activity zero-shot: fight/shove in progress", ae["confidence"]),
        ) if i]
        act_id = new_id("ev")
        evidence[act_id] = Evidence(act_id, "activity", ae["start_s"], ae["end_s"],
                                    f"Zero-shot activity classifier: violent activity in {ae['frames_hit']}/{ae['frames_checked']} sampled frames "
                                    f"(peak {ae['peak_score']:.2f}, mean {ae['mean_score']:.2f}); scores {ae['scores']}",
                                    None, ae["confidence"], dict(ae))
        ids.append(act_id)
        if existing is not None:
            existing.confidence = round(min(0.95, max(existing.confidence, 0.5 * existing.confidence + 0.5 * ae["confidence"]) + 0.1), 3)
            existing.evidence_ids.extend(ids)
            existing.attributes["activity_zero_shot"] = ae["peak_score"]
            existing.summary += f" Independent zero-shot activity classifier also sees fighting (peak {ae['peak_score']:.2f})."
            continue
        movers = people_by_motion(result, ae["start_s"], ae["end_s"])
        subj = _person(movers[0]) if movers else "PERSON_UNKNOWN"
        obj = _person(movers[1]) if len(movers) > 1 else "PERSON_UNKNOWN"
        events.append(Event(new_id("evt"), "aggressive_interaction", ae["start_s"], ae["end_s"], subj, "FIGHTS_WITH", obj,
                            result.location, ae["confidence"], SEVERITY_PRIOR["aggressive_interaction"],
                            f"A zero-shot activity classifier flagged fighting/shoving between people in {ae['frames_hit']} of "
                            f"{ae['frames_checked']} sampled frames ({ae['start_s']:.1f}-{ae['end_s']:.1f}s, peak score {ae['peak_score']:.2f}). "
                            f"Pose keypoints did not resolve a striking motion (small or occluded figures); the people moving most in the window are "
                            f"{subj} and {obj}.", ids,
                            {"activity_zero_shot": ae["peak_score"], "frames_hit": ae["frames_hit"], "people": [_person(t) for t in movers[:4]],
                             "source": "activity_zero_shot"}))

    babies = [p.track_id for p in (result.scene.people if result.scene else []) if p.age_group == "baby"]
    adults = [p.track_id for p in (result.scene.people if result.scene else []) if p.age_group in {"adult", "elderly"}]
    crying = [a for a in result.acoustic_events if a.label in {"baby_crying", "crying"}]
    if babies and crying:
        start, end = min(a.start_s for a in crying), max(a.end_s for a in crying)
        ids = [i for i in (frame_evidence(start, "Baby in view while crying is heard", 0.6),
                           frame_evidence((start + end) / 2, "Crying continues", 0.6)) if i]
        for a in crying:
            aid = new_id("ev")
            evidence[aid] = Evidence(aid, "acoustic", a.start_s, a.end_s, f"Acoustic event: {a.label} ({a.confidence:.2f})", None, a.confidence, a.to_dict())
            ids.append(aid)
        conf = min(0.9, 0.45 + 0.4 * max(a.confidence for a in crying) + (0.1 if not adults else 0.0))
        events.append(Event(new_id("evt"), "infant_distress", start, end, _person(babies[0]), "CRYING", None, result.location, conf,
                            SEVERITY_PRIOR["infant_distress"],
                            f"{_person(babies[0])} (baby) is crying for {end - start:.0f}s" + (" with no adult in view" if not adults else " (an adult is in view")
                            + ". A caregiver should be told; not an emergency unless injury or choking is seen.", ids,
                            {"age_group": "baby", "adult_present": bool(adults), "crying_s": round(end - start, 1)}))

    if result.audio_segments:
        risk = result.text_risk
        original = " ".join(s.text for s in result.audio_segments)
        english = " ".join(s.translation_en or s.text for s in result.audio_segments)
        lang = result.audio_segments[0].language
        ids = []
        for seg in result.audio_segments:
            sid = new_id("ev")
            evidence[sid] = Evidence(sid, "audio", seg.start_s, seg.end_s,
                                     f"[{seg.language}] \"{seg.text}\"" + (f" → EN: \"{seg.translation_en}\"" if seg.translation_en and seg.language != "en" else ""),
                                     None, 0.8, seg.to_dict())
            ids.append(sid)
        text_id = new_id("ev")
        evidence[text_id] = Evidence(text_id, "text", result.audio_segments[0].start_s, result.audio_segments[-1].end_s,
                                     f"Text-risk model {risk.model}: threat={risk.threat:.2f} hate={risk.hate:.2f} "
                                     f"toxicity={risk.toxicity:.2f}; cues={risk.matched_cues}", None,
                                     max(risk.threat, risk.hate), risk.to_dict())
        ids.append(text_id)
        speaker = "SPEAKER_01"
        start, end = result.audio_segments[0].start_s, result.audio_segments[-1].end_s
        from ..perception.audio import DISTRESS_CUES, match_cues

        distress = match_cues(original, DISTRESS_CUES) + match_cues(english, DISTRESS_CUES)
        music = getattr(result, "audio_kind", "speech") in {"music", "mixed"}
        music_note = (" The audio is classified as MUSIC/SINGING: these are sung lyrics, not speech directed at a person, "
                      "so they are not treated as a threat.") if music else ""
        if music and risk.threat >= 0.5:
            events.append(Event(new_id("evt"), "hateful_speech" if risk.hate >= 0.5 else "threatening_speech", start, end, speaker,
                                "SUNG_LYRICS", None, result.location, 0.3, 0.2,
                                f"Song lyrics in {lang} contain violent/offensive wording (threat score {risk.threat:.2f})." + music_note
                                + f" English: \"{english[:160]}\"", ids,
                                {"language": lang, "original": original, "english": english, "cues": risk.matched_cues, "music": True}))
        elif risk.threat >= 0.5:
            events.append(Event(new_id("evt"), "threatening_speech", start, end, speaker, "THREATENED", "TARGET_PERSON",
                                result.location, min(0.95, risk.threat), SEVERITY_PRIOR["threatening_speech"],
                                f"Speech in {lang} contains explicit threat language (score {risk.threat:.2f}; cues: "
                                f"{', '.join(risk.matched_cues[:3]) or 'model-only'}). English: \"{english[:200]}\"", ids,
                                {"language": lang, "original": original, "english": english, "cues": risk.matched_cues}))
        elif music:
            pass
        elif risk.hate >= 0.5 and risk.threat < 0.5:
            events.append(Event(new_id("evt"), "hateful_speech", start, end, speaker, "HATE_SPEECH", "GROUP",
                                result.location, min(0.9, risk.hate), SEVERITY_PRIOR["hateful_speech"],
                                f"Speech in {lang} contains hateful language (score {risk.hate:.2f}). English: \"{english[:200]}\"",
                                ids, {"language": lang, "original": original, "english": english, "cues": risk.matched_cues}))
        argument = any(a.label in {"angry_argument", "shouting"} for a in result.acoustic_events)
        if (not music) and risk.threat < 0.5 and risk.hate < 0.5 and (risk.toxicity >= 0.55 or (argument and risk.toxicity >= 0.35)):
            events.append(Event(new_id("evt"), "abusive_speech", start, end, speaker, "ABUSED", "TARGET_PERSON",
                                result.location, min(0.9, 0.4 + 0.5 * risk.toxicity), SEVERITY_PRIOR["abusive_speech"],
                                f"Speech in {lang} is abusive/insulting (toxicity {risk.toxicity:.2f}" + (", raised voices" if argument else "")
                                + f") without an explicit threat to life. English: \"{english[:200]}\"", ids,
                                {"language": lang, "original": original, "english": english, "toxicity": risk.toxicity, "argument": argument}))
        if distress:
            events.append(Event(new_id("evt"), "distress_speech", start, end, speaker, "REPORTED_DISTRESS", None,
                                result.location, 0.7, SEVERITY_PRIOR["distress_speech"],
                                f"Speaker said phrases consistent with distress: {', '.join(distress[:4])}. "
                                f"English: \"{english[:160]}\"", ids,
                                {"language": lang, "cues": distress, "original": original, "english": english}))
    for ac in result.acoustic_events:
        aid = new_id("ev")
        evidence[aid] = Evidence(aid, "acoustic", ac.start_s, ac.end_s, f"Acoustic event: {ac.label} ({ac.confidence:.2f})",
                                 None, ac.confidence, ac.to_dict())
        sev = ACOUSTIC_SEVERITY.get(ac.label, 0.3)
        if ac.label in {"music", "singing"}:
            continue
        if sev >= 0.35 or ac.label in {"baby_crying", "crying"}:
            events.append(Event(new_id("evt"), "acoustic_alarm", ac.start_s, ac.end_s, "ENVIRONMENT", "SOUND_OF",
                                ac.label.upper(), result.location, ac.confidence, sev,
                                f"Sound classified as {ac.label} at {ac.start_s:.1f}s (confidence {ac.confidence:.2f}).",
                                [aid], ac.to_dict()))

    if result.media_kind == "video" and not any(v.kind == "frame" for v in evidence.values()):
        for f in (0.15, 0.5, 0.85):
            frame_evidence(result.duration_s * f, "Context frame", 0.1)
        ctx_ids = [k for k, v in evidence.items() if v.kind == "frame"]
        for e in events:
            e.evidence_ids.extend(ctx_ids)

    _consolidate_group_fight(events, result)
    _fold_gunshot_into_weapon(events)

    if not events:
        ids = [i for i in (frame_evidence(result.duration_s * f, "Coverage sample", 0.1) for f in (0.1, 0.5, 0.9)) if i]
        people = sorted({d.track_id for o in result.scan for d in o.detections if d.label == "person" and d.track_id is not None})
        music = getattr(result, "audio_kind", "none") in {"music", "mixed"}
        extra = ""
        if music:
            lyric = " ".join(s.translation_en or s.text for s in result.audio_segments)[:120]
            extra = f" The audio is music/singing (lyrics transcribed: \"{lyric}\") — a melody, not a conversation."
        elif result.audio_segments:
            extra = f" Speech was transcribed ({result.audio_segments[0].language}) and contains no threat, abuse or distress."
        events.append(Event(new_id("evt"), "normal_activity", 0.0, result.duration_s,
                            _person(people[0]) if people else "NOBODY", "NORMAL_ACTIVITY", None, result.location, 0.6,
                            0.0, f"No fall, weapon, aggressive motion, distress or threatening speech was detected; "
                                 f"{len(people)} person track(s) observed." + extra, ids,
                            {"people": [_person(p) for p in people], "music": music}))
    events.sort(key=lambda e: (e.start_s, -e.severity))
    return events, evidence
