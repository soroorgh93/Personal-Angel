"""Audio perception: multilingual ASR → English translation → text-risk
classification → acoustic event detection.

Real backends
  * faster-whisper (CTranslate2) for transcription (auto language) and the
    built-in `translate` task (X→English).
  * Detoxify multilingual (unitary/multilingual-toxic-xlm-roberta) exposing a
    `threat` score; optional Spanish threat model
    (insiktml/threat_detection_xmlRoberta_ES) via transformers.
  * CLAP (laion/clap-htsat-unfused) zero-shot audio classification for
    scream / gunshot / baby crying / glass breaking / moaning.
A transparent multilingual cue matcher (en/es) always runs so the report can
show *which phrase* triggered a threat hypothesis.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..schema import AcousticEvent, AudioSegment, TextRiskScores

log = logging.getLogger(__name__)

THREAT_CUES: dict[str, list[str]] = {
    "en": [
        r"\bi(?:'m| am| will|'ll) (?:going to |gonna )?(?:kill|hurt|shoot|stab|beat|burn|destroy) (?:you|him|her|them|your)\b",
        r"\b(?:kill|shoot|stab|hurt|beat|burn)\s+(?:you|him|her|them|your (?:family|kids|children|wife|husband))\b",
        r"\byou(?:'re| are) (?:going to |gonna )?(?:die|pay|regret)\b",
        r"\bi know where you live\b", r"\bwatch your back\b", r"\bi have a (?:gun|knife|weapon)\b",
        r"\bi(?:'ll| will) find you\b", r"\bthis is your last (?:warning|chance)\b",
    ],
    "es": [
        r"\b(?:te|los|las|le|lo) (?:voy a|vamos a) (?:matar|lastimar|golpear|apu[nñ]alar|disparar|quemar|destruir)\b",
        r"\bvoy a (?:matar|lastimar|golpear|apu[nñ]alar|disparar)(?:te|lo|la|los|las)?\b",
        r"\bte (?:mato|mataré|matare|voy a matar|lastimo|golpeo)\b",
        r"\bs[eé] d[oó]nde vives\b", r"\bvas a (?:morir|pagar|arrepentirte)\b",
        r"\btengo (?:una|un) (?:pistola|arma|cuchillo|navaja)\b",
        r"\b(?:cuídate|cuidate) (?:la )?espalda\b", r"\b(?:te voy a|voy a) encontrar\b",
        r"\b(?:última|ultima) (?:advertencia|oportunidad)\b", r"\ba tu familia\b",
    ],
}
HATE_CUES: dict[str, list[str]] = {
    "en": [r"\b(?:all|every) (?:of )?(?:you|them|those) (?:people|kind) (?:should|deserve to) (?:die|burn|suffer)\b",
           r"\bgo back to (?:your|where you)\b", r"\b(?:sub-?human|vermin|filth)\b"],
    "es": [r"\btodos (?:ustedes|esos|esas) (?:deber[ií]an|merecen) (?:morir|sufrir)\b",
           r"\bregresa a tu pa[ií]s\b", r"\b(?:escoria|basura humana)\b"],
}
DISTRESS_CUES: dict[str, list[str]] = {
    "en": [r"\bi (?:don't|do not|can't|cannot) (?:feel (?:well|good)|breathe)\b", r"\bhelp me\b",
           r"\bcall (?:911|an ambulance|a doctor)\b", r"\bit hurts\b", r"\bi(?:'m| am) (?:in pain|pregnant|bleeding|dizzy)\b",
           r"\bsomething(?:'s| is) wrong\b", r"\bmy (?:chest|stomach|belly|baby)\b", r"\bcontractions?\b"],
    "es": [r"\bno me siento bien\b", r"\bay[uú]d(?:a|e)me\b", r"\bllam(?:a|e) (?:a )?(?:una ambulancia|al 911|a un m[eé]dico)\b",
           r"\bme duele\b", r"\bestoy (?:embarazada|sangrando|mareada|mareado)\b", r"\balgo (?:anda|est[aá]) mal\b",
           r"\bcontracciones\b", r"\bno puedo respirar\b"],
}

def match_cues(text: str, table: dict[str, list[str]]) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for patterns in table.values():
        for pattern in patterns:
            for m in re.finditer(pattern, lowered):
                found.append(m.group(0))

    seen: set[str] = set()
    return [f for f in found if not (f in seen or seen.add(f))]

class AudioAnalyzer:
    name = "abstract"

    def analyze(self, wav_path: Path, hints: dict[str, Any] | None = None
                ) -> tuple[list[AudioSegment], list[AcousticEvent], TextRiskScores]:
        raise NotImplementedError

def rule_risk(segments: list[AudioSegment]) -> TextRiskScores:
    original = " ".join(s.text for s in segments)
    english = " ".join(s.translation_en or "" for s in segments)
    threat_cues = match_cues(original, THREAT_CUES) + match_cues(english, THREAT_CUES)
    hate_cues = match_cues(original, HATE_CUES) + match_cues(english, HATE_CUES)
    threat = min(0.95, 0.35 + 0.3 * len(set(threat_cues))) if threat_cues else 0.05
    hate = min(0.9, 0.4 + 0.25 * len(set(hate_cues))) if hate_cues else 0.05
    return TextRiskScores(threat=threat, hate=hate, toxicity=max(threat, hate), model="rule_cues",
                          matched_cues=list(dict.fromkeys(threat_cues + hate_cues)))

class FixtureAudioAnalyzer(AudioAnalyzer):
    name = "fixture_audio"

    def __init__(self, fixture: dict[str, Any]) -> None:
        self.spec = fixture.get("audio", {})

    def analyze(self, wav_path, hints=None):
        segments = [AudioSegment(**{k: v for k, v in s.items() if k in AudioSegment.__dataclass_fields__})
                    for s in self.spec.get("segments", [])]
        acoustic = [AcousticEvent(**a) for a in self.spec.get("acoustic", [])]
        risk = rule_risk(segments)
        override = self.spec.get("risk")
        if override:
            risk = TextRiskScores(threat=float(override.get("threat", risk.threat)),
                                  hate=float(override.get("hate", risk.hate)),
                                  toxicity=float(override.get("toxicity", risk.toxicity)),
                                  self_harm=float(override.get("self_harm", 0.0)),
                                  model=str(override.get("model", "fixture_classifier")),
                                  matched_cues=risk.matched_cues)
        return segments, acoustic, risk

class FasterWhisperAnalyzer(AudioAnalyzer):
    name = "faster_whisper"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._asr = None
        self._threat = None
        self._clap = None

    def warm_up(self) -> None:
        """Load whisper, the text-risk model and CLAP now, so the first recording does not pay ~20 s of loading."""
        self._load_asr()
        self._load_threat()
        self._load_clap()

    def _load_asr(self, force_cpu: bool = False):
        if self._asr is None or force_cpu:
            from faster_whisper import WhisperModel

            device = self.config.get("device", "auto")
            if device == "auto":
                device = "cuda" if _cuda_available() else "cpu"
            model_name = str(self.config.get("asr_model", "small"))
            if device == "cuda" and (force_cpu or not _ctranslate2_cuda()):

                fallback = str(self.config.get("cpu_fallback_model", "medium" if model_name.startswith("large") else model_name))
                log.warning("faster-whisper: CTranslate2 has no CUDA support here; using CPU int8 with model '%s'", fallback)
                device, model_name = "cpu", fallback
                self.asr_note = f"whisper on CPU ({fallback}): CTranslate2 without CUDA"
            compute = self.config.get("compute_type", "int8" if device == "cpu" else "float16")
            if device == "cpu" and str(compute).startswith("float16"):
                compute = "int8"
            self._asr = WhisperModel(model_name, device=device, compute_type=compute)
            self.name = f"faster_whisper_{model_name}"
        return self._asr

    def _transcribe(self, wav_path: Path, task: str, language: str | None = None):
        model = self._load_asr()
        try:
            segments, info = model.transcribe(str(wav_path), task=task, language=language,
                                              vad_filter=True, beam_size=5, word_timestamps=False)
            return list(segments), info
        except (ValueError, RuntimeError) as error:
            if "CUDA" not in str(error) and "cuda" not in str(error):
                raise
            log.warning("faster-whisper failed on the GPU (%s); retrying on CPU", error)
            model = self._load_asr(force_cpu=True)
            segments, info = model.transcribe(str(wav_path), task=task, language=language,
                                              vad_filter=True, beam_size=5, word_timestamps=False)
            return list(segments), info

    def _load_threat(self):
        if self._threat is not None:
            return self._threat
        kind = str(self.config.get("threat_classifier", "detoxify_multilingual"))
        try:
            if kind == "detoxify_multilingual":
                from detoxify import Detoxify

                model = Detoxify("multilingual")
                self._threat = ("detoxify_multilingual", lambda text: model.predict(text))
            elif kind == "insikt_es":
                from transformers import pipeline

                clf = pipeline("text-classification", model="insiktml/threat_detection_xmlRoberta_ES",
                               top_k=None, truncation=True)
                self._threat = ("insikt_es", lambda text: {d["label"].lower(): d["score"] for d in clf(text)[0]})
            else:
                self._threat = ("none", None)
        except Exception as error:
            log.warning("Threat classifier unavailable (%s); using rule cues only", error)
            self._threat = ("none", None)
        return self._threat

    def _load_clap(self):
        if self._clap is not None or str(self.config.get("acoustic_backend", "none")) != "clap":
            return self._clap
        try:
            from transformers import pipeline

            self._clap = pipeline("zero-shot-audio-classification", model="laion/clap-htsat-unfused")
        except Exception as error:
            log.warning("CLAP unavailable (%s)", error)
            self._clap = False
        return self._clap

    def analyze(self, wav_path, hints=None):
        raw, info = self._transcribe(wav_path, task="transcribe")
        language = getattr(info, "language", "unknown") or "unknown"
        import math as _math

        segments = [AudioSegment(start_s=float(s.start), end_s=float(s.end), text=s.text.strip(), language=language,
                                 confidence=round(float(_math.exp(min(0.0, float(getattr(s, "avg_logprob", 0.0) or 0.0)))), 3))
                    for s in raw if s.text.strip()]
        if segments and language != "en" and bool(self.config.get("translate", True)):
            translated, _ = self._transcribe(wav_path, task="translate", language=language)

            for seg in segments:
                best = None
                for tr in translated:
                    overlap = min(seg.end_s, tr.end) - max(seg.start_s, tr.start)
                    if overlap > 0 and (best is None or overlap > best[0]):
                        best = (overlap, tr.text.strip())
                seg.translation_en = best[1] if best else None
            if all(s.translation_en is None for s in segments) and translated:
                segments[0].translation_en = " ".join(t.text.strip() for t in translated)
        elif language == "en":
            for seg in segments:
                seg.translation_en = seg.text
        risk = rule_risk(segments)
        name, clf = self._load_threat()
        if clf is not None and segments:
            text_for_model = " ".join(s.text for s in segments)
            english = " ".join(s.translation_en or "" for s in segments).strip()
            try:
                scores = clf(text_for_model)
                if name == "detoxify_multilingual":

                    if english and language != "en":
                        try:
                            en_scores = clf(english)
                            scores = {k: max(float(scores.get(k, 0.0)), float(en_scores.get(k, 0.0))) for k in set(scores) | set(en_scores)}
                        except Exception as error:
                            log.warning("threat scoring of the translation failed: %s", error)
                    risk = TextRiskScores(
                        threat=max(risk.threat, float(scores.get("threat", 0.0))),
                        hate=max(risk.hate, float(scores.get("identity_attack", 0.0))),
                        toxicity=float(scores.get("toxicity", 0.0)), model=name + "+translation",
                        matched_cues=risk.matched_cues)

                    if risk.matched_cues:
                        risk.threat = max(risk.threat, 0.6)
                else:
                    threat_score = float(scores.get("threat", scores.get("label_1", 0.0)))
                    risk = TextRiskScores(threat=max(risk.threat, threat_score), hate=risk.hate,
                                          toxicity=max(risk.toxicity, threat_score), model=name,
                                          matched_cues=risk.matched_cues)
            except Exception as error:
                log.warning("Threat classifier failed: %s", error)
        acoustic: list[AcousticEvent] = []
        clap = self._load_clap()
        if clap:
            labels = list(self.config.get("acoustic_labels", []))
            try:
                import numpy as np
                import soundfile as sf

                audio, sr = sf.read(str(wav_path))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                target_sr = int(getattr(getattr(clap, "feature_extractor", None), "sampling_rate", 48000))
                window = int(sr * 5)
                for start in range(0, max(len(audio), 1), window):
                    chunk = np.asarray(audio[start:start + window], dtype=np.float32)
                    if len(chunk) < sr:
                        continue
                    if sr != target_sr:
                        import librosa

                        chunk = librosa.resample(chunk, orig_sr=sr, target_sr=target_sr)
                    out = clap(chunk, candidate_labels=labels)
                    top = out[0]
                    if top["label"] in {"music", "singing"} and top["score"] > 0.4:
                        acoustic.append(AcousticEvent(start_s=start / sr, end_s=(start + len(chunk)) / sr,
                                                      label=top["label"].replace(" ", "_"), confidence=round(float(top["score"]), 3)))
                        continue
                    if top["label"] not in {"speech", "silence"} and top["score"] > 0.45:
                        acoustic.append(AcousticEvent(start_s=start / sr, end_s=(start + len(chunk)) / sr,
                                                      label=top["label"].replace(" ", "_"),
                                                      confidence=float(top["score"])))
            except Exception as error:
                log.warning("Acoustic event detection failed: %s", error)
        return segments, acoustic, risk

def _ctranslate2_cuda() -> bool:
    """faster-whisper runs on CTranslate2, whose CUDA support is separate from torch's."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:
        return False

def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False

def create_audio_analyzer(config: dict[str, Any], fixture: dict[str, Any] | None = None) -> AudioAnalyzer:
    backend = str(config.get("backend", "fixture"))
    if backend == "fixture":
        return FixtureAudioAnalyzer(fixture or {})
    if backend == "faster_whisper":
        from .registry import cached

        return cached("audio", config, lambda: FasterWhisperAnalyzer(config))
    raise ValueError(f"Unknown audio backend: {backend}")
