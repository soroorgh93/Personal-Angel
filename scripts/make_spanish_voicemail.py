"""Synthesize the demo voicemails with an open TTS (Kokoro-82M, Apache-2.0):
a threatening Spanish message, a benign Spanish control and an English control,
then apply a telephone band-limit + noise so they sound like real voicemail.

  pip install kokoro>=0.9.2 soundfile   (Spanish also needs the espeak-ng system package: apt/brew/winget install espeak-ng)
  python scripts/make_spanish_voicemail.py --out data/demo_audio
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCRIPTS = {
    "voicemail_threat_es": ("e", "em_alex",
        "Escúchame bien. Sé dónde vives y sé a qué hora sales de tu casa. "
        "Si no me pagas mañana, te voy a matar, a ti y a tu familia. Esta es tu última advertencia."),
    "voicemail_benign_es": ("e", "ef_dora",
        "Hola, soy Guillermo de la clínica. Le llamo para confirmar su cita del jueves a las diez. "
        "Si necesita cambiarla, llámenos por favor. Que tenga un buen día."),
    "voicemail_threat_en": ("a", "am_michael",
        "Listen carefully. I know where you live and I know when you leave. "
        "If you don't pay me tomorrow, I will hurt you and your family. This is your last warning."),
    "passenger_distress_es": ("e", "ef_dora",
        "No me siento bien... me duele mucho el vientre. Creo que son contracciones. Por favor, ayúdeme."),
}

def telephone_effect(audio: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    sos = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    y = sosfilt(sos, audio)
    y = y + np.random.default_rng(0).normal(0, 0.004, size=y.shape)
    y = np.tanh(y * 1.4)
    return (y / (np.abs(y).max() + 1e-6) * 0.9).astype(np.float32)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/demo_audio")
    parser.add_argument("--no-phone-effect", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    from kokoro import KPipeline

    pipelines: dict[str, KPipeline] = {}
    for name, (lang, voice, text) in SCRIPTS.items():
        pipe = pipelines.setdefault(lang, KPipeline(lang_code=lang))
        chunks = [audio for _, _, audio in pipe(text, voice=voice, speed=1.0) if audio is not None]
        audio = np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
        sr = 24000
        if not args.no_phone_effect:
            audio = telephone_effect(audio, sr)
        path = out / f"{name}.wav"
        sf.write(str(path), audio, sr)
        (out / f"{name}.txt").write_text(text, encoding="utf-8")
        print(f"  ✔ {path} ({len(audio) / sr:.1f}s)")

if __name__ == "__main__":
    main()
