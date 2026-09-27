"""Multilingual voicemail demo clips + dubbed demo videos (real threat text, local pipeline verified).

Why this exists: public *audio* of real threatening phone calls barely exists in any language, so the
demo voices REAL threat sentences (HateCheck multilingual test suites, CC BY 4.0 — Spanish, French,
German, Italian, Portuguese, Arabic, Hindi, Mandarin, Polish, Dutch) plus a few scripted lines for
languages HateCheck does not cover (Russian, Turkish, Persian, Japanese, Korean). Every clip gets a
phone-line sound (300-3400 Hz band, light compression). The investigator then has to: detect the
language, transcribe, translate to English, score threat/hate, and decide — all locally.

TTS engines tried in order: edge-tts (needs internet ONLY while generating; not used by the app),
kokoro (offline, Python <= 3.12), Windows SAPI (offline, installed voices only).

    python scripts/make_multilingual_voicemails.py --out data/demo             # all languages + dubbed videos
    python scripts/make_multilingual_voicemails.py --langs es,fr,fa --no-dub    # subset

Also writes two dubbed demo videos when the source clips exist in --out (Pexels, Pexels License):
  car_passenger_unwell_spanish_dubbed.mp4   = people in a car + Spanish distress speech ("me duele el pecho...")
  baby_in_crib_crying_dubbed.mp4            = baby in a crib + real ESC-50 crying audio
Dubbed clips are labelled as such in library.json.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "Mozilla/5.0 PersonalAngel-demo/1.0"}

LANGS: dict[str, dict] = {
    "es": {"voice": "es-MX-JorgeNeural", "hatecheck": "Paul/hatecheck-spanish", "name": "Spanish",
           "benign": "Hola, soy Marcos del taller. Su coche ya está listo, puede pasar a recogerlo mañana por la mañana. Gracias.",
           "threat": "Escúchame bien. Sé dónde vives y a qué hora sales. Si vuelves a hablar con la policía te voy a hacer daño. No es una broma."},
    "fr": {"voice": "fr-FR-HenriNeural", "hatecheck": "Paul/hatecheck-french", "name": "French",
           "benign": "Bonjour, c'est Julie de la pharmacie. Votre commande est prête, vous pouvez passer la chercher quand vous voulez.",
           "threat": "Écoute-moi bien. Je sais où tu habites. Si tu parles encore à la police, je vais te faire du mal. Ce n'est pas une blague."},
    "de": {"voice": "de-DE-ConradNeural", "hatecheck": "Paul/hatecheck-german", "name": "German",
           "benign": "Hallo, hier ist Thomas vom Büro. Das Meeting wurde auf Donnerstag um zehn verschoben. Bis dann.",
           "threat": "Hör mir gut zu. Ich weiß, wo du wohnst. Wenn du noch einmal mit der Polizei sprichst, tue ich dir weh. Das ist kein Scherz."},
    "it": {"voice": "it-IT-DiegoNeural", "hatecheck": "Paul/hatecheck-italian", "name": "Italian",
           "benign": "Ciao, sono Luca del ristorante. La prenotazione per sabato sera è confermata, a presto.",
           "threat": "Ascoltami bene. So dove abiti. Se parli ancora con la polizia ti faccio del male. Non è uno scherzo."},
    "pt": {"voice": "pt-BR-AntonioNeural", "hatecheck": "Paul/hatecheck-portuguese", "name": "Portuguese",
           "benign": "Oi, é o Rafael da oficina. Seu carro está pronto, pode buscar amanhã de manhã. Obrigado.",
           "threat": "Me escuta bem. Eu sei onde você mora. Se falar com a polícia de novo, eu vou te machucar. Não é brincadeira."},
    "ar": {"voice": "ar-EG-ShakirNeural", "hatecheck": "Paul/hatecheck-arabic", "name": "Arabic",
           "benign": "مرحبا، أنا أحمد من الصيدلية. طلبك جاهز، يمكنك استلامه في أي وقت. شكرا.",
           "threat": "اسمعني جيدا. أنا أعرف أين تسكن. إذا تحدثت مع الشرطة مرة أخرى سأؤذيك. هذه ليست مزحة."},
    "hi": {"voice": "hi-IN-MadhurNeural", "hatecheck": "Paul/hatecheck-hindi", "name": "Hindi",
           "benign": "नमस्ते, मैं राहुल बोल रहा हूँ। आपकी गाड़ी तैयार है, कल सुबह ले जा सकते हैं। धन्यवाद।",
           "threat": "ध्यान से सुनो। मुझे पता है तुम कहाँ रहते हो। अगर तुमने फिर पुलिस से बात की तो मैं तुम्हें नुकसान पहुँचाऊँगा। यह मज़ाक नहीं है।"},
    "zh": {"voice": "zh-CN-YunxiNeural", "hatecheck": "Paul/hatecheck-mandarin", "name": "Mandarin",
           "benign": "你好，我是修车厂的小王。您的车已经修好了，明天上午可以来取。谢谢。",
           "threat": "你给我听好了。我知道你住在哪里。你再跟警察说话，我就伤害你。这不是开玩笑。"},
    "ru": {"voice": "ru-RU-DmitryNeural", "hatecheck": None, "name": "Russian",
           "benign": "Здравствуйте, это Андрей из автосервиса. Ваша машина готова, можете забрать завтра утром. Спасибо.",
           "threat": "Слушай меня внимательно. Я знаю, где ты живёшь. Если ещё раз поговоришь с полицией, я сделаю тебе больно. Это не шутка."},
    "tr": {"voice": "tr-TR-AhmetNeural", "hatecheck": None, "name": "Turkish",
           "benign": "Merhaba, ben servisten Mehmet. Arabanız hazır, yarın sabah alabilirsiniz. Teşekkürler.",
           "threat": "Beni iyi dinle. Nerede oturduğunu biliyorum. Bir daha polisle konuşursan sana zarar veririm. Bu şaka değil."},
    "fa": {"voice": "fa-IR-FaridNeural", "hatecheck": None, "name": "Persian",
           "benign": "سلام، من رضا از تعمیرگاه هستم. ماشین شما آماده است، فردا صبح می‌توانید تحویل بگیرید. ممنون.",
           "threat": "خوب گوش کن. من می‌دانم کجا زندگی می‌کنی. اگر دوباره با پلیس حرف بزنی، بلایی سرت می‌آورم. این شوخی نیست."},
    "ja": {"voice": "ja-JP-KeitaNeural", "hatecheck": None, "name": "Japanese",
           "benign": "もしもし、修理工場の田中です。お車の修理が終わりました。明日の午前中に取りに来てください。",
           "threat": "よく聞け。お前の住所は知っている。また警察に話したら、痛い目に遭わせる。冗談じゃない。"},
    "en": {"voice": "en-US-GuyNeural", "hatecheck": "Paul/hatecheck", "name": "English",
           "benign": "Hi, this is Mark from the garage. Your car is ready, you can pick it up tomorrow morning. Thanks.",
           "threat": "Listen carefully. I know where you live and I know where your kids go to school. If you show up at work tomorrow, I will hurt you. This is not a joke."},
}
SPANISH_DISTRESS = ("Disculpe... no me siento bien. Me duele mucho el pecho y me cuesta respirar. "
                    "Creo que me voy a desmayar. Por favor, ayúdeme.")

ABUSE_DIALOGUE = {
    "en": [(0, "You are worthless, you always ruin everything. Nobody here can stand you."),
           (1, "Please stop talking to me like that."),
           (0, "Shut up. You are pathetic and stupid, and everyone knows it."),
           (1, "I'm leaving now.")],
    "es": [(0, "Eres un inútil, siempre lo arruinas todo. Nadie aquí te soporta."),
           (1, "Por favor, no me hables así."),
           (0, "Cállate. Eres patético y estúpido, y todos lo saben."),
           (1, "Me voy ahora mismo.")],
}

def hatecheck_derogatory(repo: str, cache: Path) -> list[str]:
    """Real derogatory (non-threat) sentences from a HateCheck suite (functionality 'derog_neg_emote_h')."""
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / (repo.split("/")[-1] + ".csv")
    if not dest.exists():
        return []
    out = []
    with open(dest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("functionality", "")).lower().startswith("derog_neg_emote") and str(row.get("label_gold", "")).lower() == "hateful":
                text = str(row.get("test_case", "")).strip()
                if 15 <= len(text) <= 120:
                    out.append(text)
    return out[:1]

def synth_voice(text: str, voice: str, lang: str, out_wav: Path, tmp: Path) -> str | None:
    """Like synth() but with an explicit voice and no phone filter (room conversation)."""
    tmp.mkdir(parents=True, exist_ok=True)
    mp3 = tmp / (out_wav.stem + ".mp3")
    try:
        asyncio.run(_edge(text, voice, mp3))
        if mp3.exists() and mp3.stat().st_size > 1000 and shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp3), "-ar", "16000", "-ac", "1", str(out_wav)], check=True)
            return "edge-tts"
    except Exception as error:
        print(f"    (edge-tts unavailable: {str(error)[:80]})")
    return None

def concat_wavs(parts: list[Path], dest: Path) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    lst = dest.parent / ".tts" / "concat.txt"
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-af", "apad=pad_dur=0.4", "-ar", "16000", "-ac", "1", str(dest)]
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0 and dest.exists()

def hatecheck_threats(repo: str, cache: Path) -> list[str]:
    """Real direct-threat sentences from a HateCheck test suite (functionality 'threat_dir_h')."""
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / (repo.split("/")[-1] + ".csv")
    if not dest.exists():
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/test.csv"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
        except Exception as error:
            print(f"    ({repo}: {error})")
            return []
    out = []
    with open(dest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            func = str(row.get("functionality", "")).lower()
            if func.startswith("threat_dir") and str(row.get("label_gold", "")).lower() == "hateful":
                text = str(row.get("test_case", "")).strip()
                if 20 <= len(text) <= 160:
                    out.append(text)
    return out[:3]

async def _edge(text: str, voice: str, mp3: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice).save(str(mp3))

def synth(text: str, lang: str, out_wav: Path, tmp: Path) -> str | None:
    """Returns the engine name used, or None."""
    tmp.mkdir(parents=True, exist_ok=True)
    voice = LANGS[lang]["voice"]
    mp3 = tmp / (out_wav.stem + ".mp3")
    try:
        asyncio.run(_edge(text, voice, mp3))
        if mp3.exists() and mp3.stat().st_size > 1000:
            phone_filter(mp3, out_wav)
            return "edge-tts"
    except Exception as error:
        print(f"    (edge-tts unavailable: {str(error)[:80]})")
    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline

        code = {"es": "e", "fr": "f", "it": "i", "pt": "p", "zh": "z", "ja": "j", "hi": "h", "en": "a"}.get(lang)
        if code:
            pipe = KPipeline(lang_code=code)
            chunks = [a for _, _, a in pipe(text) if a is not None]
            raw = tmp / (out_wav.stem + ".raw.wav")
            sf.write(raw, np.concatenate([np.asarray(a, dtype=np.float32) for a in chunks]), 24000)
            phone_filter(raw, out_wav)
            return "kokoro"
    except Exception as error:
        print(f"    (kokoro unavailable: {str(error)[:80]})")
    if sys.platform == "win32":
        try:
            raw = tmp / (out_wav.stem + ".sapi.wav")
            ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.SetOutputToWaveFile('{raw}'); $s.Speak([Console]::In.ReadToEnd()); $s.Dispose()")
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], input=text, text=True, encoding="utf-8", check=True, timeout=120)
            phone_filter(raw, out_wav)
            return "windows-sapi"
        except Exception as error:
            print(f"    (Windows SAPI unavailable: {str(error)[:80]})")
    return None

def phone_filter(src: Path, dest: Path) -> None:
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af",
                        "highpass=f=300,lowpass=f=3400,acompressor=threshold=-18dB:ratio=3,volume=1.4", "-ar", "16000", "-ac", "1", str(dest)],
                       check=True)
    else:
        shutil.copy(src, dest)

def dub_video(video: Path, audio: Path, dest: Path, loop_audio: bool = False) -> bool:
    if not (video.exists() and audio.exists() and shutil.which("ffmpeg")):
        return False
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video)]
    if loop_audio:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-shortest", str(dest)]
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0 and dest.exists()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "demo")
    parser.add_argument("--langs", default=",".join(LANGS))
    parser.add_argument("--no-dub", action="store_true")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / ".tts"
    cache = out / ".hatecheck"
    library_path = out / "library.json"
    library = {"items": []}
    if library_path.exists():
        try:
            library = json.loads(library_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    items = {m["file"]: m for m in library.get("items", [])}
    engines = set()
    for lang in [x.strip() for x in args.langs.split(",") if x.strip() in LANGS]:
        spec = LANGS[lang]
        real = hatecheck_threats(spec["hatecheck"], cache) if spec["hatecheck"] else []
        threat_text = (" ".join(real) + " " + spec["threat"]) if real else spec["threat"]
        source = (f"HateCheck ({spec['hatecheck']}, CC BY 4.0) threat sentences + scripted line" if real else "scripted threat (no public dataset for this language)")
        for kind, text, src in (("threat", threat_text, source), ("benign", spec["benign"], "scripted benign message")):
            name = f"voicemail_{lang}_{kind}.wav"
            dest = out / name
            print(f"[{spec['name']}] {kind} -> {name}")
            if dest.exists() and dest.stat().st_size > 2000:
                print("    already present")
            else:
                engine = synth(text, lang, dest, tmp)
                if not engine:
                    print("    ✖ no TTS engine could voice this language (pip install edge-tts)")
                    continue
                engines.add(engine)
                print(f"    ✔ voiced with {engine}")
            items[name] = {"file": name, "title": f"Voicemail — {'threatening' if kind == 'threat' else 'benign'} message in {spec['name']}",
                           "group": "voicemail", "source": f"{src}; voiced by local/edge TTS", "license": "CC BY 4.0 (text) / demo",
                           "expected": "ALERT: threatening_speech, translated to English, police report (simulated)" if kind == "threat" else "CLEAR"}

    for lang, voices, lines in (
        ("en", ("en-US-GuyNeural", "en-US-JennyNeural"), None),
        ("es", ("es-MX-JorgeNeural", "es-MX-DaliaNeural"), None),
    ):
        spec = LANGS[lang]
        derog = hatecheck_derogatory(spec["hatecheck"], cache) if spec.get("hatecheck") else []
        script = ABUSE_DIALOGUE[lang]
        if derog:
            script = [(0, derog[0])] + script
        name = f"abusive_conversation_{lang}.wav"
        dest = out / name
        print(f"[{spec['name']}] abusive conversation -> {name}")
        if dest.exists() and dest.stat().st_size > 2000:
            print("    already present")
        else:
            parts = []
            for i, (who, text) in enumerate(script):
                part = tmp / f"abuse_{lang}_{i}.wav"
                engine = synth_voice(text, voices[who], lang, part, tmp)
                if engine:
                    parts.append(part)
                    engines.add(engine)
            if parts and concat_wavs(parts, dest):
                print("    ✔ built")
            else:
                print("    ✖ could not build (TTS unavailable)")
        if dest.exists():
            items[name] = {"file": name, "title": f"Abusive conversation in {spec['name']} (insults, no threat to life; acted from HateCheck sentences)",
                           "group": "voicemail", "source": "HateCheck derogatory sentences (CC BY 4.0) + scripted lines; two TTS voices", "license": "CC BY 4.0 (text) / demo",
                           "expected": "WATCH: abusive_speech -> asks 'do you want me to report it?' -> yes: report with transcript / no: safety advice (no 911)"}
    if not args.no_dub:
        print("[dub] Spanish distress speech onto the car clip")
        speech = out / ".tts" / "es_distress.wav"
        if synth(SPANISH_DISTRESS, "es", speech, tmp):
            car = out / "pexels_couple_in_car_6788032.mp4"
            dest = out / "car_passenger_unwell_spanish_dubbed.mp4"
            if dub_video(car, speech, dest):
                items[dest.name] = {"file": dest.name, "title": "Car cabin — passenger says she feels unwell (Spanish speech dubbed for demo)",
                                    "group": "vehicle", "source": "Pexels #6788032 video + scripted Spanish distress speech (TTS)", "license": "Pexels License / demo",
                                    "expected": "asks in Spanish whether to go to the ER; silence -> reroute + 911 (simulated)"}
                print(f"    ✔ {dest.name}")
        print("[dub] real ESC-50 crying onto the crib clip")
        crib, cry = out / "pexels_baby_in_crib_16418098.mp4", out / "esc50_baby_crying_1.wav"
        dest = out / "baby_in_crib_crying_dubbed.mp4"
        if dub_video(crib, cry, dest, loop_audio=True):
            items[dest.name] = {"file": dest.name, "title": "Nursery — baby in a crib, crying (real ESC-50 crying audio dubbed for demo)",
                                "group": "nursery", "source": "Pexels #16418098 video + ESC-50 crying_baby clip (CC BY-NC)", "license": "Pexels License / CC BY-NC",
                                "expected": "infant_distress -> notify parents (simulated)"}
            print(f"    ✔ {dest.name}")

    fixes = {"pexels_man_unwell_in_car_8641684.mp4": "Car cabin — man sitting in a car, talking (Pexels; negative control, no incident)",
             "pexels_couple_in_car_6788032.mp4": "Car cabin — two people sitting in a car (Pexels; negative control)"}
    for f, title in fixes.items():
        if f in items:
            items[f]["title"] = title
    library["items"] = list(items.values())
    library["generated"] = time.strftime("%Y-%m-%d %H:%M")
    library_path.write_text(json.dumps(library, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\ndone -> {out} (engines used: {', '.join(sorted(engines)) or 'none'}); library.json updated")

if __name__ == "__main__":
    main()
