"""Web backend (FastAPI when installed, plain Starlette otherwise — same routes).

Routes
  GET  /                          UI
  GET  /api/health                profile + LLM endpoint health
  POST /api/upload                multipart media → media_id
  POST /api/runs                  start an investigation → run_id (SSE at /api/runs/{id}/events)
  GET  /api/runs/{id}/events      Server-Sent Events stream of the agent loop
  POST /api/runs/{id}/answer      the person's answer to an ask_user question
  GET  /api/runs/{id}/report      final report JSON
  GET  /api/runs                  list of runs
  GET  /api/library               example recordings found under data/demo and data/real (optional)
  GET  /files/{path}              evidence images / annotated clips / uploaded media
  POST /api/runs/{id}/chat        follow-up question over the finished run (semantic cache)
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..config import load_profile, resolve_path
from ..memory.semantic_cache import SemanticCache
from ..runner import stream_investigation
from ..schema import UserQuestion
from ..video import media_kind

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UI_DIR = PROJECT_ROOT / "ui"

LIBRARY_DIRS = ["data/demo", "data/real"]
MEDIA_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".jpg", ".jpeg", ".png"}

class RunHandle:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self.report: dict[str, Any] | None = None
        self.pending: UserQuestion | None = None
        self.answer: str | None = None
        self.answer_event = threading.Event()
        self.status = "queued"

    def push(self, event: dict[str, Any]) -> None:
        self.events.append(event)

class AppState:
    def __init__(self, profile: str) -> None:
        self.profile = profile
        self.config = load_profile(profile)
        self.upload_dir = resolve_path(self.config, "runs/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, RunHandle] = {}
        self.media: dict[str, Path] = {}
        self.semantic_cache = SemanticCache()
        self.lock = threading.Lock()
        self.scene_selftest: dict[str, Any] | None = None
        self.warm: dict[str, Any] = {"state": "cold"}

    def start_warm_up(self) -> None:
        """Load every perception model in the background right after start, so the first investigation of the
        demo is as fast as the tenth (models stay resident — see perception/registry.py)."""
        if self.config.get("detector", {}).get("backend", "fixture") == "fixture":
            self.warm = {"state": "n/a (fixture profile)"}
            return

        def job() -> None:
            from ..perception.registry import warm_up

            self.warm = {"state": "loading"}
            try:
                report = warm_up(self.config, PROJECT_ROOT)
                self.warm = {"state": "resident" if report.get("ok") else "partial", **report}
            except Exception as error:
                self.warm = {"state": "failed", "error": f"{type(error).__name__}: {error}"}

        threading.Thread(target=job, daemon=True, name="angel-warmup").start()

STATE: AppState | None = None

def _state() -> AppState:
    assert STATE is not None
    return STATE

async def index(request: Request):
    return FileResponse(UI_DIR / "index.html")

async def health(request: Request):
    from ..agent.llm import create_llm

    st = _state()
    cfg = st.config
    llm = create_llm(cfg.get("llm", {}))
    return JSONResponse({
        "ok": True, "profile": cfg.get("project", {}).get("mode"), "llm": llm.health(),
        "models": {
            "detector": cfg.get("detector", {}).get("backend"), "pose": cfg.get("pose", {}).get("backend"),
            "audio": cfg.get("audio", {}).get("backend"), "asr_model": cfg.get("audio", {}).get("asr_model"),
            "scene": cfg.get("scene", {}).get("backend"), "llm": cfg.get("llm", {}).get("model"),
            "planner": cfg.get("agent", {}).get("planner"),
        },
        "warm": st.warm,
        "local_only": cfg.get("edge_cloud", {}).get("mode", "local_only") == "local_only",
        "simulate_actions": cfg.get("policy", {}).get("simulate_all_external_actions", True),
        "scene_selftest": _scene_selftest(st),
    })

def _scene_selftest(st: "AppState") -> dict[str, Any]:
    """Load the scene model once and run every prompt set on a synthetic image, so a broken install is visible
    in the header instead of silently producing 'unknown' scenes."""
    if st.scene_selftest is not None:
        return st.scene_selftest
    cfg = st.config.get("scene", {})
    if cfg.get("backend", "fixture") in {"fixture", "none"}:
        st.scene_selftest = {"ok": True, "backend": cfg.get("backend", "fixture")}
        return st.scene_selftest
    try:
        from ..perception.scene import create_scene_analyzer

        analyzer = create_scene_analyzer(cfg, PROJECT_ROOT, None, None)
        st.scene_selftest = analyzer.selftest() if analyzer is not None else {"ok": False, "error": "scene analyzer could not be created (see runs/desktop.log)"}
    except Exception as error:
        st.scene_selftest = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return st.scene_selftest

async def library(request: Request):
    """Example recordings: real public clips fetched by scripts/fetch_real_demo_clips.py (with attribution).
    Purely optional: the app is built for any uploaded video/audio/image."""
    st = _state()
    items = []
    for rel in LIBRARY_DIRS:
        root = resolve_path(st.config, rel)
        if not root.exists():
            continue
        manifest = {}
        mpath = root / "library.json"
        if mpath.exists():
            try:
                manifest = {m["file"]: m for m in json.loads(mpath.read_text(encoding="utf-8")).get("items", [])}
            except Exception:
                manifest = {}
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in MEDIA_SUFFIXES or p.name.endswith(".raw.mp4") or "annotated" in p.name:
                continue
            key = str(p.relative_to(root)).replace("\\", "/")
            meta = manifest.get(key, {})
            items.append({"path": _rel(p), "name": meta.get("title") or p.stem.replace("_", " "),
                          "group": meta.get("group") or rel.split("/")[-1], "source": meta.get("source"),
                          "license": meta.get("license"), "expected": meta.get("expected"), "kind": media_kind(p),
                          "url": f"/files/{_rel(p)}"})
    return JSONResponse({"items": items})

async def upload(request: Request):
    st = _state()
    form = await request.form()
    file = form.get("file")
    if file is None:
        return JSONResponse({"error": "no file"}, status_code=400)
    suffix = Path(file.filename or "upload.bin").suffix.lower()
    try:
        media_kind("x" + suffix)
    except ValueError:
        return JSONResponse({"error": f"unsupported type {suffix}"}, status_code=400)
    media_id = uuid.uuid4().hex[:10]
    dest = st.upload_dir / f"{media_id}{suffix}"
    max_bytes = float(st.config.get("security", {}).get("max_upload_mb", 500)) * 1e6
    written = 0
    with open(dest, "wb") as handle:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                handle.close()
                dest.unlink(missing_ok=True)
                return JSONResponse({"error": "file too large"}, status_code=413)
            handle.write(chunk)
    dest = _normalise_browser_recording(dest)
    st.media[media_id] = dest
    return JSONResponse({"media_id": media_id, "filename": file.filename, "url": f"/files/runs/uploads/{dest.name}",
                         "kind": media_kind(dest)})

def _normalise_browser_recording(path: Path) -> Path:
    """Browser MediaRecorder output (webm, or mp4 from the live panel) has no duration/index and a variable frame
    rate, which breaks seeking in the frame reader. Re-encode to a plain 25 fps H.264 mp4 (or 16 kHz wav when
    there is no picture). Any failure keeps the original file, so a normal upload is never rejected here."""
    is_live = path.name.startswith("live_") or path.suffix.lower() == ".webm"
    if not is_live or shutil.which("ffmpeg") is None:
        return path
    has_video = False
    if shutil.which("ffprobe"):
        try:
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type",
                                    "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60, check=False)
            has_video = "video" in probe.stdout
        except Exception:
            has_video = path.suffix.lower() != ".wav"
    else:
        has_video = "mic" not in path.name
    out = path.with_name(path.stem + "_norm" + (".mp4" if has_video else ".wav"))
    if has_video:
        cmd = ["ffmpeg", "-y", "-v", "error", "-fflags", "+genpts", "-i", str(path), "-r", "25",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-movflags", "+faststart", str(out)]
    else:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    except Exception:
        return path
    if result.returncode != 0 or not out.exists() or out.stat().st_size < 1000:
        out.unlink(missing_ok=True)
        return path
    path.unlink(missing_ok=True)
    return out

async def start_run(request: Request):
    st = _state()
    body = await request.json()
    media_path: Path | None = None
    if body.get("media_id") in st.media:
        media_path = st.media[body["media_id"]]
    elif body.get("media_path"):
        candidate = resolve_path(st.config, str(body["media_path"]))
        try:
            candidate.resolve().relative_to(PROJECT_ROOT.resolve())
            media_path = candidate
        except ValueError:
            media_path = None
    if media_path is None or not media_path.exists():
        return JSONResponse({"error": "media not found"}, status_code=404)
    try:
        media_kind(media_path)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    objective = str(body.get("question") or "Investigate this recording for safety incidents and decide what to do.")
    hint = (body.get("context") or body.get("hint") or "").strip() or None
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    handle = RunHandle(run_id)
    st.runs[run_id] = handle
    profile: str | dict[str, Any] = body.get("profile") or st.profile
    depth = str(body.get("depth") or "").lower()
    if depth in {"fast", "deep"} and isinstance(profile, str):

        cfg = copy.deepcopy(load_profile(profile))
        cfg.setdefault("agent", {})["planner"] = "hybrid" if depth == "fast" else "llm"
        profile = cfg
    timeout_s = float(st.config.get("agent", {}).get("ask_timeout_s", 20))

    def answer_provider(q: UserQuestion) -> str | None:
        handle.pending = q
        handle.answer_event.clear()
        handle.push({"type": "await_answer", "question": q.to_dict()})
        got = handle.answer_event.wait(timeout=max(q.timeout_s, timeout_s))
        handle.pending = None
        return handle.answer if got else None

    def worker() -> None:
        handle.status = "running"
        try:
            for event in stream_investigation(media_path, objective, profile, hint, run_id, answer_provider):
                if event.get("type") == "final":
                    handle.report = event["report"]
                    handle.push({"type": "final", "report": _slim(event["report"])})
                else:
                    handle.push(event)
            handle.status = "done"
        except Exception as error:
            handle.push({"type": "error", "detail": f"{type(error).__name__}: {error}"})
            handle.status = "failed"

    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse({"run_id": run_id, "media_url": f"/files/{_rel(media_path)}", "kind": media_kind(media_path)})

def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")

def _slim(report: dict[str, Any]) -> dict[str, Any]:
    slim = dict(report)
    slim["graph"] = {"node_count": report["graph"]["node_count"], "edge_count": report["graph"]["edge_count"],
                     "edges": report["graph"]["edges"][:120], "nodes": report["graph"]["nodes"][:120]}
    for ev in slim.get("evidence", {}).values():
        if ev.get("path"):
            ev["url"] = "/files/" + _rel(Path(ev["path"]))
    slim["clips"] = [{**c, "url": "/files/" + _rel(Path(c["path"]))} for c in report.get("clips", [])]
    return slim

async def run_events(request: Request):
    st = _state()
    handle = st.runs.get(request.path_params["run_id"])
    if handle is None:
        return JSONResponse({"error": "unknown run"}, status_code=404)

    async def gen():
        import anyio

        idx = 0
        last_beat = time.time()
        while True:
            while idx < len(handle.events):
                yield f"data: {json.dumps(_with_urls(handle.events[idx]), default=str)}\n\n"
                idx += 1
            if handle.status in {"done", "failed"} and idx >= len(handle.events):
                yield "event: end\ndata: {}\n\n"
                break
            if time.time() - last_beat > 10:
                yield ": keepalive\n\n"
                last_beat = time.time()
            await anyio.sleep(0.05)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

def _with_urls(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("type") == "events":
        out = dict(event)
        ev = {}
        for k, v in event["evidence"].items():
            v = dict(v)
            if v.get("path"):
                v["url"] = "/files/" + _rel(Path(v["path"]))
            ev[k] = v
        out["evidence"] = ev
        out["clips"] = [{**c, "url": "/files/" + _rel(Path(c["path"]))} for c in event.get("clips", [])]
        return out
    if event.get("type") == "observation" and event.get("images"):
        out = dict(event)
        out["image_urls"] = ["/files/" + _rel(Path(p)) for p in event["images"]]
        return out
    return event

async def answer(request: Request):
    st = _state()
    handle = st.runs.get(request.path_params["run_id"])
    if handle is None:
        return JSONResponse({"error": "unknown run"}, status_code=404)
    body = await request.json()
    handle.answer = str(body.get("answer", "")).strip() or None
    handle.answer_event.set()
    return JSONResponse({"ok": True, "answer": handle.answer})

async def report(request: Request):
    st = _state()
    handle = st.runs.get(request.path_params["run_id"])
    if handle is None or handle.report is None:
        return JSONResponse({"error": "report not ready"}, status_code=404)
    return JSONResponse(_slim(handle.report))

async def list_runs(request: Request):
    st = _state()
    return JSONResponse({"runs": [{"run_id": r.run_id, "status": r.status, "final": (r.report or {}).get("final_answer"),
                                   "verdict": (r.report or {}).get("verdict")} for r in st.runs.values()]})

async def chat(request: Request):
    """Follow-up question over a finished run: exact/semantic cache first, then the local model with a compact
    context (final report, verdict, events, decisions, what the person said, the steps' thoughts)."""
    from ..agent.llm import create_llm

    st = _state()
    handle = st.runs.get(request.path_params["run_id"])
    if handle is None or handle.report is None:
        return JSONResponse({"error": "The investigation has not finished yet."}, status_code=404)
    body = await request.json()
    question = str(body.get("question", "")).strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)
    cached = st.semantic_cache.get(handle.run_id, question)
    if cached:
        return JSONResponse({"answer": cached["answer"], "cache": cached["kind"], "similarity": cached.get("similarity")})
    rep = handle.report
    context = {
        "verdict": rep.get("verdict"), "final_answer": rep["final_answer"], "uncertainty": rep.get("final_uncertainty"),
        "scene": (rep.get("scene") or {}).get("description"),
        "hypothesis": rep["hypothesis"],
        "events": [{k: e.get(k) for k in ("event_id", "kind", "subject", "obj", "start_s", "end_s", "confidence", "severity", "summary")} for e in rep["events"][:8]],
        "decisions": [{k: d.get(k) for k in ("action", "executed", "result", "rationale")} for d in rep["decisions"]],
        "question_asked": rep["questions"], "transcript": rep["audio"]["segments"][:10],
        "steps": [{"step": s["step"], "thought": (s.get("thought") or "")[:300], "action": s["action"], "observation": (s.get("observation") or "")[:300]} for s in rep["steps"]],
    }
    llm = create_llm(st.config.get("llm", {}))
    t0 = time.time()
    if not llm.is_real:
        answer_text = f"[fixture] Based on the investigation: {rep['final_answer']}"
    else:
        import anyio

        def ask() -> str:
            resp = llm.chat([{"role": "system", "content": "You are the PersonalAngel operator assistant. Answer the operator's question about a FINISHED safety investigation using ONLY the JSON context. Be direct and specific (who, when in seconds, which evidence, which decision and why). If the operator disagrees with the decision, explain the policy reasoning honestly and say what evidence would change it. If the context does not contain the answer, say so. Reply in PLAIN ENGLISH PROSE, 3-6 sentences — no JSON, no code fences, no bullet lists."},
                             {"role": "user", "content": f"CONTEXT:\n{json.dumps(context, default=str)[:9000]}\n\nQUESTION: {question}"}], max_tokens=400)
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text[4:] if text.lower().startswith("json") else text
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    text = str(data.get("answer") or data.get("response") or data.get("text") or text)
            except Exception:
                pass
            return text.strip()

        try:
            answer_text = await anyio.to_thread.run_sync(ask)
        except Exception as error:
            return JSONResponse({"error": f"The local model could not answer ({type(error).__name__}: {str(error)[:160]}). Is the model server running?"}, status_code=502)
    st.semantic_cache.put(handle.run_id, question, answer_text)
    return JSONResponse({"answer": answer_text, "cache": "miss", "seconds": round(time.time() - t0, 1)})

def create_app(profile: str = "fixture"):
    global STATE
    STATE = AppState(profile)
    STATE.start_warm_up()
    runs_dir = resolve_path(STATE.config, STATE.config.get("output", {}).get("directory", "runs"))
    data_dir = PROJECT_ROOT / "data"
    runs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    routes = [
        Route("/", index), Route("/api/health", health), Route("/api/library", library),
        Route("/api/upload", upload, methods=["POST"]), Route("/api/runs", start_run, methods=["POST"]),
        Route("/api/runs", list_runs, methods=["GET"]), Route("/api/runs/{run_id}/events", run_events),
        Route("/api/runs/{run_id}/answer", answer, methods=["POST"]), Route("/api/runs/{run_id}/report", report),
        Route("/api/runs/{run_id}/chat", chat, methods=["POST"]),
        Mount("/files/runs", StaticFiles(directory=str(runs_dir), check_dir=False), name="files_runs"),
        Mount("/files/data", StaticFiles(directory=str(data_dir), check_dir=False), name="files_data"),
        Mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui"),
    ]
    try:
        from fastapi import FastAPI

        app = FastAPI(title="PersonalAngel", version="1.0.0", routes=routes)
    except Exception:
        app = Starlette(routes=routes)
    return app

def serve(profile: str = "fixture", host: str = "127.0.0.1", port: int = 8600) -> None:
    import uvicorn

    app = create_app(profile)
    print(f"PersonalAngel UI -> http://{host}:{port}  (profile: {profile})")
    uvicorn.run(app, host=host, port=port, log_level="warning")

def cleanup_uploads(older_than_s: float = 6 * 3600) -> None:
    st = _state()
    now = time.time()
    for p in st.upload_dir.glob("*"):
        if now - p.stat().st_mtime > older_than_s:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
