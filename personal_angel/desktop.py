"""Desktop application mode: the whole system (web backend + operator console) runs as ONE local
process and opens in a native window — no browser tab, no URL to type, no internet needed.

    python -m personal_angel desktop --profile pc_cpu

Window strategy (first that works):
  1. pywebview (Edge WebView2 on Windows, WebKitGTK on Linux, WebKit on macOS)
  2. Chrome / Edge / Chromium in --app mode (a chromeless window)
  3. the default browser (last resort)

It also makes sure the local model server is up: Ollama is started automatically on the PC
profiles; on a GPU workstation the vLLM service is checked and the UI says so if it is down.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("personal_angel.desktop")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _free_port(preferred: int) -> int:
    for port in (preferred, preferred + 1, preferred + 2, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred

def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False

def _hidden_popen(cmd: list[str], **kw) -> subprocess.Popen:
    if platform.system() == "Windows":
        kw.setdefault("creationflags", 0x08000000)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)

def ensure_model_server(config: dict) -> dict:
    """Start Ollama if the profile points at it and it is not running. vLLM is only checked."""
    llm = config.get("llm", {})
    base = str(llm.get("base_url", "")).rstrip("/")
    info = {"base_url": base, "ok": False, "started": False, "backend": "fixture"}
    if llm.get("backend") != "openai_compatible" or not base:
        info["ok"] = True
        return info
    if _http_ok(f"{base}/models", 3):
        info.update(ok=True, backend="running")
        return info
    if ":11434" in base:
        exe = shutil.which("ollama")
        if exe is None and platform.system() == "Windows":
            for cand in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
                         Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe"):
                if cand.exists():
                    exe = str(cand)
                    break
        if exe:
            log.info("starting Ollama (%s)", exe)
            try:
                _hidden_popen([exe, "serve"])
                info["started"] = True
                for _ in range(40):
                    if _http_ok(f"{base}/models", 2):
                        info.update(ok=True, backend="ollama")
                        return info
                    time.sleep(0.5)
            except Exception as error:
                log.warning("could not start Ollama: %s", error)
        info["backend"] = "ollama (not running)"
        return info
    info["backend"] = "vllm (not running) — run: bash scripts/launch_vllm.sh"
    return info

def _open_window(url: str, title: str = "PersonalAngel") -> bool:
    """Returns True when a window was opened and has been closed by the user (blocking).
    A chromeless Edge/Chrome window is preferred because it supports the camera and microphone of the Live panel;
    set ANGEL_WINDOW=webview to use the embedded pywebview window instead."""
    def _webview() -> bool:
        try:
            import webview

            webview.create_window(title, url, width=1580, height=1000, min_size=(1100, 720), background_color="#0b1020")
            webview.start()
            return True
        except Exception as error:
            log.info("pywebview not available (%s)", error)
            return False
    if os.environ.get("ANGEL_WINDOW", "").lower() == "webview" and _webview():
        return True
    candidates: list[str] = []
    system = platform.system()
    if system == "Windows":
        pf, pf86, local = os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", ""), os.environ.get("LOCALAPPDATA", "")
        candidates += [str(Path(pf86) / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
                       str(Path(pf) / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
                       str(Path(pf) / "Google" / "Chrome" / "Application" / "chrome.exe"),
                       str(Path(local) / "Google" / "Chrome" / "Application" / "chrome.exe")]
    elif system == "Darwin":
        candidates += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "brave-browser"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    profile_dir = PROJECT_ROOT / "runs" / ".app-window-profile"
    for exe in candidates:
        if not exe or not Path(exe).exists():
            continue
        try:
            proc = subprocess.Popen([exe, f"--app={url}", "--window-size=1580,1000", f"--user-data-dir={profile_dir}",
                                     "--no-first-run", "--disable-features=TranslateUI", "--class=PersonalAngel"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.wait()
            return True
        except Exception as error:
            log.info("%s failed: %s", exe, error)
    if os.environ.get("ANGEL_WINDOW", "").lower() != "webview" and _webview():
        return True
    firefox = shutil.which("firefox")
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) or platform.system() != "Linux"
    if firefox and has_display:
        try:
            subprocess.Popen([firefox, "--kiosk" if os.environ.get("ANGEL_KIOSK") else "--new-window", url],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return False
        except Exception:
            pass
    import webbrowser

    try:
        webbrowser.open(url)
    except Exception:
        pass
    return False

def _show_error_window(title: str, detail: str, log_path: Path) -> None:
    """Never fail silently: show what went wrong in a window (or a message box) and point at the log."""
    import html

    page = (f"<html><body style='font-family:Segoe UI,sans-serif;background:#0b1020;color:#e8ecf8;padding:32px'>"
            f"<h2 style='color:#ff5c6c'>{html.escape(title)}</h2><p>Log: <code>{html.escape(str(log_path))}</code></p>"
            f"<pre style='white-space:pre-wrap;background:#151d38;padding:16px;border-radius:10px;font-size:12px'>{html.escape(detail)}</pre>"
            f"<p>Fix: open PowerShell in the project folder and run <code>.\\scripts\\run_windows.ps1</code> to see the full error, "
            f"or re-run <code>.\\scripts\\setup_windows.ps1</code>.</p></body></html>")
    try:
        import webview

        webview.create_window("PersonalAngel — startup error", html=page, width=980, height=640)
        webview.start()
        return
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, f"{title}\n\n{detail[:1500]}\n\nLog: {log_path}", "PersonalAngel", 0x10)
            return
        except Exception:
            pass
    print(f"{title}\n{detail}", file=sys.stderr)

def launch(profile: str = "pc_cpu", port: int = 8600, window: bool = True) -> int:
    from .config import load_profile
    from .server.app import create_app

    logs = PROJECT_ROOT / "runs"
    logs.mkdir(exist_ok=True)
    log_path = logs / "desktop.log"
    handlers: list[logging.Handler] = [logging.FileHandler(log_path, encoding="utf-8")]
    if sys.stdout is None or sys.stderr is None:

        stream = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    else:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", handlers=handlers)
    config = load_profile(profile)
    model = ensure_model_server(config)
    log.info("model server: %s", model)
    port = _free_port(port)
    app = create_app(profile)

    failure: list[str] = []

    def serve() -> None:
        try:
            import uvicorn

            uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", log_config=None)
        except Exception as error:
            log.exception("backend failed to start")
            failure.append(f"{type(error).__name__}: {error}")

    threading.Thread(target=serve, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    healthy = False
    for _ in range(150):
        if failure:
            break
        if _http_ok(f"{url}api/health", 2):
            healthy = True
            break
        time.sleep(0.2)
    log.info("backend %s at %s (profile %s); model server: %s", "ready" if healthy else "NOT READY", url, profile, model["backend"])
    print(f"PersonalAngel desktop -> {url} (profile {profile}); model server: {model['backend']}")
    if not healthy:
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        except Exception:
            pass
        _show_error_window("PersonalAngel could not start its local backend",
                           (failure[0] if failure else "the backend did not answer within 30 s") + "\n\n" + tail, log_path)
        return 1
    if not window:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0
    closed = _open_window(url)
    if not closed:
        print(f"Open {url} in a browser (on a headless machine: ssh -L {port}:127.0.0.1:{port} <user>@<host>). Ctrl+C quits.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0
