#!/usr/bin/env bash
# PersonalAngel — Linux GPU workstation (ARM64 or x86_64, CUDA 13) application environment.
# The LLM/VLM is served separately by vLLM; this script prepares the CV/ASR app.
#
#   bash scripts/setup_linux.sh            # creates .venv-gpu with CUDA-13 PyTorch + ultralytics + faster-whisper
#   bash scripts/launch_vllm.sh       # serve Qwen2.5-VL-7B-Instruct with vLLM (already cached on the device)
#   ./PersonalAngel                        # desktop app (or WEB=1 bash scripts/run_linux.sh + ssh -L 8600:127.0.0.1:8600)
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== PersonalAngel GPU setup in $(pwd)"
nvidia-smi || { echo "nvidia-smi not found — is an NVIDIA driver installed?"; exit 1; }
python3 -m venv --system-site-packages .venv-gpu   # system site-packages: WebKitGTK bindings for the native window
source .venv-gpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/base.txt
# PyTorch CUDA 13 wheels
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements/vision.txt
# faster-whisper: the official CTranslate2 wheel has no CUDA on aarch64. Try the prebuilt CUDA-13 wheel first,
# fall back to CPU int8 (still fine for a 30 s clip) — .
python -m pip install -r requirements/audio.txt || true
python -m pip install "https://github.com/assix/ctranslate2-aarch64-cuda13-binaries/releases/latest/download/ctranslate2-4.6.0-cp312-cp312-linux_aarch64.whl" 2>/dev/null || echo "(ctranslate2 CUDA wheel not installed; faster-whisper will run on CPU)"
python -m pip install -r requirements/desktop.txt || echo "(pywebview not installed; a chromeless browser window will be used)"
python -m pip install -r requirements/tts.txt || echo "(kokoro not installed; voicemail demo clips will be skipped)"
python -m pip install -e . --no-deps
python scripts/download_models.py --asr large-v3
python scripts/fetch_real_demo_clips.py --out data/demo || echo "(demo clips: some downloads failed; the app works with any uploaded file)"
python -m pytest -q
# TensorRT export of the detectors (optional, about 1 ms per frame on a modern GPU)
python - <<'EOF' || true
from ultralytics import YOLO
for w in ("models/yolo11n.pt", "models/yolo11n-pose.pt", "models/gun-knife-yolo11n.pt"):
    try:
        YOLO(w).export(format="engine", half=True, imgsz=640, device=0)
        print("exported", w)
    except Exception as e:
        print("export skipped", w, e)
EOF
echo "== done. Next: bash scripts/launch_vllm.sh  then  ./PersonalAngel   (or WEB=1 bash scripts/run_linux.sh)"
