#!/usr/bin/env bash
# Serve the local vision-language model with vLLM (OpenAI-compatible API on http://127.0.0.1:8000/v1).
#   bash scripts/launch_vllm.sh                                   # Qwen/Qwen2.5-VL-7B-Instruct
#   MODEL=Qwen/Qwen3-VL-8B-Instruct bash scripts/launch_vllm.sh
#   GPU_FRACTION=0.35 bash scripts/launch_vllm.sh                 # share of GPU memory vLLM may reserve
#   MODE=docker bash scripts/launch_vllm.sh                       # official vllm/vllm-openai container
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${PORT:-8000}"
MODE="${MODE:-native}"
GPU_FRACTION="${GPU_FRACTION:-0.35}"
if curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q '"id"'; then
  echo "== a model is already served on :${PORT}"; curl -s "http://127.0.0.1:${PORT}/v1/models"; echo; exit 0
fi
if [[ "$MODE" == "docker" ]]; then
  exec docker run --rm --gpus all --ipc=host -p "127.0.0.1:${PORT}:8000" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -e HF_TOKEN="${HF_TOKEN:-}" \
    vllm/vllm-openai:latest "$MODEL" --host 0.0.0.0 --port 8000 --max-model-len 32768 \
    --limit-mm-per-prompt '{"image":12}' --gpu-memory-utilization "$GPU_FRACTION"
fi
command -v vllm >/dev/null || { echo "vllm not installed: pip install vllm  (or MODE=docker)"; exit 1; }
exec vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":12}' --gpu-memory-utilization "$GPU_FRACTION"
