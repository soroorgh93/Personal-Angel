#!/usr/bin/env bash
# Start PersonalAngel on a Linux GPU machine against the locally served model (vLLM on :8000 by default).
#   bash scripts/run_linux.sh            # desktop app window on the machine's display
#   WEB=1 bash scripts/run_linux.sh      # browser UI on :8600 (for ssh -L 8600:127.0.0.1:8600 from a laptop)
#   ANGEL_LLM__BASE_URL=http://127.0.0.1:8000/v1 bash scripts/run_linux.sh   # plain vLLM port
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-gpu/bin/activate
PROFILE="${PROFILE:-workstation}"
PORT="${PORT:-8600}"
BASE="${ANGEL_LLM__BASE_URL:-http://127.0.0.1:8000/v1}"
if ! curl -s "$BASE/models" | grep -q '"id"'; then
  echo "(model endpoint $BASE not reachable - start it with: bash scripts/launch_vllm.sh; the app still runs with the deterministic planner)"
fi
if [[ "${WEB:-0}" == "1" ]]; then
  exec python -m personal_angel serve --profile "$PROFILE" --host 0.0.0.0 --port "$PORT"
fi
exec python -m personal_angel desktop --profile "$PROFILE" --port "$PORT"
