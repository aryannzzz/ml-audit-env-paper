#!/usr/bin/env bash
# run_v2_eval.sh — Run v2-script 10-seed evaluations for GPT-4.1 and o4-mini.
#
# Usage:
#   export OPENAI_API_KEY="sk-proj-..."
#   bash scripts/run_v2_eval.sh
#
# Results are written to:
#   logs/v2_gpt41_<timestamp>.txt
#   logs/v2_o4mini_<timestamp>.txt
# Then parse with scripts/parse_v2_results.py to update all_results.txt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SEEDS="42,43,44,45,46,47,48,49,50,51"
ENV_URL="${ENV_URL:-http://localhost:7860}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it before running this script." >&2
  exit 1
fi

# Check server
if ! curl -sf "${ENV_URL}/health" > /dev/null 2>&1; then
  echo "ERROR: MLAuditBench server not reachable at ${ENV_URL}." >&2
  echo "Start it with: docker run -p 7860:7860 <image>  OR  uvicorn app:app --port 7860" >&2
  exit 1
fi
echo "Server OK: $(curl -s ${ENV_URL}/health | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d[\"environment\"],d[\"version\"])')"

run_model() {
  local model="$1"
  local logfile="$2"
  echo ""
  echo "=== Running ${model} seeds ${SEEDS} ==="
  OPENAI_API_KEY="${OPENAI_API_KEY}" \
  MODEL_NAME="${model}" \
  ENV_URL="${ENV_URL}" \
  TEMPERATURE=0.0 \
  ENFORCE_COMPARE=1 \
  python3 "${REPO_DIR}/inference.py" --seed-list "${SEEDS}" \
    2>&1 | tee "${logfile}"
  echo "Done. Log: ${logfile}"
}

# GPT-4.1 (balanced)
GPT41_LOG="${REPO_DIR}/logs/v2_gpt41_${TIMESTAMP}.txt"
run_model "gpt-4.1" "${GPT41_LOG}"

# o4-mini (reasoning)
O4MINI_LOG="${REPO_DIR}/logs/v2_o4mini_${TIMESTAMP}.txt"
run_model "o4-mini" "${O4MINI_LOG}"

echo ""
echo "All evaluations complete."
echo "Parse results with:"
echo "  python3 scripts/parse_v2_results.py ${GPT41_LOG} ${O4MINI_LOG}"
