#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:7860}"
TASK="${1:-easy}"
SEED="${2:-42}"

echo "Starting human session"
echo "BASE_URL=$BASE_URL"
echo "TASK=$TASK SEED=$SEED"

echo "\n[1/4] Reset episode"
curl -s -X POST "$BASE_URL/reset?task=$TASK&seed=$SEED" | tee /tmp/human_reset.json

echo "\n[2/4] Show state"
curl -s "$BASE_URL/state" | tee /tmp/human_state.json

echo "\n[3/4] Example step payloads"
cat <<'EOF'
Inspect:
{"action":{"type":"inspect","artifact":"preprocessing"}}

Compare:
{"action":{"type":"compare","artifact_a":"validation_strategy","artifact_b":"eval_report"}}

Flag:
{"action":{"type":"flag","violation_type":"V1","evidence_artifact":"preprocessing","evidence_quote":"scaler.fit_transform(X_all)","severity":"high"}}

Submit:
{"action":{"type":"submit","verdict":"reject","summary":"Detected leakage with grounded evidence"}}
EOF

echo "\n[4/4] Execute step manually (edit payload.json first)"
cat > /tmp/payload.json <<'JSON'
{"action":{"type":"inspect","artifact":"preprocessing"}}
JSON

curl -s -X POST "$BASE_URL/step" -H "Content-Type: application/json" -d @/tmp/payload.json | tee /tmp/human_step.json

echo "Session helper complete. Log outputs are in /tmp/human_*.json"
