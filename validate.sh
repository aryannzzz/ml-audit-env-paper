#!/bin/bash
# ML Experiment Integrity Auditor — Full Validation Script
set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ML Audit Env — Pre-Submission Validation               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

echo "=== [1/7] Running unit tests ==="
pytest tests/ -v --tb=short
echo "✅ Tests passed"
echo ""

echo "=== [2/7] Pool integrity ==="
python3 -c "
from environment.generator import POOL, get_pool_stats
import json
stats = get_pool_stats()
print(json.dumps(stats, indent=2))
all_violations = set()
for tier in ['easy', 'medium', 'hard']:
    for exp in POOL[tier]:
        for v in exp['ground_truth']['violations']:
            all_violations.add(v)
print(f'Violation types covered: {sorted(all_violations)}')
assert all_violations == {'V1','V2','V3','V4','V5','V6','V7','V8'}, f'Missing: {set([\"V1\",\"V2\",\"V3\",\"V4\",\"V5\",\"V6\",\"V7\",\"V8\"]) - all_violations}'
all_archetypes = set()
for tier in ['easy', 'medium', 'hard', 'clean']:
    for exp in POOL[tier]:
        all_archetypes.add(exp.get('archetype', 'unknown'))
print(f'Archetypes covered: {sorted(all_archetypes)}')
for exp in POOL['clean']:
    assert len(exp['ground_truth']['violations']) == 0
print(f'All {len(POOL[\"clean\"])} clean experiments verified violation-free')
print('✅ Pool integrity verified')
"
echo ""

echo "=== [3/7] Import check ==="
python3 -c "
from environment.models import Action, Observation, EpisodeState
from environment.env import MLAuditEnv
from environment.grader import grade, grade_single_flag, evidence_found
from environment.generator import POOL, generate, get_pool_stats
print('✅ All imports successful')
"
echo ""

echo "=== [4/7] Environment lifecycle ==="
python3 -c "
from environment.env import MLAuditEnv
from environment.models import Action
for task in ['easy', 'medium', 'hard']:
    env = MLAuditEnv(task=task)
    obs = env.reset(seed=42)
    assert obs.experiment_id and obs.goal and obs.available_artifacts
    action = Action(type='inspect', artifact=obs.available_artifacts[0])
    obs2, reward, done, info = env.step(action)
    assert obs2.steps_used == 1
    action = Action(type='submit', verdict='pass', summary='test')
    obs3, reward, done, info = env.step(action)
    assert done and 'score' in info and 0.0 <= info['score'] <= 1.0
    env.close()
    print(f'  {task}: score={info[\"score\"]:.4f} ✓')
print('✅ Lifecycle verified')
"
echo ""

echo "=== [5/7] Reproducibility (seed=42) ==="
python3 -c "
from environment.env import MLAuditEnv
for task in ['easy', 'medium', 'hard']:
    e1 = MLAuditEnv(task=task); o1 = e1.reset(seed=42)
    e2 = MLAuditEnv(task=task); o2 = e2.reset(seed=42)
    assert o1.experiment_id == o2.experiment_id, f'{task}: not deterministic'
    print(f'  {task}: {o1.experiment_id} ✓')
print('✅ Deterministic seeding verified')
"
echo ""

echo "=== [6/7] API endpoint tests ==="
kill $(lsof -t -i:7860) 2>/dev/null || true
sleep 1
uvicorn app:app --port 7860 --log-level warning &
SERVER_PID=$!
sleep 3

for EP in "GET /health" "GET /tasks" "GET /scoring"; do
    METHOD=$(echo $EP | cut -d' ' -f1)
    PATH_EP=$(echo $EP | cut -d' ' -f2)
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7860$PATH_EP)
    [ "$STATUS" = "200" ] && echo "  $EP: ✅" || echo "  $EP: ❌ $STATUS"
done

STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:7860/reset?task=easy&seed=42")
[ "$STATUS" = "200" ] && echo "  POST /reset: ✅" || echo "  POST /reset: ❌ $STATUS"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" -d '{"action":{"type":"inspect","artifact":"preprocessing"}}')
[ "$STATUS" = "200" ] && echo "  POST /step: ✅" || echo "  POST /step: ❌ $STATUS"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7860/state)
[ "$STATUS" = "200" ] && echo "  GET /state: ✅" || echo "  GET /state: ❌ $STATUS"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:7860/close)
[ "$STATUS" = "200" ] && echo "  POST /close: ✅" || echo "  POST /close: ❌ $STATUS"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" -d '{"task":"easy","flags":[],"verdict":"pass","steps_used":3}')
[ "$STATUS" = "200" ] && echo "  POST /grader: ✅" || echo "  POST /grader: ❌ $STATUS"

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
echo ""

echo "=== [7/7] Docker build ==="
if command -v docker &> /dev/null; then
    docker build -t ml-audit-env . && echo "✅ Docker build successful" || echo "❌ Docker build failed"
else
    echo "⚠️  Docker not available — skip"
fi
echo ""

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Validation complete!                                    ║"
echo "╚══════════════════════════════════════════════════════════╝"