#!/usr/bin/env bash
# Push the PRAETOR LLM layer to the team repo as a PR branch.
#
# Run FROM THE PRAETOR REPO ROOT on your machine (Git Bash on Windows):
#     bash scripts/push_llm_to_team.sh
#
# What it does:
#   1. clones the team repo (your git credentials) into a temp dir
#   2. creates branch feat/llm-information-layer from the default branch
#      (or bootstraps 'main' if the repo is empty)
#   3. copies the LLM parts in (never overwrites their README; merges .gitignore)
#   4. commits, pushes, prints the ready-made PR link
#
# Env overrides: TEAM_URL, BRANCH
set -euo pipefail

TEAM_URL="${TEAM_URL:-https://github.com/Saadzwak/Crusoe.git}"
BRANCH="${BRANCH:-feat/llm-information-layer}"
SRC="$(pwd)"

[ -d "$SRC/backend/agent" ] || { echo "!! run me from the PRAETOR repo root"; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
echo "==> cloning $TEAM_URL"
git clone "$TEAM_URL" "$WORK/team"
cd "$WORK/team"

# Default branch (empty repo → bootstrap main)
DEFAULT="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
if [ -z "$(git branch -a | grep -v HEAD || true)" ]; then
  echo "==> team repo is EMPTY — bootstrapping '$BRANCH' directly (no PR base exists)"
  DEFAULT=""
  git checkout -b "$BRANCH" 2>/dev/null || git switch -c "$BRANCH"
else
  DEFAULT="${DEFAULT:-main}"
  echo "==> default branch: $DEFAULT"
  git switch -c "$BRANCH" "origin/$DEFAULT" 2>/dev/null || git switch -c "$BRANCH"
fi

echo "==> grafting LLM parts"
copy() {  # copy preserving relative path; warn instead of clobbering their files
  local rel="$1"
  local rel_dst
  if [ -e "$rel" ] && [ -n "${2:-}" ]; then
    echo "  !! $rel exists in team repo — copying as $2 instead"
    rel_dst="$2"
  else
    rel_dst="${2:-$rel}"
    [ -e "$rel" ] && [ -z "${2:-}" ] && echo "  .. overwriting $rel (ours is newer)"
  fi
  mkdir -p "$(dirname "$rel_dst")"
  cp -r "$SRC/$rel" "$rel_dst"
}

copy backend
find backend -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -f backend/agent/praetor_state.db* 2>/dev/null || true
copy docs/CRUSOE.md
copy docs/DATAFLOW.md
copy pinn/PINN_architecture-project-crusoe.md
copy pinn/data/README.md
copy pinn/data/train_FD001.txt
copy pinn/data/test_FD001.txt
copy pinn/data/RUL_FD001.txt
copy pinn/data/ai4i2020.csv
copy scripts/crusoe_latency_test.py
copy scripts/crusoe_sanity.py
copy .env.example .env.example.praetor   # only renamed if theirs exists
copy README.md docs/LLM_LAYER_README.md  # never clobber their README
copy docs/PR_BODY_LLM_LAYER.md

# .gitignore: merge our lines into theirs (create if absent)
if [ -f .gitignore ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && ! grep -qxF "$line" .gitignore && echo "$line" >> .gitignore
  done < "$SRC/.gitignore"
else
  cp "$SRC/.gitignore" .gitignore
fi

# Identity fallback (normally your global git config applies)
git config user.name  >/dev/null 2>&1 || git config user.name  "Eric Bjarstal"
git config user.email >/dev/null 2>&1 || git config user.email "eric.bjarstal@etu.utc.fr"

git add -A
git commit -m "feat(llm): PRAETOR information layer — Crusoe two-tier agent, debate, jury, tool-calling operator

Ported from the PRAETOR working repo (commits 81e40db + ebc5d42):
- 3-tier pipeline: python thresholds -> DeepSeek V4 Flash -> Nemotron Ultra 550B
- Advocate/Skeptic debate (<=2 rounds, safe-hold escalation)
- Jury LLM-as-judge gate on every advisory (+ no-digits deterministic backstop)
- Tool-calling operator chatbot (LangChain bind_tools): sensor history, machine
  spec, live diagnostic, HMAC custody-chain verify, drift analysis + PINN/camera
  stubs; programmatic Data Provenance on every answer
- Boss-LLM plant brief; shared SQLite store (no agent-to-agent mesh)
- FastAPI + SSE agent console; Michelin Roanne knowledge with citations
- Runs offline (MOCK_LLM=1) and live via CRUSOE_API_KEY; docs/DATAFLOW.md
- 3 test suites green in mock. See docs/LLM_LAYER_README.md"

echo "==> pushing $BRANCH"
git push -u origin "$BRANCH"

REPO_WEB="${TEAM_URL%.git}"
echo ""
echo "================================================================"
if [ -n "$DEFAULT" ]; then
  echo "OPEN THE PR HERE (form pre-filled, paste docs/PR_BODY_LLM_LAYER.md):"
  echo "  $REPO_WEB/compare/$DEFAULT...$BRANCH?expand=1"
else
  echo "Repo was empty: '$BRANCH' pushed as the first branch."
  echo "Set it as default or push it to main — no PR base existed."
fi
echo "================================================================"
