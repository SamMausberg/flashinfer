#!/usr/bin/env bash
# Open the upstream FlashInfer PR for one draft in drafts/<name>/.
# Usage: ./submit.sh <name>        (gh must be logged in as SamMausberg)
#        ./submit.sh --list        (show drafts and their branches)
set -euo pipefail
cd "$(dirname "$0")"
shopt -s nullglob

if [[ "${1:-}" == "--list" || -z "${1:-}" ]]; then
  for d in drafts/*/; do
    n=$(basename "$d")
    printf '%-32s %-45s %s\n' "$n" "$(cat "$d/branch")" "$(cat "$d/title")"
  done
  exit 0
fi

d="drafts/$1"
[[ -d "$d" ]] || { echo "no draft named $1 (try --list)"; exit 1; }
branch=$(cat "$d/branch")

# The branch must exist on the fork before a PR can be opened from it.
gh api "repos/SamMausberg/flashinfer/branches/$branch" --silent \
  || { echo "branch $branch not found on SamMausberg/flashinfer"; exit 1; }

gh pr create --repo flashinfer-ai/flashinfer --base main \
  --head "SamMausberg:$branch" \
  --title "$(cat "$d/title")" \
  --body-file "$d/body.md"
