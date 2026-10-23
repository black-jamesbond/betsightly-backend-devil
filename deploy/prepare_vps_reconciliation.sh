#!/usr/bin/env bash
# Stage a three-way merge of VPS-only work and the tested GitHub release.
# NO live checkout/branch/file changes; NO restart, remote push or deployment.
# Usage:
# bash /root/betsightly-prepare-merge.sh \
#   /root/.openclaw/workspace/betsightly-backend-devil \
#   /root/betsightly-vps-snapshots/20261023T110219Z
set -euo pipefail
umask 077

LIVE="$1"
BACKUP="$2"
RELEASE="c44e00939976b54d7b9d3e457d85d296c0fe41c8"
STAMP="$(basename "$BACKUP")"
SOURCE="$BACKUP/live-source-worktree"
MERGE="$BACKUP/reconciled-release-worktree"
SOURCE_BRANCH="vps/source-$STAMP"
MERGE_BRANCH="vps/reconcile-$STAMP"
REPORT="$BACKUP/merge-report"

abort() { echo "ABORT: $*" >&2; exit 2; }
[[ -d "$LIVE/.git" && -f "$LIVE/dashboard/app.py" ]] ||
  abort "Not the expected independent BetSightly checkout"
[[ -f "$BACKUP/live-HEAD.txt" && -f "$BACKUP/dirty-tracked.patch" &&
   -f "$BACKUP/untracked-paths.null" ]] ||
  abort "Missing preflight snapshot"
[[ ! -e "$SOURCE" && ! -e "$MERGE" ]] ||
  abort "Merge worktrees already exist; do not overwrite"
[[ "$(git -C "$LIVE" rev-parse HEAD)" == "$(cat "$BACKUP/live-HEAD.txt")" ]] ||
  abort "Live HEAD changed since preflight"
git -C "$LIVE" cat-file -e "$RELEASE^{commit}" ||
  abort "Tested release SHA unavailable"

# Abort if the live tree changed since its backup was made. Do not mix two
# different historical states or trust a stale safety snapshot.
cmp -s "$BACKUP/dirty-tracked.patch" <(git -C "$LIVE" diff --binary HEAD) ||
  abort "Tracked changes differ from preflight; take a NEW snapshot"
cmp -s "$BACKUP/untracked-paths.null" \
  <(git -C "$LIVE" ls-files --others --exclude-standard -z) ||
  abort "Untracked filenames differ from preflight; take a NEW snapshot"

mkdir -p "$REPORT"

# Stage only source. All existing model, calibration, media and data files
# remain in the private backup, outside the source-code auto-merge.
is_source() {
  local name
  name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$name" in
    models/*|data/*|cache/*|bot_results/*|logs/*|\
    .venv/*|venv/*|node_modules/*|.git/*|\
    *.parquet|*.joblib|*.pkl|*.pickle|*.sqlite|*.sqlite3|\
    *.png|*.jpg|*.jpeg|*.webp|*.gif|*.pdf|*.zip|\
    *.tar|*.db|*.bin|*.onnx|*.h5|*.pt|*.pth|diagram.*|\
    .env|.env.*|*/.env|*/.env.*|\
    *secret*|*credential*|*password*|*private_key*|*.pem|*.key)
      return 1 ;;
  esac
  case "$name" in
    *.py|*.pyi|*.sh|*.html|*.css|*.js|*.jsx|*.ts|*.tsx|\
    *.md|*.yml|*.yaml|*.toml|*.service|*.timer|*.sql|\
    .gitignore|procfile|dockerfile|requirements*.txt)
      return 0 ;;
  esac
  return 1
}

git -C "$LIVE" worktree add --detach "$SOURCE" "$(cat "$BACKUP/live-HEAD.txt")"
git -C "$SOURCE" switch -c "$SOURCE_BRANCH"
: > "$REPORT/source-file-paths.txt"
: > "$REPORT/excluded-file-paths.txt"

copy_source_path() {
  local p="$1"
  [[ -n "$p" ]] || return 0
  if ! is_source "$p" || [[ -L "$LIVE/$p" ]]; then
    printf '%s\n' "$p" >> "$REPORT/excluded-file-paths.txt"
    return 0
  fi
  if [[ -f "$LIVE/$p" ]]; then
    mkdir -p "$SOURCE/$(dirname "$p")"
    cp -p -- "$LIVE/$p" "$SOURCE/$p"
  elif [[ -e "$SOURCE/$p" ]]; then
    rm -f -- "$SOURCE/$p"
  fi
  git -C "$SOURCE" add -A -- "$p"
  printf '%s\n' "$p" >> "$REPORT/source-file-paths.txt"
}

while IFS= read -r -d '' path; do
  copy_source_path "$path"
done < <(git -C "$LIVE" diff --name-only --no-renames -z HEAD)
while IFS= read -r -d '' path; do
  copy_source_path "$path"
done < "$BACKUP/untracked-paths.null"

if ! git -C "$SOURCE" diff --cached --quiet; then
  git -C "$SOURCE" -c user.name="VPS Local Source Snapshot" \
    -c user.email="local-snapshot@localhost" -c core.hooksPath=/dev/null \
    commit -m "snapshot: VPS-only source at $STAMP (local only)"
fi
SOURCE_SHA="$(git -C "$SOURCE" rev-parse HEAD)"

git -C "$LIVE" worktree add --detach "$MERGE" "$RELEASE"
git -C "$MERGE" switch -c "$MERGE_BRANCH"

# Expected: merge conflicts in dashboard/app.py etc. All are isolated here.
set +e
git -C "$MERGE" -c core.hooksPath=/dev/null merge \
  --no-commit --no-ff "$SOURCE_BRANCH" \
  > "$REPORT/git-merge-output.txt" 2>&1
MERGE_RC=$?
set -e
git -C "$MERGE" diff --name-only --diff-filter=U \
  > "$REPORT/conflicted-paths.txt" || true
git -C "$MERGE" status --short > "$REPORT/candidate-status.txt" || true
git -C "$MERGE" diff --stat "$RELEASE" \
  > "$REPORT/candidate-diff-stat.txt" || true

echo "=== VPS THREE-WAY SOURCE RECONCILIATION ==="
echo "Live HEAD: $(cat "$BACKUP/live-HEAD.txt")"
echo "Source snapshot SHA: $SOURCE_SHA"
echo "Snapshot branch: $SOURCE_BRANCH (LOCAL ONLY)"
echo "Tested release: $RELEASE"
echo "Merge candidate: $MERGE"
echo "Private report: $REPORT"
echo "Source paths copied: $(wc -l < "$REPORT/source-file-paths.txt")"
echo "Excluded data/secret paths: $(wc -l < "$REPORT/excluded-file-paths.txt")"
echo "Git merge exit: $MERGE_RC"
echo "Conflict count: $(wc -l < "$REPORT/conflicted-paths.txt")"
echo "--- Conflict paths only ---"
cat "$REPORT/conflicted-paths.txt"
echo "Live tree: UNTOUCHED | Gunicorn: UNTOUCHED | GitHub: NO PUSH"
echo "No deployment: merge conflicts and cross-feature tests require review."
