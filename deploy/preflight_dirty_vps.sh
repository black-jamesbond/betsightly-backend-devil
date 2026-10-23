#!/usr/bin/env bash
# Non-destructive preflight for independent BetSightly port 5050.
# No reset, clean, live checkout, restart, or replacement occurs.
# Usage: bash /root/betsightly-vps-preflight.sh /root/.openclaw/workspace/betsightly-backend-devil
set -euo pipefail
umask 077

if [[ $# -gt 0 ]]; then
  LIVE_ROOT="$1"
else
  LIVE_ROOT="/root/.openclaw/workspace/betsightly-backend-devil"
fi
RELEASE_REF="refs/remotes/origin/release/vps-2026-10-23"
EXPECTED_RELEASE="c44e00939976b54d7b9d3e457d85d296c0fe41c8"

if [[ ! -d "$LIVE_ROOT/.git" || ! -f "$LIVE_ROOT/dashboard/app.py" ]]; then
  echo "ABORT: expected standalone BetSightly repository and dashboard" >&2
  exit 2
fi
if ! git -C "$LIVE_ROOT" cat-file -e "$EXPECTED_RELEASE^{commit}"; then
  echo "ABORT: tested release commit unavailable; fetch the release branch first" >&2
  exit 2
fi
if ! git -C "$LIVE_ROOT" merge-base --is-ancestor "$EXPECTED_RELEASE" "$RELEASE_REF"; then
  echo "ABORT: fetched release branch no longer contains the tested commit" >&2
  exit 2
fi

NOW="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="/root/betsightly-vps-snapshots/$NOW"
STAGING="/root/betsightly-release-candidate-$NOW"
mkdir -p "$BACKUP"
echo "Snapshot directory: $BACKUP"

# Preserve the local-only HEAD and BOTH tracked and untracked changes.
git -C "$LIVE_ROOT" rev-parse HEAD > "$BACKUP/live-HEAD.txt"
git -C "$LIVE_ROOT" branch --show-current > "$BACKUP/live-branch.txt"
git -C "$LIVE_ROOT" status --porcelain=v1 --untracked-files=all > "$BACKUP/dirty-paths.txt"
git -C "$LIVE_ROOT" diff --binary HEAD > "$BACKUP/dirty-tracked.patch"
git -C "$LIVE_ROOT" diff --binary --cached HEAD > "$BACKUP/staged.patch"
git -C "$LIVE_ROOT" ls-files --others --exclude-standard -z > "$BACKUP/untracked-paths.null"
tar -C "$LIVE_ROOT" --null -T "$BACKUP/untracked-paths.null" \
  -cf "$BACKUP/untracked-files.tar"
chmod 600 "$BACKUP/untracked-files.tar"
git -C "$LIVE_ROOT" bundle create "$BACKUP/local-git.bundle" --all
tar -C "$LIVE_ROOT" -czf "$BACKUP/live-dashboard.tar.gz" \
  dashboard/app.py dashboard/templates dashboard/static

# Tested code goes into a separate directory. The active tree is untouched.
git -C "$LIVE_ROOT" worktree add --detach "$STAGING" "$EXPECTED_RELEASE"

# Keep private diffs for reconciling the VPS-only user interface.
for FILE in dashboard/app.py dashboard/templates/index.html \
            dashboard/static/theme.css fixtures_livescore.py; do
  if [[ -f "$LIVE_ROOT/$FILE" && -f "$STAGING/$FILE" ]]; then
    NAME="$(printf '%s' "$FILE" | tr / _)"
    diff -u "$LIVE_ROOT/$FILE" "$STAGING/$FILE" > "$BACKUP/$NAME.diff" || true
  fi
done

echo
echo "=== SAFE PREFLIGHT RESULT ==="
echo "Live root: $LIVE_ROOT"
echo "Live HEAD: $(cat "$BACKUP/live-HEAD.txt")"
echo "Live branch: $(cat "$BACKUP/live-branch.txt")"
echo "Dirty entries: $(wc -l < "$BACKUP/dirty-paths.txt")"
echo "Release candidate: $STAGING"
echo "Candidate HEAD: $(git -C "$STAGING" rev-parse HEAD)"
echo "Private backups: $BACKUP"
echo "Live service: NOT RESTARTED"
echo "Live working tree: NOT MODIFIED"
echo "Commit graph live vs tested release (left/right):"
git -C "$LIVE_ROOT" rev-list --left-right --count "HEAD...$EXPECTED_RELEASE" || true
echo
echo "Must reconcile live-only Rollover, Fixtures, Daily Report and"
echo "other local edits before replacing any live files. Do not reset/clean."
