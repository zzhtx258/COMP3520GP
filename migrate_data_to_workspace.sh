#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-$PROJECT_DIR/data}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/.nanobot/workspace}"
TARGET_DATA_DIR="${TARGET_DATA_DIR:-$WORKSPACE_DIR/data}"

DRY_RUN=0
FORCE=0
KEEP_SOURCE=0
NO_SYMLINK=0

usage() {
  cat <<'EOF'
Usage: ./migrate_data_to_workspace.sh [options]

Move project data from ~/COMP3520GP/data to ~/.nanobot/workspace/data.

Options:
  --dry-run      Show planned actions without changing anything
  --force        Remove an existing target data symlink before migrating
  --keep-source  Copy data instead of moving it
  --no-symlink   Do not create ~/COMP3520GP/data -> ~/.nanobot/workspace/data
  -h, --help     Show this help message

Environment overrides:
  PROJECT_DIR
  SOURCE_DATA_DIR
  WORKSPACE_DIR
  TARGET_DATA_DIR
EOF
}

log() {
  printf '[migrate-data] %s\n' "$*"
}

fail() {
  printf '[migrate-data] ERROR: %s\n' "$*" >&2
  exit 1
}

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --force)
      FORCE=1
      ;;
    --keep-source)
      KEEP_SOURCE=1
      ;;
    --no-symlink)
      NO_SYMLINK=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
  shift
done

if [ ! -e "$SOURCE_DATA_DIR" ]; then
  fail "Source data directory does not exist: $SOURCE_DATA_DIR"
fi

if [ -L "$TARGET_DATA_DIR" ]; then
  if [ "$FORCE" -eq 1 ]; then
    log "Removing existing target symlink: $TARGET_DATA_DIR"
    run rm "$TARGET_DATA_DIR"
  else
    fail "Target path is a symlink. Re-run with --force if you want to replace it: $TARGET_DATA_DIR"
  fi
fi

if [ -e "$TARGET_DATA_DIR" ] && [ ! -d "$TARGET_DATA_DIR" ]; then
  fail "Target exists but is not a directory: $TARGET_DATA_DIR"
fi

log "Project dir : $PROJECT_DIR"
log "Source data : $SOURCE_DATA_DIR"
log "Workspace   : $WORKSPACE_DIR"
log "Target data : $TARGET_DATA_DIR"

run mkdir -p "$WORKSPACE_DIR"

if [ -d "$TARGET_DATA_DIR" ]; then
  if [ -n "$(find "$TARGET_DATA_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    if [ -L "$SOURCE_DATA_DIR" ] && [ "$(readlink "$SOURCE_DATA_DIR")" = "$TARGET_DATA_DIR" ]; then
      log "Source already points to target; nothing to migrate."
      exit 0
    fi
    fail "Target data directory is not empty: $TARGET_DATA_DIR"
  fi

  if [ "$KEEP_SOURCE" -eq 1 ]; then
    log "Copying source contents into existing empty target directory"
    run cp -a "$SOURCE_DATA_DIR"/. "$TARGET_DATA_DIR"/
  else
    log "Moving source directory into existing empty target directory"
    run rmdir "$TARGET_DATA_DIR"
    run mv "$SOURCE_DATA_DIR" "$TARGET_DATA_DIR"
  fi
else
  if [ "$KEEP_SOURCE" -eq 1 ]; then
    log "Copying source directory to target"
    run cp -a "$SOURCE_DATA_DIR" "$TARGET_DATA_DIR"
  else
    log "Moving source directory to target"
    run mv "$SOURCE_DATA_DIR" "$TARGET_DATA_DIR"
  fi
fi

if [ "$NO_SYMLINK" -eq 0 ]; then
  if [ -e "$SOURCE_DATA_DIR" ] || [ -L "$SOURCE_DATA_DIR" ]; then
    if [ "$KEEP_SOURCE" -eq 1 ]; then
      log "Keeping source directory in place because --keep-source was set; skipping symlink"
    else
      fail "Source path still exists after migration, cannot create symlink safely: $SOURCE_DATA_DIR"
    fi
  else
    run mkdir -p "$PROJECT_DIR"
    log "Creating compatibility symlink: $SOURCE_DATA_DIR -> $TARGET_DATA_DIR"
    run ln -s "$TARGET_DATA_DIR" "$SOURCE_DATA_DIR"
  fi
fi

log "Migration complete."
if [ "$DRY_RUN" -eq 0 ]; then
  log "You can verify with:"
  log "  ls -l \"$PROJECT_DIR\""
  log "  ls \"$TARGET_DATA_DIR\""
fi
