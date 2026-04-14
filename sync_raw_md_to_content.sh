#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/.nanobot/workspace}"
SOURCE_DIR="${SOURCE_DIR:-$WORKSPACE_DIR/data/raw}"
TARGET_DIR="${TARGET_DIR:-$WORKSPACE_DIR/data/content}"

DRY_RUN=0
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage: ./sync_raw_md_to_content.sh [options]

Recursively copy all .md files from ~/.nanobot/workspace/data/raw
to ~/.nanobot/workspace/data/content, preserving directory structure.

Options:
  --dry-run      Show planned copies without changing files
  --overwrite    Replace existing target files
  -h, --help     Show this help message

Environment overrides:
  WORKSPACE_DIR
  SOURCE_DIR
  TARGET_DIR
EOF
}

log() {
  printf '[sync-md] %s\n' "$*"
}

fail() {
  printf '[sync-md] ERROR: %s\n' "$*" >&2
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
    --overwrite)
      OVERWRITE=1
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

[ -d "$SOURCE_DIR" ] || fail "Source directory does not exist: $SOURCE_DIR"
run mkdir -p "$TARGET_DIR"

log "Source: $SOURCE_DIR"
log "Target: $TARGET_DIR"

copied=0
skipped=0
found=0

while IFS= read -r -d '' src; do
  found=$((found + 1))
  rel="${src#"$SOURCE_DIR"/}"
  dest="$TARGET_DIR/$rel"
  dest_dir="$(dirname "$dest")"

  run mkdir -p "$dest_dir"

  if [ -e "$dest" ] && [ "$OVERWRITE" -ne 1 ]; then
    log "Skip existing: $rel"
    skipped=$((skipped + 1))
    continue
  fi

  run cp -f "$src" "$dest"
  log "Copied: $rel"
  copied=$((copied + 1))
done < <(find "$SOURCE_DIR" -type f -name '*.md' -print0 | sort -z)

if [ "$found" -eq 0 ]; then
  log "No .md files found under $SOURCE_DIR"
  exit 0
fi

log "Done. copied=$copied skipped=$skipped found=$found"
