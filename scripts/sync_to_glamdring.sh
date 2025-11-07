#!/bin/bash
set -euo pipefail

# ---- local source ----
SRC_BASE="$HOME/Projects/CMBOlympics"

# ---- glamdring destination ----
DEST_USER="rstiskalek"
DEST_HOST="glamdring.physics.ox.ac.uk"
DEST_PATH="/mnt/extraspace/rstiskalek/CMBOlympics"
SSH_KEY="$HOME/.ssh/glamdring"

usage() {
    echo "Usage: $0 [results|data] [--delete]"
    exit 1
}

ensure_remote_dir() {
    local subdir="$1"
    echo "[INFO] Ensuring remote destination exists: ${DEST_PATH}/${subdir}"
    ssh -i "$SSH_KEY" "$DEST_USER@$DEST_HOST" "mkdir -p '${DEST_PATH}/${subdir}'"
}

sync_dir() {
    local subdir="$1"
    local local_path="${SRC_BASE}/${subdir}"

    if [[ ! -d "$local_path" ]]; then
        echo "[ERROR] Local directory not found: ${local_path}" >&2
        exit 1
    fi

    ensure_remote_dir "$subdir"
    echo "[INFO] Pushing '${subdir}' to glamdring -> ${DEST_PATH}/${subdir}/"
    rsync -avh --progress -e "ssh -i $SSH_KEY" \
        "${rsync_delete_flag[@]}" \
        "${local_path}/" \
        "$DEST_USER@$DEST_HOST:${DEST_PATH}/${subdir}/"
}

# ---- parse arguments ----
if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
fi

DELETE_FLAG=0
TARGET="$1"
if [[ $# -eq 2 ]]; then
    if [[ "$2" == "--delete" ]]; then
        DELETE_FLAG=1
    else
        usage
    fi
fi

rsync_delete_flag=()
if [[ $DELETE_FLAG -eq 1 ]]; then
    rsync_delete_flag=(--delete)
    echo "[INFO] Remote files absent locally will be deleted."
fi

case "$TARGET" in
    results)
        sync_dir "results"
        ;;
    data)
        sync_dir "data"
        ;;
    *)
        usage
        ;;
esac

echo "[INFO] Sync complete."
