#!/bin/bash
set -euo pipefail

# ---- local destination ----
DEST_BASE="$HOME/Projects/CMBOlympics"

# ---- glamdring source ----
SRC_USER="rstiskalek"
SRC_HOST="glamdring.physics.ox.ac.uk"
SRC_PATH="/mnt/extraspace/rstiskalek/CMBOlympics"
SSH_KEY="$HOME/.ssh/glamdring"

usage() {
    echo "Usage: $0 [results|data|notebooks|scripts|cmbo|git|gitignore|readme|license|setup|requirements|all]"
    exit 1
}

# ---- parse argument ----
if [[ $# -ne 1 ]]; then
    usage
fi

echo "[INFO] Ensuring local destination exists: ${DEST_BASE}"
mkdir -p "$DEST_BASE"

case "$1" in
    results)
        echo "[INFO] Pulling 'results' from glamdring -> ${DEST_BASE}"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/results/" \
          "$DEST_BASE/results/"
        ;;
    data)
        echo "[INFO] Pulling 'data' from glamdring -> ${DEST_BASE}/data/"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/data/" \
          "$DEST_BASE/data/"
        ;;
    notebooks)
        echo "[INFO] Pulling 'notebooks' from glamdring -> ${DEST_BASE}/notebooks/"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/notebooks/" \
          "$DEST_BASE/notebooks/"
        ;;
    scripts)
        echo "[INFO] Pulling 'scripts' from glamdring -> ${DEST_BASE}/scripts/"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/scripts/" \
          "$DEST_BASE/scripts/"
        ;;
    cmbo)
        echo "[INFO] Pulling 'cmbo' from glamdring -> ${DEST_BASE}/cmbo/"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/cmbo/" \
          "$DEST_BASE/cmbo/"
        ;;
    git)
        echo "[INFO] Pulling '.git' from glamdring -> ${DEST_BASE}/.git/"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/.git/" \
          "$DEST_BASE/.git/"
        ;;
    gitignore)
        echo "[INFO] Pulling '.gitignore' from glamdring -> ${DEST_BASE}/.gitignore"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/.gitignore" \
          "$DEST_BASE/.gitignore"
        ;;
    readme)
        echo "[INFO] Pulling 'README.md' from glamdring -> ${DEST_BASE}/README.md"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/README.md" \
          "$DEST_BASE/README.md"
        ;;
    license)
        echo "[INFO] Pulling 'LICENSE' from glamdring -> ${DEST_BASE}/LICENSE"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/LICENSE" \
          "$DEST_BASE/LICENSE"
        ;;
    setup)
        echo "[INFO] Pulling 'setup.py' from glamdring -> ${DEST_BASE}/setup.py"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/setup.py" \
          "$DEST_BASE/setup.py"
        ;;
    requirements)
        echo "[INFO] Pulling 'requirements.txt' from glamdring -> ${DEST_BASE}/requirements.txt"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/requirements.txt" \
          "$DEST_BASE/requirements.txt"
        ;;
    all)
        $0 results
        $0 data
        $0 notebooks
        $0 scripts
        $0 cmbo
        $0 git
        $0 gitignore
        $0 readme
        $0 license
        $0 setup
        $0 requirements
        ;;
    *)
        usage
        ;;
esac

echo "[INFO] Sync complete."
