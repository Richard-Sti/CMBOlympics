#!/bin/bash
set -euo pipefail

memory=7
queue="berg"
env="/mnt/users/rstiskalek/CMBOlympics/venv_cmob/bin/python"
default_nthreads=16

usage() {
    echo "Usage: $0 <on_login:{0|1}> <script.py>"
    exit 1
}

if [[ $# -ne 2 ]]; then
    usage
fi

on_login="$1"
target_script="$2"

if [[ "$on_login" != "0" && "$on_login" != "1" ]]; then
    usage
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_file="${script_dir}/config.toml"

extract_njobs() {
    local cfg="$1"
    [[ -f "$cfg" ]] || return 1
    awk -F'=' '
        /^\s*\[runtime\]\s*$/ {in_runtime=1; next}
        /^\s*\[/ {in_runtime=0}
        in_runtime && $1 ~ /n_jobs/ {
            val=$2
            sub(/#.*/, "", val)
            gsub(/[[:space:]]/, "", val)
            if (val != "") {
                print val
                exit 0
            }
        }
    ' "$cfg"
}

if njobs_value=$(extract_njobs "$config_file"); then
    nthreads="$njobs_value"
    echo "[INFO] Using nthreads=${nthreads} from ${config_file}"
else
    nthreads=$default_nthreads
    echo "[WARN] Falling back to default nthreads=${nthreads}"
fi

python_cmd="$target_script"
if [[ -n "$env" ]]; then
    python_cmd="$env $python_cmd"
fi

if [[ $on_login -eq 1 ]]; then
    echo "$python_cmd"
    eval "$python_cmd"
else
    submit_cmd="addqueue -s -q $queue -n $nthreads -m $memory $python_cmd"
    echo "Submitting:"
    echo "$submit_cmd"
    echo
    eval "$submit_cmd"
fi
