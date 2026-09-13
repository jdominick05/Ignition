#!/usr/bin/env bash
# Shared setup for Ignition pipeline and management scripts. Source this, don't run it.
#
# Runs under Git Bash on Windows. NOT WSL: the AMD XDNA1 AIE2 NPU requires Windows PyXRT runtime.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
BLUE=$'\033[34m'; DIM=$'\033[2m'; OFF=$'\033[0m'

step() { printf '\n%s==> %s%s\n' "$BOLD$BLUE" "$*" "$OFF"; }
info() { printf '%s    %s%s\n' "$DIM" "$*" "$OFF"; }
warn() { printf '%s!!  %s%s\n' "$YELLOW" "$*" "$OFF" >&2; }
ok()   { printf '%s OK %s%s\n' "$GREEN" "$*" "$OFF"; }
die()  { printf '\n%sERROR: %s%s\n' "$RED$BOLD" "$*" "$OFF" >&2; exit 1; }

_conda_ready=0
conda_init() {
    [ "$_conda_ready" = 1 ] && return 0
    local base
    base="$(conda info --base 2>/dev/null || true)"
    for c in "$base/etc/profile.d/conda.sh" \
             "$HOME/miniforge3/etc/profile.d/conda.sh" \
             "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "$c" ]; then
            set +u; . "$c"; set -u
            _conda_ready=1
            return 0
        fi
    done
    die "could not find conda.sh. Is miniforge/miniconda installed?"
}

use_env() {
    conda_init
    set +u; conda activate "$1"; set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "$1" ] || die "failed to activate $1"
    info "env: $1  ($(python -c 'import sys;print(sys.version.split()[0])'))"
}

check_npu_contention() {
    local smi="/c/Windows/System32/AMD/xrt-smi.exe"
    if [ -x "$smi" ]; then
        local out
        out="$("$smi" examine -r aie-partitions 2>/dev/null || true)"
        if [ -n "$out" ] && ! echo "$out" | grep -q "No hardware contexts running"; then
            warn "Active hardware contexts detected on NPU! Concurrency will degrade benchmarks."
            info "$out"
        fi
    fi
}

need_file() { [ -f "$1" ] || die "missing $1${2:+ -- $2}"; }
need_dir()  { [ -d "$1" ] || die "missing directory $1${2:+ -- $2}"; }

run_logged() {
    local log="$1"; shift
    mkdir -p "$(dirname "$log")"
    info "log: $log"
    set +e
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    return $rc
}

tmp_dir() {
    local t="${TEMP:-${TMP:-}}"
    if [ -n "$t" ] && command -v cygpath >/dev/null 2>&1; then
        cygpath -u "$t"
    else
        echo "/c/Users/${USERNAME:-${USER:-Default}}/AppData/Local/Temp"
    fi
}

free_gb() { df -k "${1:-/c}" | awk 'NR==2{printf "%d", $4/1048576}'; }

require_disk() {
    local need="$1" what="$2" have
    have="$(free_gb "$(tmp_dir)")"
    info "disk: ${have} GB free, ~${need} GB needed for $what"
    if [ "$have" -lt "$need" ]; then
        die "not enough free disk: ${have} GB free, ~${need} GB needed for $what."
    fi
    [ "$have" -lt $((need * 2)) ] && warn "this will leave about $((have - need)) GB free"
    return 0
}

usage() {
    awk 'NR==1 && /^#!/ {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$1"
}
