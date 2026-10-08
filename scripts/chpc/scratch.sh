# shellcheck shell=bash
# Node-local scratch staging for CSDV Slurm jobs.
#
# CHPC asks jobs to do their I/O on /scratch/local and copy data in and out.
# Usage inside an sbatch script:
#
#   source "$CSDV_REPO/scripts/chpc/scratch.sh"
#   scratch_init
#   stage_in "$CSDV_DATA_ROOT/naip/SITE/2018/x.tif" "$NODE_SCRATCH/naip"
#   register_stage_out "$NODE_SCRATCH/out" "$CSDV_DATA_ROOT/out"
#   csdv ...
#
# On exit, successful or not, the EXIT trap copies registered outputs back and
# deletes the scratch directory. Outputs are copied to their destination only
# when the job succeeded, because downstream steps treat an existing file as
# done. After a failure they go to $CSDV_STORE/failed/<job id>/ for inspection.

_CSDV_STAGE_OUT=()

scratch_init() {
    : "${SLURM_JOB_ID:?scratch_init must run inside a Slurm job}"
    NODE_SCRATCH="${CSDV_SCRATCH_ROOT:-/scratch/local}/$USER/$SLURM_JOB_ID"
    mkdir -p "$NODE_SCRATCH"
    export NODE_SCRATCH
    # Keep temp files off the shared /tmp as well.
    export TMPDIR="$NODE_SCRATCH/tmp"
    mkdir -p "$TMPDIR"
    trap _scratch_exit EXIT
    trap 'exit 143' TERM
    echo "node scratch: $NODE_SCRATCH" \
        "($(df -h --output=avail "$NODE_SCRATCH" | tail -1 | tr -d ' ') free)"
}

# stage_in SRC DEST_DIR: copy a file or a directory's contents into DEST_DIR.
stage_in() {
    local src="$1" dest="$2"
    mkdir -p "$dest"
    if [[ -d "$src" ]]; then
        rsync -a "$src"/ "$dest"/
    elif [[ -f "$src" ]]; then
        rsync -a "$src" "$dest"/
    else
        echo "ERROR: stage_in source not found: $src" >&2
        return 1
    fi
    echo "staged in: $src -> $dest"
}

# register_stage_out SRC_DIR DEST_DIR: copy SRC_DIR's contents to DEST_DIR at exit.
register_stage_out() {
    _CSDV_STAGE_OUT+=("$1|$2")
}

_scratch_exit() {
    local rc=$?
    local pair src dest
    set +e
    for pair in ${_CSDV_STAGE_OUT[@]+"${_CSDV_STAGE_OUT[@]}"}; do
        src="${pair%%|*}"
        dest="${pair#*|}"
        [[ -d "$src" && -n "$(ls -A "$src" 2>/dev/null)" ]] || continue
        if ((rc != 0)); then
            dest="${CSDV_STORE:-$HOME}/failed/$SLURM_JOB_ID/$(basename "$src")"
        fi
        mkdir -p "$dest"
        if rsync -a "$src"/ "$dest"/; then
            echo "staged out: $src -> $dest"
        else
            echo "ERROR: stage-out failed: $src -> $dest" >&2
            ((rc == 0)) && rc=1
        fi
    done
    rm -rf "$NODE_SCRATCH"
    echo "exit code $rc at $(date -Is)"
    exit "$rc"
}
