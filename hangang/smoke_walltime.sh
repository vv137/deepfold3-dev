#!/usr/bin/env bash
# Smoke test for the walltime checkpoint: the real train.sbatch on 2 nodes.
#
# Usage (from /scratch):
#   cd /scratch/$USER/deepfold3/deepfold3-dev
#   bash hangang/smoke_walltime.sh submit my-smoke      # 2 linked jobs, new run name
#   bash hangang/smoke_walltime.sh check my-smoke       # after both ended
#
# Link 1 gets USR1 about 15 min in (startup takes ~5 min) and must write a
# checkpoint and exit 0. Link 2 must resume from that checkpoint. No regular
# checkpoint falls in this window, so one on disk can only be the shutdown's.
# Needs 2 free slots in the 5-job queue limit.
# CONFIG_SRC and EXTRA_ARGS as in submit.sh; the default uses stage2 from scratch.

set -euo pipefail
cd "$(dirname "$0")/.."
source hangang/env.sh

MODE="${1:?usage: smoke_walltime.sh submit|check <run-name>}"
RUN_NAME="${2:?usage: smoke_walltime.sh submit|check <run-name>}"
RUN_DIR="${RUNS}/${RUN_NAME}"
LOGS="${RUN_DIR}/slurm"

submit_link() {  # <--time> <--signal> [sbatch args...]
    local time=$1 sig=$2
    shift 2
    sbatch --parsable "$@" \
        --job-name="df3-${RUN_NAME}" \
        --nodes="${NODES:-2}" \
        --partition="${PARTITION:-normal}" \
        --comment="$(df3_slurm_comment)" \
        --time="${time}" \
        --signal="${sig}" \
        --output="${LOGS}/%j.out" \
        --export=ALL,RUN_NAME="${RUN_NAME}",CONFIG_SRC="${CONFIG_SRC:-configs/stage2.yaml}",EXTRA_ARGS="--override eval_before=false --override eval_interval=1000000 ${EXTRA_ARGS:-}" \
        hangang/train.sbatch
}

case "${MODE}" in
submit)
    case "$(pwd -P)" in /scratch/*) ;; *) echo "Submit from /scratch (cwd is $(pwd -P))." >&2; exit 1 ;; esac
    [[ ! -e "${RUN_DIR}" ]] || { echo "${RUN_DIR} exists; use a new run name." >&2; exit 1; }
    mkdir -p "${LOGS}"
    first=$(submit_link 00:40:00 B:USR1@1500)
    second=$(submit_link 00:20:00 B:USR1@600 --dependency="afterany:${first}")
    echo "link 1: job ${first}, link 2: job ${second}"
    echo "then: bash hangang/smoke_walltime.sh check ${RUN_NAME}"
    ;;
check)
    mapfile -t logs < <(ls "${LOGS}"/*.out | sort -V)
    (( ${#logs[@]} == 2 )) || { echo "FAIL: expected 2 logs in ${LOGS}, found ${#logs[@]}" >&2; exit 1; }
    first=${logs[0]} second=${logs[1]}
    fail=0
    expect() {  # <description> <log> <pattern>
        if grep -aqE "$3" "$2"; then echo "ok    $1"; else echo "FAIL  $1"; fail=1; fi
    }
    reject() {
        if grep -aqE "$3" "$2"; then echo "FAIL  $1"; fail=1; else echo "ok    $1"; fi
    }
    expect "link 1 received USR1" "${first}" 'USR1: walltime near'
    expect "rank 0 saw the stop file" "${first}" 'Stop file .* found'
    expect "trainer shut down" "${first}" 'Shutdown (mid-window )?at step [0-9]+'
    reject "srun did not cancel the step" "${first}" 'forcing job termination|CANCELLED AT'
    step=$(grep -aoE 'Shutdown (mid-window )?at step [0-9]+' "${first}" | tail -n1 | grep -oE '[0-9]+$' || true)
    if [[ -n "${step}" ]]; then
        expect "link 1 saved checkpoint_${step}.pt" "${first}" "Saved checkpoint to .*/checkpoint_${step}\.pt"
        expect "link 2 resumed from checkpoint_${step}.pt" "${second}" "Resumed from .*/checkpoint_${step}\.pt"
        expect "link 2 trained past step ${step}" "${second}" "trainer: step=$((step + 10 - step % 10)) "
    fi
    if compgen -G "${RUN_DIR}/.stop-*" >/dev/null; then echo "FAIL  stop file left behind"; fail=1; else echo "ok    stop file removed"; fi
    sacct -j "$(basename "${first}" .out),$(basename "${second}" .out)" -X -n -o JobID,State,ExitCode,Elapsed
    (( fail == 0 )) && echo "=== walltime smoke passed" || { echo "=== walltime smoke FAILED"; exit 1; }
    ;;
*)
    echo "usage: smoke_walltime.sh submit|check <run-name>" >&2
    exit 1
    ;;
esac
