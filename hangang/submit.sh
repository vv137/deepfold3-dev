#!/usr/bin/env bash
# Submit a DeepFold3 training run on Hangang as a chain of 24 h jobs.
#
# Usage (from /scratch — Slurm only accepts jobs submitted there):
#   cd /scratch/$USER/deepfold3/deepfold3-dev
#   bash hangang/submit.sh my-run                    # 64 nodes (256 GPUs), 5 × 24 h links
#   NODES=16 LINKS=5 bash hangang/submit.sh my-run   # smaller allocation
#   EXTRA_ARGS="--wandb deepfold3-stage0 --override max_steps=50000" bash hangang/submit.sh my-run
#   CONFIG_SRC=configs/df3-stage0-r12.yaml bash hangang/submit.sh my-r12
#
# Each link starts when the previous one ends for any reason (afterany) and
# resumes from the newest checkpoint in $RUNS/<run>. Partition normal allows
# 5 jobs per user, pending included, so LINKS <= 5; re-run this script to extend a chain
# (it waits on the last queued link of the same name).

set -euo pipefail
cd "$(dirname "$0")/.."
source hangang/env.sh

RUN_NAME="${1:?usage: submit.sh <run-name>}"
NODES="${NODES:-64}"  # 64 × 4 GH200 = 256 GPUs
LINKS="${LINKS:-5}"
PARTITION="${PARTITION:-normal}"
if [[ -z "${DF3_SLURM_FIELD}" || -z "${DF3_SLURM_APPL}" ]]; then
    echo "DF3_SLURM_FIELD / DF3_SLURM_APPL are empty; Slurm requires --comment (values from 'showappl')." >&2
    exit 1
fi
case "$(pwd -P)" in /scratch/*) ;; *) echo "Submit from /scratch (cwd is $(pwd -P))." >&2; exit 1 ;; esac

mkdir -p "${RUNS}/${RUN_NAME}/slurm"
# Continue an existing chain if one is queued for this run.
prev=$(squeue -u "${USER}" -h -n "df3-${RUN_NAME}" -o '%i' | sort -n | tail -n1 || true)
for ((i = 1; i <= LINKS; i++)); do
    dep=()
    [[ -n "${prev}" ]] && dep=(--dependency="afterany:${prev}")
    prev=$(sbatch --parsable "${dep[@]}" \
        --job-name="df3-${RUN_NAME}" \
        --nodes="${NODES}" \
        --partition="${PARTITION}" \
        --comment="$(df3_slurm_comment)" \
        --output="${RUNS}/${RUN_NAME}/slurm/%j.out" \
        --export=ALL,RUN_NAME="${RUN_NAME}",CONFIG_SRC="${CONFIG_SRC:-configs/initial.yaml}",EXTRA_ARGS="${EXTRA_ARGS:-}" \
        hangang/train.sbatch)
    echo "link ${i}/${LINKS}: job ${prev}"
done
echo "logs: ${RUNS}/${RUN_NAME}/slurm/   cancel: scancel -n df3-${RUN_NAME}"
