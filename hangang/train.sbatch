#!/bin/bash
# DeepFold3 training on Hangang GH200 nodes (4 GPUs/node, one srun task per GPU).
# Submit through hangang/submit.sh, which sets the run name, node count and
# the 24 h job chain; the trainer auto-resumes from the newest checkpoint in
# the run directory, so every link of the chain runs the same command.
# auto_resume is forced on below: every shipped config inherits
# `auto_resume: false` from initial.yaml, which would make each link restart
# from step 0 (or from --params) instead of continuing the chain.
#
#SBATCH --job-name=df3
#SBATCH --partition=normal
#SBATCH --comment="field=sciai;appl=pytorch-ddp"
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --distribution=block:block
#SBATCH --signal=B:USR1@600
#SBATCH --output=%x-%j.out

set -euo pipefail
: "${RUN_NAME:?submit via hangang/submit.sh}"
source "${SLURM_SUBMIT_DIR}/hangang/env.sh"
cd "${REPO}"

RUN_DIR="${RUNS}/${RUN_NAME}"
CONFIG_SRC="${CONFIG_SRC:-configs/initial.yaml}"
mkdir -p "${RUN_DIR}"
# Freeze the config on the first link of the chain; later links reuse it so a
# repo edit mid-run cannot change a running experiment.
[[ -f "${RUN_DIR}/config.yaml" ]] || df3_make_config "${CONFIG_SRC}" "${RUN_DIR}/config.yaml"

echo "job ${SLURM_JOB_ID}: ${SLURM_JOB_NUM_NODES} nodes x 4 GPUs -> ${RUN_DIR}"
echo "nodes: ${SLURM_JOB_NODELIST}"

export OMP_NUM_THREADS=8
if (( SLURM_JOB_NUM_NODES > 1 )); then
    df3_slingshot_env
fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 10 min before the walltime Slurm sends USR1 to this script; it creates the
# stop file. Rank 0 sees it on its next microstep, the trainer's
# ShutdownCoordinator stops all ranks at that microstep and writes
# checkpoint_<step>.pt, which the next chain link resumes from. Do not send
# SIGTERM to srun here: srun cancels the step and SIGKILLs the ranks ~30 s later.
export DF3_STOP_FILE="${RUN_DIR}/.stop-${SLURM_JOB_ID}"
trap 'echo "USR1: walltime near, asking ranks to checkpoint"; touch "${DF3_STOP_FILE}" || true' USR1

# --gres-flags=allow-task-sharing lets each task open all 4 GPUs of its node
# (NCCL P2P / CUDA IPC between ranks); Slurm accepts it only on srun.
srun --mpi=none --gpu-bind=none --gres-flags=allow-task-sharing --kill-on-bad-exit=1 \
    "${REPO}/hangang/srun_rank.sh" \
    "${REPO}/.venv/bin/python" -m deepfold3.scripts.train \
    --config "${RUN_DIR}/config.yaml" \
    --output "${RUN_DIR}" \
    --ccd "${DATA}/components.cif.gz" \
    --ccd-cache "${DATA}/ccd_cache.json" \
    --mmcif-dir "${DATA}/pdb/divided" \
    --override auto_resume=true \
    ${EXTRA_ARGS:-} &
SRUN_PID=$!
set +e
wait "${SRUN_PID}"
rc=$?
# A trapped signal interrupts `wait`; keep waiting while the ranks checkpoint.
while kill -0 "${SRUN_PID}" 2>/dev/null; do
    wait "${SRUN_PID}"
    rc=$?
done
rm -f "${DF3_STOP_FILE}"
exit "${rc}"
