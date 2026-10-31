#!/usr/bin/env bash
# Copy the packed staging tree (hangang/pack_all.sh) to Hangang's /scratch
# through the data-mover node, then verify it there.
#
# Hangang logins need an OTP + password per SSH connection, so this opens ONE
# master connection (ControlMaster) and multiplexes every rsync stream over it:
# you authenticate once per `connect`. The tree is a few thousand multi-GB
# shards, so N parallel streams of whole files is the efficient shape
# (no per-file round trips as with the unpacked data).
#
# Usage (on the host that holds $DEST):
#   export HANGANG_USER=<account>                       # KISTI account id
#   bash hangang/transfer.sh connect                    # OTP + password once; persists $PERSIST
#   bash hangang/transfer.sh push                       # parallel resumable rsync of the whole tree
#   ONLY="distillation/patchr_training" bash hangang/transfer.sh push   # one subtree
#   bash hangang/transfer.sh verify                     # sha256 of every shard, on the data mover
#   bash hangang/transfer.sh status                     # local vs remote sizes
#   bash hangang/transfer.sh close
#
# Re-running `push` resumes: finished files are skipped by size+mtime, partial
# ones continue (--append-verify re-checks the whole file after appending).
#
# The other direction -- training output back to the host that will run
# inference on it:
#   bash hangang/transfer.sh pull                       # $RUNS_REMOTE -> $RUNS_LOCAL
#   ONLY="df3-stage2-001" bash hangang/transfer.sh pull  # one run
#   bash hangang/transfer.sh runs                        # what is there, and what is missing here
#
# `pull` never deletes locally, so a checkpoint kept here after Hangang's
# 15-day purge took it (keepalive.sh) stays. One stream measured 480 MB/s on
# 5 GB checkpoints, which is the line, so there is nothing to parallelise.

set -euo pipefail

: "${HANGANG_USER:?set HANGANG_USER to your KISTI account}"
HOST="${HANGANG_HOST:-hangang-dm.ksc.re.kr}"   # data mover: no CPU-time limit, allows SSH/SFTP
DEST="${DEST:-/store/deepfold3/packed}"         # local staging tree
REMOTE="${REMOTE:-/scratch/${HANGANG_USER}/deepfold3/data}"
STREAMS="${STREAMS:-8}"
PERSIST="${PERSIST:-24h}"
ONLY="${ONLY:-}"                                 # optional subtree(s), relative to $DEST (push) or $RUNS_REMOTE (pull)
RUNS_LOCAL="${RUNS_LOCAL:-/store/deepfold3/runs}"
RUNS_REMOTE="${RUNS_REMOTE:-/scratch/${HANGANG_USER}/deepfold3/runs}"
# WandB's run directory is not under runs/ (it is $REPO/wandb), so this excludes
# nothing today; it is here so that a later run writing its own wandb/ beside
# the checkpoints does not quietly double the pull.
PULL_EXCLUDE="${PULL_EXCLUDE:-wandb .wandb _logs *.tmp}"
CM="${HOME}/.ssh/cm-hangang-${HANGANG_USER}"
SSH=(ssh -o ControlPath="${CM}" -o ServerAliveInterval=60 -o ServerAliveCountMax=10)
WORK="${DEST}/_logs/transfer"

remote() { "${SSH[@]}" "${HANGANG_USER}@${HOST}" "$@"; }

need_master() {
    if ! "${SSH[@]}" -O check "${HANGANG_USER}@${HOST}" 2>/dev/null; then
        echo "No live master connection; run: bash $0 connect" >&2
        exit 1
    fi
}

# Files to send, relative to $DEST: everything except local logs and build state.
list_files() {
    local roots=(".")
    [[ -n "${ONLY}" ]] && read -r -a roots <<<"${ONLY}"
    (cd "${DEST}" && find "${roots[@]}" \( -name _logs -o -name _build \) -prune -o -type f -printf '%s\t%p\n' |
        sed 's#\t\./#\t#')
}

case "${1:-}" in
connect)
    mkdir -p "$(dirname "${CM}")"
    echo "Opening master connection to ${HANGANG_USER}@${HOST} (OTP, then password)..."
    ssh -o ControlMaster=yes -o ControlPath="${CM}" -o ControlPersist="${PERSIST}" \
        -o ServerAliveInterval=60 -fN "${HANGANG_USER}@${HOST}"
    remote "command -v rsync >/dev/null || { echo 'rsync missing on ${HOST}' >&2; exit 1; }; mkdir -p '${REMOTE}'; lfs quota -h /scratch 2>/dev/null | head -5 || true"
    echo "Connected; master persists for ${PERSIST}."
    ;;

push)
    need_master
    mkdir -p "${WORK}"
    list_files | sort -t$'\t' -k1,1nr >"${WORK}/files.tsv"
    n=$(wc -l <"${WORK}/files.tsv")
    total=$(awk -F'\t' '{s+=$1} END {printf "%.2f", s/2^40}' "${WORK}/files.tsv")
    echo "Pushing ${n} files (${total} TiB) in ${STREAMS} streams -> ${HOST}:${REMOTE}"
    # Greedy byte balancing: largest file to the currently lightest stream.
    rm -f "${WORK}"/stream.*
    awk -F'\t' -v S="${STREAMS}" -v W="${WORK}" '
        BEGIN { for (i = 0; i < S; i++) load[i] = 0 }
        { m = 0; for (i = 1; i < S; i++) if (load[i] < load[m]) m = i
          load[m] += $1; print $2 > (W "/stream." m) }' "${WORK}/files.tsv"
    # Create the directory skeleton first so streams never race on mkdir.
    (cd "${DEST}" && cut -f2 "${WORK}/files.tsv" | xargs -r -n1 dirname | sort -u) >"${WORK}/dirs.txt"
    remote "cd '${REMOTE}' && xargs -r mkdir -p" <"${WORK}/dirs.txt"
    pids=()
    for f in "${WORK}"/stream.*; do
        rsync -a --partial --append-verify --files-from="${f}" -e "${SSH[*]}" \
            --info=stats1 "${DEST}/" "${HANGANG_USER}@${HOST}:${REMOTE}/" \
            >"${f}.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "${p}" || fail=1; done
    if ((fail)); then
        echo "Some streams failed (see ${WORK}/stream.*.log); re-run push to resume." >&2
        exit 1
    fi
    echo "Push complete. Next: bash $0 verify"
    ;;

verify)
    need_master
    # Shards: SHA256SUMS written by the packer. Everything else (index files,
    # meta.json, manifests, ids.txt, CCD): a checksum list made here.
    (cd "${DEST}" && find . \( -name _logs -o -name _build -o -path '*/_pack/shards' \) -prune -o -type f -print0 |
        xargs -0 -r sha256sum) >"${DEST}/_logs/plain.sha256"
    remote "cat > '${REMOTE}/.plain.sha256'" <"${DEST}/_logs/plain.sha256"
    remote "cd '${REMOTE}' && sha256sum --quiet -c .plain.sha256 && echo 'plain files OK'"
    remote "cd '${REMOTE}' && find . -path '*/_pack/SHA256SUMS' -printf '%h\n'" | while read -r p; do
        echo "--- ${p}"
        # One sha256sum per shard, run in parallel on the data mover.
        remote "cd '${REMOTE}/${p}' && sed 's#  #\t#' SHA256SUMS | \
            xargs -P 16 -L 1 sh -c 'echo \"\$0  \$1\" | sha256sum --quiet -c -' && echo OK" </dev/null
    done
    ;;

status)
    need_master
    list_files | awk -F'\t' '{print $2"\t"$1}' | sort >"${WORK}/local.tsv"
    remote "cd '${REMOTE}' && find . -type f ! -name '.plain.sha256' -printf '%P\t%s\n'" | sort >"${WORK}/remote.tsv"
    join -t$'\t' -a1 -e MISSING -o 0,1.2,2.2 "${WORK}/local.tsv" "${WORK}/remote.tsv" |
        awk -F'\t' '$2 != $3 {bad++; print} END {printf "%d file(s) missing or incomplete\n", bad+0}'
    ;;

pull)
    need_master
    ex=()
    for p in ${PULL_EXCLUDE}; do ex+=(--exclude "${p}"); done
    srcs=()
    if [[ -n "${ONLY}" ]]; then
        read -r -a want <<<"${ONLY}"
        for w in "${want[@]}"; do srcs+=("${HANGANG_USER}@${HOST}:${RUNS_REMOTE}/${w}"); done
    else
        srcs+=("${HANGANG_USER}@${HOST}:${RUNS_REMOTE}/")
    fi
    mkdir -p "${RUNS_LOCAL}"
    echo "Pulling ${ONLY:-everything} from ${HOST}:${RUNS_REMOTE} -> ${RUNS_LOCAL}"
    echo "  excluding: ${PULL_EXCLUDE}"
    # A checkpoint being written right now has the wrong size, so it is skipped
    # and picked up by the next pull rather than landing here truncated.
    rsync -a --partial --append-verify --info=stats1,progress2 \
        "${ex[@]}" -e "${SSH[*]}" "${srcs[@]}" "${RUNS_LOCAL}/"
    ;;

runs)
    need_master
    echo "--- ${HOST}:${RUNS_REMOTE}"
    remote "cd '${RUNS_REMOTE}' && du -sh */ 2>/dev/null"
    echo "--- missing or short in ${RUNS_LOCAL}"
    remote "cd '${RUNS_REMOTE}' && find . -type f -printf '%P\t%s\n'" | sort >"/tmp/hangang-runs-remote.$$"
    (cd "${RUNS_LOCAL}" 2>/dev/null && find . -type f -printf '%P\t%s\n' | sort) >"/tmp/hangang-runs-local.$$"
    join -t$'\t' -a1 -e MISSING -o 0,1.2,2.2 \
        "/tmp/hangang-runs-remote.$$" "/tmp/hangang-runs-local.$$" |
        awk -F'\t' '$2 != $3 {n++; printf "%-56s remote %14s  local %14s\n", $1, $2, $3}
                    END {printf "%d file(s) to pull\n", n+0}'
    rm -f "/tmp/hangang-runs-remote.$$" "/tmp/hangang-runs-local.$$"
    ;;

close)
    "${SSH[@]}" -O exit "${HANGANG_USER}@${HOST}" || true
    ;;

*)
    sed -n '2,31p' "$0"
    exit 1
    ;;
esac
