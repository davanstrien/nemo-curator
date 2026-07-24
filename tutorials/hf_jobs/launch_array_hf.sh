#!/bin/bash
# =============================================================================
# NeMo Curator — Hugging Face Jobs array launch script
#
# The HF Jobs analog of tutorials/slurm/submit_array.sh: launches one
# independent Job per shard and exports the NEMO_CURATOR_SLURM_ARRAY_*
# variables that drive Curator's deterministic array sharding. The pipeline
# (tutorials/slurm/array_pipeline.py) runs unmodified inside each Job; see
# run_shard_hf.py and the README for details.
#
# Usage:
#   bash tutorials/hf_jobs/launch_array_hf.sh <total_shards> [shard_index ...]
#
#   bash tutorials/hf_jobs/launch_array_hf.sh 20        # launch shards 0-19
#   bash tutorials/hf_jobs/launch_array_hf.sh 20 3 7    # retry only shards 3 and 7
#
# Configuration (environment variables, all optional except BUCKET):
#   BUCKET               hf://buckets/<your-username>/curator-tutorial-staging
#   FLAVOR               Job flavor (default: cpu-basic)
#   JOB_TIMEOUT          per-Job timeout (default: 30m)
#   SHARE_DIR            shared prefix on the mounted bucket (default:
#                        /mnt/curator-array-demo; the bucket is mounted at /mnt)
#   INPUT_FILE_TYPE      jsonl (default) or parquet
#   OUTPUT_FILE_TYPE     jsonl (default) or parquet
#   FILES_PER_PARTITION  files grouped into each source task (default: 1)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if (($# < 1)); then
    echo "Usage: launch_array_hf.sh <total_shards> [shard_index ...]" >&2
    exit 2
fi

TOTAL_SHARDS="$1"
shift
SHARDS=("$@")
if ((${#SHARDS[@]} == 0)); then
    SHARDS=($(seq 0 $((TOTAL_SHARDS - 1))))
fi

BUCKET="${BUCKET:-hf://buckets/<your-username>/curator-tutorial-staging}"
FLAVOR="${FLAVOR:-cpu-basic}"
JOB_TIMEOUT="${JOB_TIMEOUT:-30m}"
SHARE_DIR="${SHARE_DIR:-/mnt/curator-array-demo}"
INPUT_FILE_TYPE="${INPUT_FILE_TYPE:-jsonl}"
OUTPUT_FILE_TYPE="${OUTPUT_FILE_TYPE:-jsonl}"
FILES_PER_PARTITION="${FILES_PER_PARTITION:-1}"

for k in "${SHARDS[@]}"; do
    echo "Launching shard ${k} of ${TOTAL_SHARDS} (flavor: ${FLAVOR})"
    hf jobs uv run --detach --flavor "${FLAVOR}" --timeout "${JOB_TIMEOUT}" \
        -s HF_TOKEN \
        -e NEMO_CURATOR_SLURM_ARRAY_ENABLED=1 \
        -e NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX="${k}" \
        -e NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS="${TOTAL_SHARDS}" \
        -e INPUT_DIR="${SHARE_DIR}/input" \
        -e OUTPUT_DIR="${SHARE_DIR}/out" \
        -e CHECKPOINT_PATH="${SHARE_DIR}/ckpt" \
        -e INPUT_FILE_TYPE="${INPUT_FILE_TYPE}" \
        -e OUTPUT_FILE_TYPE="${OUTPUT_FILE_TYPE}" \
        -e FILES_PER_PARTITION="${FILES_PER_PARTITION}" \
        -v "${BUCKET}:/mnt" \
        --name "curator-array-shard-${k}" \
        "${SCRIPT_DIR}/run_shard_hf.py"
    # Space out launches: attaching the bucket mount on freshly provisioned
    # nodes is more reliable with a short gap between Jobs.
    sleep 10
done
