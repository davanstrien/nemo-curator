# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "nemo-curator @ git+https://github.com/NVIDIA-NeMo/Curator.git@f0910665adc57a10295a5ed9bf8ad829e3bcde9a",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///

"""Hugging Face Jobs driver for ``tutorials/slurm/array_pipeline.py``.

This file plays the role that ``submit_array.sh`` plays on Slurm: all
scheduler-side plumbing lives here, and the pipeline itself runs unmodified.
``hf jobs uv run`` uploads only this single script into the Job container, so
the driver fetches ``array_pipeline.py`` from this repository at a pinned
commit and verifies its SHA-256 before executing it — the pipeline that runs
is byte-identical to the one in ``tutorials/slurm``.

Three properties of the Jobs container environment are absorbed here so the
pipeline file stays clean (object-store bucket mounts differ from POSIX
scratch filesystems in a few ways):

1. The container starts with its working directory on the FUSE bucket mount.
   Ray's runtime-env packaging hashes the working directory, which is
   unboundedly slow over a FUSE object-store mount, so the driver switches to
   a local temporary directory before starting Ray.
2. Ray autodetects the cgroup CPU limit (1 CPU on the ``cpu-basic`` flavor),
   which is below the streaming executor's resource floor. The driver passes
   an explicit logical CPU count — a safe overcommit for the I/O-bound demo
   stages. Larger flavors autodetect enough CPUs and do not need this.
3. Bucket mounts do not support POSIX rename, which Curator's atomic
   manifest writes (write + rename) rely on. The driver checkpoints to local
   disk and then publishes the completion-manifest JSON files to the shared
   mount with plain writes, so ``retry_array.py`` works unchanged.

Refer to the README for more details.
"""

from __future__ import annotations

import hashlib
import os
import runpy
import sys
import tempfile
import urllib.request
from pathlib import Path

# Pin the pipeline to a specific commit of this repository. Bump both values
# together whenever tutorials/slurm/array_pipeline.py changes.
UPSTREAM_COMMIT = "f0910665adc57a10295a5ed9bf8ad829e3bcde9a"
ARRAY_PIPELINE_SHA256 = "80870f20a0a617a8e0ead74be0fca0eff7deaf5478b197a41c0a64834c16af12"
ARRAY_PIPELINE_URL = (
    f"https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/{UPSTREAM_COMMIT}/tutorials/slurm/array_pipeline.py"
)

# Shared paths on the bucket mount; the launcher forwards these. Defaults
# match launch_array_hf.sh.
INPUT_DIR = os.environ.get("INPUT_DIR", "/mnt/curator-array-demo/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/curator-array-demo/out")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/mnt/curator-array-demo/ckpt")
INPUT_FILE_TYPE = os.environ.get("INPUT_FILE_TYPE", "jsonl")
OUTPUT_FILE_TYPE = os.environ.get("OUTPUT_FILE_TYPE", "jsonl")
FILES_PER_PARTITION = os.environ.get("FILES_PER_PARTITION", "1")

COMPLETION_SUBDIR = Path(".nemo_curator_metadata") / ".slurm_array_completion"


def fetch_pinned_pipeline(dest_dir: Path) -> Path:
    """Download array_pipeline.py at the pinned commit and verify its SHA-256."""
    dest = dest_dir / "array_pipeline.py"
    with urllib.request.urlopen(ARRAY_PIPELINE_URL, timeout=60) as response:  # noqa: S310
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ARRAY_PIPELINE_SHA256:
        msg = (
            f"array_pipeline.py sha256 {digest} does not match pinned "
            f"{ARRAY_PIPELINE_SHA256}; update UPSTREAM_COMMIT and "
            "ARRAY_PIPELINE_SHA256 together."
        )
        raise RuntimeError(msg)
    dest.write_bytes(data)
    print(
        f"[hf_jobs] fetched array_pipeline.py @ {UPSTREAM_COMMIT[:9]} (sha256 verified)",
        flush=True,
    )
    return dest


def apply_ray_cpu_override() -> None:
    """Give RayClient an explicit logical CPU count (note 2).

    Ray autodetects the container cgroup limit — 1 CPU on ``cpu-basic`` —
    which is below the streaming executor's resource floor. Overcommitting is
    safe for the I/O-bound demo stages. Set ``HF_JOBS_RAY_NUM_CPUS=0`` to
    disable the override on flavors that autodetect enough CPUs.
    """
    num_cpus = int(os.environ.get("HF_JOBS_RAY_NUM_CPUS", "8"))
    if num_cpus <= 0:
        return

    import nemo_curator.core.client as curator_client

    base_client = curator_client.RayClient

    class JobsRayClient(base_client):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("num_cpus", num_cpus)
            kwargs.setdefault("include_dashboard", False)
            super().__init__(*args, **kwargs)

    curator_client.RayClient = JobsRayClient


def publish_completion_manifests(local_checkpoint: Path) -> None:
    """Copy completion manifests from local disk to the shared mount (note 3).

    Curator writes ``run.json`` and the per-shard completion manifests
    atomically (write + rename), which bucket mounts do not support. Plain
    writes are supported, so the manifests are published to the shared
    checkpoint path after the pipeline finishes. ``retry_array.py`` then reads
    them exactly as it would on a shared cluster filesystem.
    """
    src = local_checkpoint / COMPLETION_SUBDIR
    dst = Path(CHECKPOINT_PATH) / COMPLETION_SUBDIR
    dst.mkdir(parents=True, exist_ok=True)
    for manifest in sorted(src.glob("*.json")):
        (dst / manifest.name).write_bytes(manifest.read_bytes())
        print(f"[hf_jobs] published manifest {manifest.name}", flush=True)


def main() -> None:
    shard = os.environ.get("NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX", "?")
    total = os.environ.get("NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS", "?")
    print(
        f"[hf_jobs] shard {shard}/{total} input={INPUT_DIR} output={OUTPUT_DIR} checkpoint={CHECKPOINT_PATH}",
        flush=True,
    )

    # Note 1: work from local disk, not from the bucket mount.
    workdir = Path(tempfile.mkdtemp(prefix="curator-hf-jobs-", dir="/tmp"))
    os.chdir(workdir)
    local_checkpoint = workdir / "ckpt"

    apply_ray_cpu_override()

    pipeline_file = fetch_pinned_pipeline(workdir)

    # Checkpoint metadata goes to local disk first (note 3); the pipeline CLI
    # already accepts the path, so no pipeline changes are needed.
    sys.argv = [
        "array_pipeline.py",
        "--input-dir",
        INPUT_DIR,
        "--input-file-type",
        INPUT_FILE_TYPE,
        "--output-dir",
        OUTPUT_DIR,
        "--output-file-type",
        OUTPUT_FILE_TYPE,
        "--files-per-partition",
        FILES_PER_PARTITION,
        "--checkpoint-path",
        str(local_checkpoint),
    ]
    runpy.run_path(str(pipeline_file), run_name="__main__")

    publish_completion_manifests(local_checkpoint)
    print(f"[hf_jobs] shard {shard}/{total} DONE", flush=True)


if __name__ == "__main__":
    main()
