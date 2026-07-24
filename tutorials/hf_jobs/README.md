# Running NeMo Curator on Hugging Face Jobs

This tutorial runs the same array-sharded workflow as [`tutorials/slurm`](../slurm) on
[Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) — a managed
container service — with **no cluster required**. The pipeline and the retry planner are the
unmodified files from the Slurm tutorial; only the submit layer changes.

## Contents

| File | Purpose |
|------|---------|
| `run_shard_hf.py` | Jobs-side driver — the analog of `submit_array.sh`. Fetches [`tutorials/slurm/array_pipeline.py`](../slurm/array_pipeline.py) at a pinned commit, verifies its SHA-256, and runs it unmodified inside the Job container |
| `launch_array_hf.sh` | Launches K independent Jobs, one per shard |

This directory adds no pipeline code. The pipeline that runs is
`tutorials/slurm/array_pipeline.py`, byte-identical (SHA-256-verified at run time), and retry
planning uses `tutorials/slurm/retry_array.py` directly from your checkout.

---

## The key concept: only the submit layer changes

Curator's array sharding is driven entirely by environment variables — each shard builds the
full deterministic source-task list and keeps a task iff
`sha256(task_id) % TOTAL_SHARDS == SHARD_INDEX` — so the contract is scheduler-agnostic. On
Slurm, `submit_array.sh` computes the shard index from `SLURM_ARRAY_TASK_ID` and exports the
`NEMO_CURATOR_SLURM_ARRAY_*` variables. On HF Jobs, `launch_array_hf.sh` exports the same
variables per Job:

```
launch_array_hf.sh 3
    │
    ├─ Job 0: NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX=0 ─┐
    ├─ Job 1: NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX=1 ─┼─ shared bucket mounted
    └─ Job 2: NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX=2 ─┘  at /mnt in every Job
                                                          ├── input/    (read)
                                                          ├── out/      (write)
                                                          └── ckpt/     (completion manifests)
```

The shared filesystem is a [Hugging Face storage
bucket](https://huggingface.co/docs/hub/storage-buckets) FUSE-mounted at `/mnt` in every Job.
Jobs never communicate directly, exactly as in the Slurm array workflow: each shard writes its
outputs and its completion manifest to shared storage, and `retry_array.py` discovers
incomplete shards from the manifests alone.

---

## Prerequisites

- A Hugging Face account with access to [Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs).
- The `hf` CLI, authenticated:

```bash
pip install -U huggingface_hub
hf auth login
```

No local NeMo Curator installation is needed to launch shards — each Job resolves its own
environment from the inline metadata in `run_shard_hf.py`. Retry planning (step 4) runs
`tutorials/slurm/retry_array.py` locally, so for that step install Curator into your checkout
as in the [Slurm tutorial](../slurm#slurm-run--bare-metal-shared-virtualenv).

---

## 1. Put input files on a bucket

Create a bucket and upload your JSONL (or Parquet) files under `curator-array-demo/input/`:

```bash
hf buckets create <your-username>/curator-tutorial-staging
hf buckets cp ./my-jsonl-dir \
    hf://buckets/<your-username>/curator-tutorial-staging/curator-array-demo/input --recursive
```

## 2. Launch the array

```bash
export BUCKET=hf://buckets/<your-username>/curator-tutorial-staging

# 3 shards, one cpu-basic Job each
bash tutorials/hf_jobs/launch_array_hf.sh 3
```

Each Job prints its ID on launch. Follow logs with:

```bash
hf jobs logs <JOB_ID>
```

A successful shard ends with:

```text
[hf_jobs] shard <k>/<K> DONE
```

## 3. Check the output

On the bucket you should see one output file per source task and one completion manifest per
finished shard:

```text
curator-array-demo/
├── input/*.jsonl
├── out/<deterministic-name>.jsonl          # idempotent re-runs overwrite in place
└── ckpt/.nemo_curator_metadata/.slurm_array_completion/
    ├── run.json                            # original shard configuration
    └── completed_slurm_array_*.json        # one per completed shard
```

Output names are deterministic, so retried shards overwrite their own files rather than
duplicating them.

## 4. Retry incomplete shards only

The retry workflow is identical to the Slurm tutorial's — same planner, same manifests, same
rules. `retry_array.py` reads the completion-manifest directory, so mirror the bucket's
checkpoint tree locally and run the planner from your checkout:

```bash
hf buckets cp \
    hf://buckets/<your-username>/curator-tutorial-staging/curator-array-demo/ckpt \
    ./ckpt-mirror --recursive

python tutorials/slurm/retry_array.py \
    --checkpoint-path ./ckpt-mirror \
    --format fields
```

An empty output means no shards need retrying. Otherwise, the first field is the missing-shard
expression and the fourth is the original total shard count. Relaunch only the missing shards,
keeping the original total:

```bash
retry_fields="$(
    python tutorials/slurm/retry_array.py \
        --checkpoint-path ./ckpt-mirror \
        --format fields
)"

if [[ -z "${retry_fields}" ]]; then
    echo "No shards need retrying."
else
    read -r RETRY_ARRAY _ _ TOTAL_SHARDS <<< "${retry_fields}"
    # Expand an expression like "0,2" or "5-7" into an explicit shard list
    SHARDS="$(python -c "
import sys
parts = sys.argv[1].split(',')
ids = []
for p in parts:
    lo, _, hi = p.partition('-')
    ids.extend(range(int(lo), int(hi or lo) + 1))
print(' '.join(map(str, ids)))
" "${RETRY_ARRAY}")"
    bash tutorials/hf_jobs/launch_array_hf.sh "${TOTAL_SHARDS}" ${SHARDS}
fi
```

As on Slurm:

- Retries happen at **shard granularity** — the full owning shard runs again.
- Retries must reuse the **same checkpoint path** (here: the same bucket prefix) so completed
  shards are not rerun, and `TOTAL_SHARDS` must remain the original logical shard count so
  deterministic task assignment does not change.
- Run retry discovery only after all launched shards have finished; a still-running shard has
  no completion manifest and therefore appears retryable.
- Shards assigned zero source tasks (possible when `TOTAL_SHARDS` exceeds the number of source
  tasks) exit successfully and still write completion manifests, so uneven fan-outs retry
  correctly.

`SHARD_INDEX_OFFSET` and `--max-array-size` are not needed in this workflow: there is no
maximum array size because every shard is its own Job, so the planner's offset field is
always `0`.

### A retry, end to end

This walkthrough is from a verified run of this tutorial (3 shards over 8 JSONL files on
`cpu-basic`, at commit `f091066`), including recovery from a real transient failure:

1. Shards 0 and 1 were launched first. Shard 1 completed (about 5.5 minutes wall clock, most
   of it environment setup; the pipeline itself took about 40 seconds). Shard 0 failed at
   startup with a transient volume-mount error, before any pipeline work ran — so it left no
   completion manifest.
2. `retry_array.py --format fields` against the mirrored manifests printed `0,2 0 0 3` —
   correctly identifying the failed shard 0 and the never-launched shard 2 from storage alone.
3. The retry wave `launch_array_hf.sh 3 0 2` completed both shards, and a final planner run
   returned an empty plan.

Result: 8/8 input files processed, 8 output files, 3 shard completion manifests plus
`run.json` on the bucket — with the pipeline and planner files byte-identical to
`tutorials/slurm`.

---

## Scaling up: GPU flavors

The launch pattern is unchanged on GPU flavors — set `FLAVOR` (for example `l4x1`) and use a
pipeline with GPU stages. Sharding, completion manifests, and retries behave identically.

<!-- SHOWCASE_NUMBERS: filled from Task C -->
Example: Curator's `FineWebEduClassifier` stage fanned out over real FineWeb rows with this
tutorial's launch pattern — 4 array shards, one shard lost mid-run (simulating a preemption),
recovered with a `retry_array.py` wave. The array contract assigned the 16 input files
across shards automatically (assignment is hash-based, so shard file counts vary: 3/4/5/4),
and the retried shard re-derived exactly its original file assignment.

| Metric | Value |
|--------|-------|
| Rows processed | 400,000 (16 JSONL files, 25k rows each) |
| Array shape | 4 shards + 1 retry-wave shard |
| Throughput (rows/s per GPU) | 292–303 sustained (GPU util 70–83%) |
| Pipeline wall time | ~5.6 min (longest wave-1 shard) + ~6.9 min retry shard |
| Flavor | `l4x1` per shard |
| Total cost | ≈ $0.50 including the interrupted shard and its retry |

The retry planner (`tutorials/slurm/retry_array.py`, unmodified) read the completion
manifests from the shared bucket and emitted `2` — exactly the lost shard — for wave 2;
after the wave it emits an empty plan.
<!-- /SHOWCASE_NUMBERS -->

---

## HF Jobs container notes

Just as `submit_array.sh` absorbs Slurm specifics (task-ID arithmetic, `SHARD_INDEX_OFFSET`,
node-local `/tmp`), `run_shard_hf.py` absorbs the Jobs container specifics so the pipeline
file stays clean. Object-store bucket mounts differ from POSIX scratch filesystems in a few
ways, which is where all three notes come from:

1. **Work from local disk, not from the mount.** The container starts with its working
   directory on the FUSE bucket mount, and Ray's runtime-env packaging hashes the working
   directory — which is unboundedly slow over a FUSE object-store mount. The driver switches
   to a local `/tmp` working directory before starting Ray.
2. **CPU autodetection.** Ray detects the container cgroup limit (1 CPU on `cpu-basic`),
   which is below the streaming executor's resource floor. The driver passes an explicit
   logical CPU count (default 8, override with `HF_JOBS_RAY_NUM_CPUS`) — a safe overcommit
   for the I/O-bound demo stages. On `cpu-upgrade` and larger flavors Ray's autodetected
   count already clears the floor, so the override is only needed on `cpu-basic`.
3. **No POSIX rename on the mount.** Curator writes completion manifests atomically
   (write + rename), which bucket mounts do not support. The driver checkpoints to local disk
   and publishes the manifest JSON files to the bucket with plain writes after the run, so
   `retry_array.py` reads them exactly as it would on a cluster filesystem. One consequence:
   fine-grained resume state stays Job-local, so a retried shard reprocesses its whole slice —
   the same shard-granularity semantics as the Slurm retry workflow.

---

## Configuration reference

### `launch_array_hf.sh` environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BUCKET` | `hf://buckets/<your-username>/curator-tutorial-staging` | Bucket mounted at `/mnt` in every Job |
| `FLAVOR` | `cpu-basic` | Job hardware flavor |
| `JOB_TIMEOUT` | `30m` | Per-Job timeout |
| `SHARE_DIR` | `/mnt/curator-array-demo` | Shared prefix on the mounted bucket (`input/`, `out/`, `ckpt/` live under it) |
| `INPUT_FILE_TYPE` | `jsonl` | `jsonl` or `parquet` |
| `OUTPUT_FILE_TYPE` | `jsonl` | `jsonl` or `parquet` |
| `FILES_PER_PARTITION` | `1` | Files grouped into each source task |

### `run_shard_hf.py` environment variables (forwarded by the launcher)

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `/mnt/curator-array-demo/input` | Input directory on the mount |
| `OUTPUT_DIR` | `/mnt/curator-array-demo/out` | Output directory on the mount |
| `CHECKPOINT_PATH` | `/mnt/curator-array-demo/ckpt` | Shared destination for completion manifests |
| `HF_JOBS_RAY_NUM_CPUS` | `8` | Logical CPU count passed to `RayClient`; set `0` to use Ray's autodetection |

---

## Adapting to your own pipeline

The scheduler contract is small: export the `NEMO_CURATOR_SLURM_ARRAY_*` variables, give every
Job the same shared storage, and make sure completion manifests land on that storage. To run
your own pipeline instead of the demo:

1. Point `UPSTREAM_COMMIT` / `ARRAY_PIPELINE_SHA256` / `ARRAY_PIPELINE_URL` in
   `run_shard_hf.py` at your pipeline file (or inline your pipeline into the driver), and
   adjust the CLI arguments it passes.
2. Keep the three container notes in mind if you restructure the driver: start Ray from local
   disk, give `RayClient` an explicit CPU count on `cpu-basic`, and keep Curator's checkpoint
   path on local disk with manifests published to the mount afterwards.
3. Pick a flavor that fits your stages (`hf jobs run --help` lists the options) and pass GPU
   counts explicitly (`RayClient(num_gpus=...)`), as in the other Curator tutorials.

---

## Troubleshooting

**Job produces no pipeline output and appears to hang after Ray starts**

The working directory is on the bucket mount and Ray's runtime-env packaging is hashing it.
The driver already switches to `/tmp` before starting Ray; if you restructured the driver,
make sure `os.chdir` to a local directory happens before any Ray interaction.

**Error about insufficient CPU resources for the streaming executor**

Ray autodetected the container cgroup limit (1 CPU on `cpu-basic`). Set
`HF_JOBS_RAY_NUM_CPUS` (or use a larger flavor, where autodetection clears the floor).

**`OSError: [Errno 95] Operation not supported` on a rename**

Something is writing atomically (write + rename) directly to the bucket mount. Keep Curator's
`--checkpoint-path` on local disk and publish manifests with plain writes, as the driver does.

**Job fails at startup with a volume-mount error**

Transient provisioning failure — the Job dies before any pipeline work runs and leaves no
completion manifest, so the standard retry flow (step 4) identifies and relaunches exactly
that shard. This is the failure mode recovered in the walkthrough above.

**`run_shard_hf.py` raises a SHA-256 mismatch**

`tutorials/slurm/array_pipeline.py` changed upstream relative to the pinned commit. Update
`UPSTREAM_COMMIT` and `ARRAY_PIPELINE_SHA256` in `run_shard_hf.py` together
(`sha256sum tutorials/slurm/array_pipeline.py` prints the new digest).
