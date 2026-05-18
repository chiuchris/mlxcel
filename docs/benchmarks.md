# Benchmarks

This page documents how benchmark claims should be recorded for `mlxcel`. It is
intentionally conservative: do not publish aggregate speedup numbers without the
raw per-model rows and the exact software/hardware versions used to produce them.

## What to record

For every benchmark run, include:

- hardware model and memory size;
- operating system version;
- `mlxcel` version or commit;
- pinned MLX commit/version;
- comparison runtime version (`mlx-lm`, `mlx-vlm`, or another baseline);
- model checkpoint name and quantization format;
- prompt length, requested decode length, batch size, and warmup policy;
- cache mode and server/generation flags;
- raw per-model prefill and decode throughput where available.

Averages are useful only after the raw rows are available. Avoid statements such
as "faster than X" unless the comparable model set and exclusions are explicit.

## Historical reference from README

The root README mentions one historical run:

- Date: 2026-05-08
- Hardware: Mac Studio M1 Ultra
- Runtime: mlxcel 0.0.25
- MLX: 0.31.2
- Baseline: mlx-lm 0.31.3
- Scope: 37 comparable 4-bit text checkpoints
- Reported decode-throughput average: about 119% of the mlx-lm baseline

That number is kept as a historical reference, not as a current release claim.
Before using it in release notes or capacity planning, add the raw table and
rerun the suite on the target hardware.

## Suggested benchmark commands

The repository contains benchmark helper scripts under `scripts/`. The exact
arguments may evolve, so inspect each script before publishing results.

```bash
# Single-model decode benchmark shape.
./scripts/bench_decode.sh -m models/<checkpoint> --runs 3

# Multi-model suite shape.
./scripts/bench_all_models.sh --hardware <name> --cooldown 45 --big-cooldown 60
```

## Recommended output layout

Add benchmark artifacts under a dedicated directory before publishing a release,
for example:

```text
benchmarks/
  2026-05-08_m1-ultra_text.csv
  2026-05-08_m1-ultra_vlm.csv
  README.md
```

Each CSV should be machine-readable and accompanied by a short Markdown note
that describes methodology, exclusions, and known failures.

## Caveats

- **Thermals matter.** Apple Silicon decode throughput changes with sustained
  load; record cooldown and run order.
- **MLX pin matters.** Kernel selection can change when the pinned MLX commit
  changes.
- **VLM comparisons are separate from text comparisons.** Vision preprocessing,
  image resolution, and prompt construction differ by family.
- **CUDA numbers are not interchangeable across GPUs.** Publish the SM target
  and driver/toolkit versions with the result.
