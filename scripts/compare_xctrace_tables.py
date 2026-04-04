#!/usr/bin/env python3
"""Compare row counts for selected xctrace table schemas across two traces.

Example:
  python3 scripts/compare_xctrace_tables.py \
    --trace-a traces/xctrace/gemma2_softcap_grouped_on_20260404.trace \
    --label-a on \
    --trace-b traces/xctrace/gemma2_softcap_grouped_off_20260404.trace \
    --label-b off \
    --output benchmarks/m5_gemma2_softcap_trace_table_counts_2026-04-04.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_SCHEMAS = [
    "metal-application-command-buffer-submissions",
    "metal-application-encoders-list",
    "metal-gpu-intervals",
    "metal-gpu-execution-points",
    "metal-command-buffer-completed",
]


def run_xctrace_export(trace_path: Path, schema: str) -> int:
    xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    proc = subprocess.run(
        [
            "xcrun",
            "xctrace",
            "export",
            "--input",
            str(trace_path),
            "--xpath",
            xpath,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return -1
    return len(re.findall(r"<row\b", proc.stdout))


def parse_profile_metrics(log_path: Path) -> dict[str, float | None]:
    if not log_path.exists():
        return {"prefill_tok_s": None, "decode_tok_s": None}
    txt = log_path.read_text(errors="ignore")
    pre = re.search(r"Prefill:.*\(([\d.]+) tok/s\)", txt)
    dec = re.search(r"Decode:.*\(([\d.]+) tok/s\)", txt)
    return {
        "prefill_tok_s": float(pre.group(1)) if pre else None,
        "decode_tok_s": float(dec.group(1)) if dec else None,
    }


def companion_log(trace_path: Path) -> Path:
    # trace path ".../<name>.trace" -> ".../<name>_target_stdout.log"
    return trace_path.with_suffix("").with_name(trace_path.with_suffix("").name + "_target_stdout.log")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-a", required=True, type=Path)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--trace-b", required=True, type=Path)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--schema", action="append", default=[])
    args = parser.parse_args()

    if not args.trace_a.exists():
        print(f"error: trace not found: {args.trace_a}", file=sys.stderr)
        return 1
    if not args.trace_b.exists():
        print(f"error: trace not found: {args.trace_b}", file=sys.stderr)
        return 1

    schemas = args.schema if args.schema else DEFAULT_SCHEMAS

    rows: list[dict[str, object]] = []
    for schema in schemas:
        a_rows = run_xctrace_export(args.trace_a, schema)
        b_rows = run_xctrace_export(args.trace_b, schema)
        delta = None if a_rows < 0 or b_rows < 0 else b_rows - a_rows
        rows.append(
            {
                "schema": schema,
                f"{args.label_a}_rows": a_rows,
                f"{args.label_b}_rows": b_rows,
                "delta_rows": delta,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["schema", f"{args.label_a}_rows", f"{args.label_b}_rows", "delta_rows"]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    a_metrics = parse_profile_metrics(companion_log(args.trace_a))
    b_metrics = parse_profile_metrics(companion_log(args.trace_b))
    print(f"wrote: {args.output}")
    print(
        f"{args.label_a} prefill/decode tok/s: "
        f"{a_metrics['prefill_tok_s']} / {a_metrics['decode_tok_s']}"
    )
    print(
        f"{args.label_b} prefill/decode tok/s: "
        f"{b_metrics['prefill_tok_s']} / {b_metrics['decode_tok_s']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
