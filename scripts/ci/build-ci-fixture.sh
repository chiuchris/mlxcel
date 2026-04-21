#!/usr/bin/env bash
#
# Build a CI fixture tarball for the pipeline-parallel smoke test.
#
# The `pipeline-parallel-ci.yml` workflow does not call HuggingFace at
# build time; instead it fetches a tarball from a pinned GitHub Release
# asset. This script is the reproducible recipe that produced that
# asset — run it once per fixture rotation and upload the output to the
# matching `ci-fixtures/<name>-v<N>` release.
#
# The script downloads the target model from HuggingFace at a pinned
# revision (so rebuilds on a different day produce byte-identical
# tarballs as long as upstream does not rewrite history) and packages
# the result as a gzipped tar. The top-level entry inside the archive
# is the model directory basename, so `tar -xzf <tarball> -C models/`
# lands the weights at `models/<name>/`.
#
# Usage:
#   scripts/ci/build-ci-fixture.sh <hf-model-id> <hf-revision> [<output-dir>]
#
# Arguments:
#   <hf-model-id>  HuggingFace repo id. Example: mlx-community/Qwen3-0.6B-4bit
#   <hf-revision>  Git revision (commit SHA) on HF to pin. Use `hf repo-info <id>`
#                  to discover the current commit.
#   <output-dir>   Directory to write the tarball into. Default:
#                  $TMPDIR/mlxcel-ci-fixtures (or /tmp/mlxcel-ci-fixtures).
#
# Output:
#   - Tarball at <output-dir>/<model-basename>.tar.gz, where `model-basename`
#     is the lowercased final component of <hf-model-id>.
#   - SHA256 of the tarball printed to stderr as a structured log line.
#   - Absolute tarball path printed to stdout so callers can chain:
#       tarball="$(scripts/ci/build-ci-fixture.sh <id> <rev>)"
#
# Requires: hf (HuggingFace CLI), tar, sha256sum or shasum -a 256.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ci/_common.sh
source "$SCRIPT_DIR/_common.sh"

export PP_CI_SCRIPT_TAG="build-ci-fixture"

usage() {
  cat >&2 <<'EOF'
Usage: build-ci-fixture.sh <hf-model-id> <hf-revision> [<output-dir>]

  Downloads an HF model at a pinned revision and packages it as a
  tar.gz suitable for uploading as a GitHub Release asset.

Example:
  scripts/ci/build-ci-fixture.sh \
    mlx-community/Qwen3-0.6B-4bit \
    0e50a30b0c2ac0fa23f8c4b9bf11e4e9ed8a4d6f
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 64
fi

HF_MODEL_ID="$1"
HF_REVISION="$2"
OUT_DIR="${3:-${TMPDIR:-/tmp}/mlxcel-ci-fixtures}"

pp_ci_require hf tar

# sha256sum on Linux, shasum on macOS — pick whichever is available.
if command -v sha256sum >/dev/null 2>&1; then
  sha_cmd=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
  sha_cmd=(shasum -a 256)
else
  pp_ci_log "missing required tool: sha256sum or shasum"
  exit 127
fi

pp_ci_mkdir_p "$OUT_DIR"

model_basename="$(basename "$HF_MODEL_ID" | tr '[:upper:]' '[:lower:]')"
workdir="$(mktemp -d -t mlxcel-ci-fixture.XXXXXX)"
trap 'rm -rf "$workdir"' EXIT

pp_ci_log "downloading $HF_MODEL_ID @ $HF_REVISION into $workdir/$model_basename"
hf download "$HF_MODEL_ID" \
  --revision "$HF_REVISION" \
  --local-dir "$workdir/$model_basename" \
  >&2

tarball="$OUT_DIR/$model_basename.tar.gz"
pp_ci_log "packaging to $tarball"
# `-C "$workdir"` keeps the top-level entry inside the archive as
# `<model_basename>/...` so consumers can extract straight into
# `models/` and end up with `models/<model_basename>/`.
#
# `--exclude='<model_basename>/.cache'` strips the `.cache/huggingface/`
# bookkeeping directory that `hf download` writes alongside the model
# weights — those files are a download-side cache artifact, not part of
# the model itself, and their presence changes the tarball's sha256 on
# every rebuild even when the resolved revision is identical.
tar --exclude="$model_basename/.cache" \
  -czf "$tarball" -C "$workdir" "$model_basename"

sha256="$("${sha_cmd[@]}" "$tarball" | awk '{print $1}')"
if command -v stat >/dev/null 2>&1; then
  size_bytes="$(stat -c%s "$tarball" 2>/dev/null || stat -f%z "$tarball")"
  size_mib=$(( size_bytes / 1024 / 1024 ))
  pp_ci_log "fixture ready: $tarball (${size_mib} MiB, ${size_bytes} bytes)"
else
  pp_ci_log "fixture ready: $tarball"
fi
pp_ci_log "sha256: $sha256"
printf '%s\n' "$tarball"
