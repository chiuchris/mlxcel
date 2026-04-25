// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! HuggingFace model repository downloader (issue #457).
//!
//! Provides a single source of truth for downloading model snapshots from the
//! HuggingFace Hub. Both the `mlxcel` CLI and the `mlxcel-server` binary call
//! into the same [`download_repo`] entry point so the supported file set and
//! flag semantics stay in lock-step.
//!
//! # Design
//!
//! - **Allow-list filtering** — files are kept based on extension/name patterns
//!   (`config.json`, `*.safetensors`, tokenizers, processor configs, ...). New
//!   model families work without code changes; non-MLX artifacts (`*.bin`,
//!   `*.gguf`, ...) are skipped to save bandwidth and disk.
//! - **Token resolution order** — explicit `--token` > `HF_TOKEN` env >
//!   `HUGGING_FACE_HUB_TOKEN` env > anonymous.
//! - **Default destination** — `models/<repo_basename>` under the current
//!   working directory, mirroring the `CLAUDE.md` "Testing Models" convention.
//! - **Caching** — without `--force`, an existing snapshot with all expected
//!   files at the right size is treated as a no-op. With `--force`, every file
//!   is re-fetched and overwritten.
//! - **Progress** — one stdout line per file at start and on completion. No
//!   incremental byte progress (the synchronous `hf-hub` 0.5 API does not
//!   expose it cheaply).

mod cli;
mod errors;
mod filters;

pub use cli::DownloadArgs;
pub use errors::map_hf_error;
pub use filters::{is_wanted_file, repo_basename};

use anyhow::{Context, Result, anyhow};
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Resolved options for a download invocation.
///
/// Constructed from CLI arguments via [`DownloadOptions::from_args`] (the
/// shared adapter both binaries use). The struct exists so that programmatic
/// callers (and unit tests) can drive [`download_repo`] without going through
/// clap parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadOptions {
    /// HuggingFace repository identifier, e.g. `mlx-community/Qwen3-4B-4bit`.
    pub repo_id: String,
    /// Local destination directory. When `None`, defaults to
    /// `models/<repo_basename>` under the current working directory.
    pub local_dir: Option<PathBuf>,
    /// Repository revision (branch, tag, or commit). Defaults to `main` when
    /// `None`.
    pub revision: Option<String>,
    /// Authentication token override. When `None`, falls back to environment
    /// variables (`HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN`).
    pub token: Option<String>,
    /// Re-download every file even when a complete snapshot is already
    /// present locally.
    pub force: bool,
}

impl DownloadOptions {
    /// Convert the binary-side clap struct into a runtime options bundle.
    pub fn from_args(args: &DownloadArgs) -> Self {
        Self {
            repo_id: args.repo_id.clone(),
            local_dir: args.local_dir.clone(),
            revision: args.revision.clone(),
            token: args.token.clone(),
            force: args.force,
        }
    }

    /// Resolve the destination directory, applying the
    /// `models/<repo_basename>` default when no explicit `--local-dir` was
    /// given.
    pub fn resolve_local_dir(&self) -> PathBuf {
        match &self.local_dir {
            Some(path) => path.clone(),
            None => PathBuf::from("models").join(repo_basename(&self.repo_id)),
        }
    }
}

/// Resolve the effective HuggingFace token using the documented precedence:
/// explicit `--token` flag, then `HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN`,
/// then anonymous (`None`).
///
/// Empty values from environment variables are treated as anonymous so that
/// `HF_TOKEN=""` does not poison the request with a malformed `Authorization`
/// header.
pub fn resolve_token(explicit: Option<&str>) -> Option<String> {
    if let Some(t) = explicit {
        let trimmed = t.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    for env_key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(value) = std::env::var(env_key) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Build a configured `hf-hub` [`Api`] honoring the resolved auth token.
fn build_api(token: Option<String>) -> Result<Api> {
    let mut builder = ApiBuilder::from_env().with_progress(false);
    if let Some(tok) = token {
        builder = builder.with_token(Some(tok));
    }
    builder
        .build()
        .map_err(|err| anyhow!("Failed to initialize Hugging Face API client: {err}"))
}

/// Open the [`Repo`] handle for the requested model + revision.
fn build_repo_handle(repo_id: &str, revision: Option<&str>) -> Repo {
    match revision {
        Some(rev) => Repo::with_revision(repo_id.to_string(), RepoType::Model, rev.to_string()),
        None => Repo::new(repo_id.to_string(), RepoType::Model),
    }
}

/// Download a HuggingFace model repository snapshot into a local directory.
///
/// On success, every allow-listed file from the upstream repository is present
/// inside `local_dir` (resolved per [`DownloadOptions::resolve_local_dir`]).
///
/// # Errors
///
/// Returns actionable [`anyhow::Error`] messages for the common failure modes:
/// invalid repo id, missing authentication on a gated repo, missing revision,
/// network failure, and on-disk I/O errors.
pub fn download_repo(opts: DownloadOptions) -> Result<()> {
    let local_dir = opts.resolve_local_dir();
    let token = resolve_token(opts.token.as_deref());
    let api = build_api(token)?;
    let repo = build_repo_handle(&opts.repo_id, opts.revision.as_deref());
    let api_repo = api.repo(repo);

    println!(
        "[mlxcel download] repo={} revision={} dest={}",
        opts.repo_id,
        opts.revision.as_deref().unwrap_or("main"),
        local_dir.display(),
    );

    let info = api_repo
        .info()
        .map_err(|err| map_hf_error(err, &opts.repo_id, opts.revision.as_deref(), None))?;
    let wanted: Vec<String> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|name| is_wanted_file(name))
        .collect();

    if wanted.is_empty() {
        return Err(anyhow!(
            "Repository '{}' contains no files matching the mlxcel allow-list \
             (config.json, tokenizer*, *.safetensors, ...). Nothing to download.",
            opts.repo_id
        ));
    }

    println!(
        "[mlxcel download] {} files queued (filtered from {} total siblings)",
        wanted.len(),
        info.siblings.len(),
    );

    fs::create_dir_all(&local_dir).with_context(|| {
        format!(
            "Failed to create destination directory {}",
            local_dir.display()
        )
    })?;

    if opts.force {
        println!("[mlxcel download] --force: refreshing every file");
    } else if snapshot_complete(&local_dir, &wanted) {
        println!(
            "[mlxcel download] all expected files already present at {}, skipping (use --force to refresh)",
            local_dir.display(),
        );
        return Ok(());
    }

    // Canonicalize `local_dir` once. We compare every per-file destination
    // parent against this prefix to refuse writes that escape the snapshot
    // directory (defense in depth on top of the basename allow-list and the
    // `is_safe_relative_path` filter in `is_wanted_file`).
    let canonical_local = fs::canonicalize(&local_dir)
        .with_context(|| format!("Failed to canonicalize destination {}", local_dir.display()))?;

    let total = wanted.len();
    let mut downloaded = 0usize;
    let mut skipped = 0usize;
    let mut total_bytes: u64 = 0;
    for (idx, filename) in wanted.iter().enumerate() {
        let dest = local_dir.join(filename);

        // Defense in depth: even if a malicious sibling slipped through the
        // allow-list, refuse to touch any path whose parent does not resolve
        // inside `canonical_local`. This catches symlink shenanigans and any
        // future regression in the basename filter.
        let dest_parent = dest.parent().unwrap_or(&local_dir);
        fs::create_dir_all(dest_parent).with_context(|| {
            format!(
                "Failed to create directory {} for {filename}",
                dest_parent.display()
            )
        })?;
        let canonical_parent = fs::canonicalize(dest_parent).with_context(|| {
            format!(
                "Failed to canonicalize destination parent {}",
                dest_parent.display()
            )
        })?;
        if !canonical_parent.starts_with(&canonical_local) {
            return Err(anyhow!(
                "Refusing to write '{filename}' outside of '{}': resolved to '{}'.",
                local_dir.display(),
                canonical_parent.display(),
            ));
        }

        if !opts.force && file_exists_nonempty(&dest) {
            println!(
                "[{idx_1}/{total}] cached: {filename}",
                idx_1 = idx + 1,
                total = total,
                filename = filename,
            );
            skipped += 1;
            continue;
        }

        println!(
            "[{idx_1}/{total}] downloading: {filename}",
            idx_1 = idx + 1,
            total = total,
            filename = filename,
        );
        let started = Instant::now();
        let cached_path = api_repo.download(filename).map_err(|err| {
            map_hf_error(
                err,
                &opts.repo_id,
                opts.revision.as_deref(),
                Some(filename.as_str()),
            )
        })?;
        copy_into_local(&cached_path, &dest, filename)?;
        let bytes = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
        total_bytes += bytes;
        downloaded += 1;
        let elapsed = started.elapsed();
        println!(
            "[{idx_1}/{total}] done: {filename} ({size} in {secs:.1}s)",
            idx_1 = idx + 1,
            total = total,
            filename = filename,
            size = format_bytes(bytes),
            secs = elapsed.as_secs_f64(),
        );
    }

    println!(
        "[mlxcel download] complete: downloaded={} cached={} total_size={} dest={}",
        downloaded,
        skipped,
        format_bytes(total_bytes),
        local_dir.display(),
    );
    Ok(())
}

/// True when every wanted file is present in `local_dir` with non-zero size.
///
/// A simple presence + non-empty check is sufficient because `hf-hub` writes
/// to a temp path and renames atomically, so partial files do not normally
/// remain. `--force` is the documented escape hatch when this heuristic is
/// not enough.
fn snapshot_complete(local_dir: &Path, wanted: &[String]) -> bool {
    if !local_dir.join("config.json").exists() {
        return false;
    }
    wanted
        .iter()
        .all(|name| file_exists_nonempty(&local_dir.join(name)))
}

fn file_exists_nonempty(path: &Path) -> bool {
    fs::metadata(path)
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false)
}

/// Copy a cached blob from `hf-hub`'s store into the user-facing destination.
///
/// `hf-hub` returns a path inside its cache (often a symlink to a content-
/// addressed blob). For the user-facing `--local-dir`, we want a real file
/// the rest of mlxcel can mmap directly, so we materialize a copy.
///
/// **Atomicity:** writes go to a sibling tempfile, then `fs::rename`
/// atomically installs it over `dest`. On the same filesystem on Unix this
/// rename is atomic, so a concurrent `mlxcel generate` reading
/// `model.safetensors` while a `mlxcel download --force` is in flight will
/// see either the old file or the new file — never `ENOENT` between the two
/// or a partially-written shard mid-copy.
fn copy_into_local(src: &Path, dest: &Path, filename: &str) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }
    let canonical_src = fs::canonicalize(src)
        .with_context(|| format!("Failed to resolve cached path for {filename}"))?;

    // Write to a sibling tempfile, then atomically rename over `dest`.
    // `fs::rename` is atomic on the same filesystem on Unix; concurrent
    // readers see either the old file or the new one, never a partial.
    let tmp_name = format!(
        ".mlxcel-partial.{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let tmp = dest.with_file_name(tmp_name);
    fs::copy(&canonical_src, &tmp).with_context(|| {
        format!(
            "Failed to copy {} from cache to {}",
            filename,
            tmp.display()
        )
    })?;
    fs::rename(&tmp, dest).with_context(|| {
        format!(
            "Failed to atomically install {} at {}",
            filename,
            dest.display()
        )
    })?;
    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes == 0 {
        return "0 B".to_string();
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
