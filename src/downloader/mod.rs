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

//! HuggingFace model repository downloader (issues #457, #648).
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
//! - **Progress** — when stderr is a tty and progress is not suppressed via
//!   env vars, per-file and aggregate `indicatif` progress bars render during
//!   the actual byte stream (Path B2 direct reqwest streaming, issue #648).
//!   When bars are suppressed (CI, piped output, `MLXCEL_NO_PROGRESS=1`,
//!   `NO_COLOR=1`), one stdout line per file is emitted instead so CI logs
//!   remain golden-text-stable.

mod cli;
mod errors;
mod filters;
mod progress;

pub use cli::DownloadArgs;
pub use errors::map_hf_error;
pub use filters::{is_wanted_file, repo_basename};
pub use progress::should_show_progress;

use anyhow::{Context, Result, anyhow};
use futures::StreamExt;
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokio::io::AsyncWriteExt;

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
///
/// We keep `with_progress(false)` because progress is driven by our own
/// indicatif bars (issue #648). hf-hub is used only for `info()` (manifest
/// fetch) — the actual file bytes come from direct reqwest streaming.
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

/// Resolve the HuggingFace endpoint base URL.
///
/// Respects `HF_ENDPOINT` env var (allows using a mirror), otherwise defaults
/// to `https://huggingface.co`.
fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

/// Build the download URL for a single file in a HuggingFace repository.
fn file_url(endpoint: &str, repo_id: &str, revision: &str, filename: &str) -> String {
    format!("{endpoint}/{repo_id}/resolve/{revision}/{filename}")
}

/// Download a single file via reqwest streaming, ticking the per-file and
/// aggregate progress bars as each chunk arrives.
///
/// Writes to a sibling tempfile first, then atomically renames to `dest`.
/// On error, the tempfile is removed and both progress bars are abandoned.
async fn stream_file(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    filename: &str,
    file_pb: &indicatif::ProgressBar,
    aggregate_pb: &indicatif::ProgressBar,
) -> Result<u64> {
    let tmp_name = format!(
        ".mlxcel-partial.{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    let tmp = dest.with_file_name(tmp_name);

    let result = stream_to_tempfile(client, url, &tmp, dest, filename, file_pb, aggregate_pb).await;
    if result.is_err() {
        // Best-effort cleanup — ignore errors from remove_file (file may not
        // exist yet if `File::create` itself failed). The original error from
        // streaming is the actionable one for the user.
        let _ = tokio::fs::remove_file(&tmp).await;
    }
    result
}

/// Inner implementation of [`stream_file`]: stream bytes into `tmp`, then
/// atomically rename to `dest`. Callers are responsible for cleaning up `tmp`
/// on error.
async fn stream_to_tempfile(
    client: &reqwest::Client,
    url: &str,
    tmp: &Path,
    dest: &Path,
    filename: &str,
    file_pb: &indicatif::ProgressBar,
    aggregate_pb: &indicatif::ProgressBar,
) -> Result<u64> {
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("HTTP request failed for {filename}"))?;

    let status = response.status();
    if !status.is_success() {
        let code = status.as_u16();
        return Err(anyhow!(
            "HTTP {code} downloading '{filename}'. \
             Check authentication (--token / HF_TOKEN) or that the repository exists."
        ));
    }

    let mut out = tokio::fs::File::create(tmp)
        .await
        .with_context(|| format!("Failed to create tempfile for {filename}"))?;

    let mut stream = response.bytes_stream();
    let mut bytes_written: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("Stream error while downloading {filename}"))?;
        out.write_all(&chunk)
            .await
            .with_context(|| format!("Write error while downloading {filename}"))?;
        let chunk_len = chunk.len() as u64;
        bytes_written += chunk_len;
        file_pb.inc(chunk_len);
        aggregate_pb.inc(chunk_len);
    }

    out.flush()
        .await
        .with_context(|| format!("Flush error for {filename}"))?;
    drop(out);

    tokio::fs::rename(tmp, dest)
        .await
        .with_context(|| format!("Failed to atomically install {filename}"))?;

    Ok(bytes_written)
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
    let api = build_api(token.clone())?;
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

    let show_bars = should_show_progress();

    // Build the reqwest client once and share across all file downloads.
    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
    let client = rt.block_on(async {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(ref tok) = token {
            let auth_val = format!("Bearer {tok}");
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&auth_val).expect("token must be ASCII"),
            );
        }
        reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .context("Failed to create HTTP client")
    })?;

    let endpoint = hf_endpoint();
    let revision = opts.revision.as_deref().unwrap_or("main");

    // Build per-file sizes map for accurate bar lengths. `hf-hub 0.5` does
    // not expose per-file sizes in the manifest (Siblings only has `rfilename`),
    // so we issue sequential HEAD requests here — N RTTs before the first byte
    // streams. For typical 5-15 file snapshots this adds ~1-2s of pre-stream
    // wallclock; converting to `futures::stream::iter(...).buffer_unordered(N)`
    // is a worthwhile follow-up if HEAD latency proves bothersome. Sizes are
    // best-effort: if a HEAD fails we fall back to 0 (indeterminate bar).
    // We skip the HEAD pass entirely for cached files (handled below) and when
    // progress bars are suppressed.
    let size_map: std::collections::HashMap<String, u64> = if show_bars {
        rt.block_on(async {
            let mut map = std::collections::HashMap::new();
            for filename in &wanted {
                let url = file_url(&endpoint, &opts.repo_id, revision, filename);
                let size = client
                    .head(&url)
                    .send()
                    .await
                    .ok()
                    .and_then(|r| {
                        r.headers()
                            .get(reqwest::header::CONTENT_LENGTH)
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.parse::<u64>().ok())
                    })
                    .unwrap_or(0);
                map.insert(filename.clone(), size);
            }
            map
        })
    } else {
        std::collections::HashMap::new()
    };

    let total_known_bytes: u64 = size_map.values().sum();

    let mp = progress::create_multi_progress();
    let aggregate_pb = progress::add_aggregate_bar(&mp, total_known_bytes);

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
            // Cached-file fast path: emit a single line, do NOT animate a bar.
            // The aggregate bar is not ticked since no bytes are transferred;
            // instead we advance it by the expected file size to keep the total
            // accurate.
            let cached_size = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
            println!("[{}/{total}] cached: {filename}", idx + 1,);
            // Advance aggregate bar by the cached file size so total progress
            // reflects what is on disk, not just what was downloaded this session.
            aggregate_pb.inc(cached_size);
            total_bytes += cached_size;
            skipped += 1;
            continue;
        }

        let file_size = *size_map.get(filename.as_str()).unwrap_or(&0);
        let file_pb = progress::add_file_bar(&mp, filename, file_size);

        if !show_bars {
            println!("[{}/{total}] downloading: {filename}", idx + 1,);
        }

        let started = Instant::now();
        let url = file_url(&endpoint, &opts.repo_id, revision, filename);

        let result = rt.block_on(stream_file(
            &client,
            &url,
            &dest,
            filename,
            &file_pb,
            &aggregate_pb,
        ));

        match result {
            Ok(bytes) => {
                total_bytes += bytes;
                downloaded += 1;
                let elapsed = started.elapsed();
                file_pb.finish_and_clear();
                println!(
                    "[{}/{total}] done: {filename} ({size} in {secs:.1}s)",
                    idx + 1,
                    size = format_bytes(bytes),
                    secs = elapsed.as_secs_f64(),
                );
            }
            Err(err) => {
                // Red-finish the per-file bar so users see which file failed.
                file_pb.abandon_with_message(format!("FAILED: {filename}"));
                return Err(err);
            }
        }
    }

    aggregate_pb.finish_and_clear();
    drop(mp);

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
/// A simple presence + non-empty check is sufficient because we write to a
/// temp path and rename atomically, so partial files do not normally remain.
/// `--force` is the documented escape hatch when this heuristic is not enough.
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
