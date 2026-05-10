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

//! Shared request-media helpers for server routes.
//!
//! Keeping image-source parsing at the HTTP edge makes it easier to add new
//! request formats without growing individual route handlers. The helpers stay
//! async so local file reads and remote URL fetches do not block Axum workers.

use base64::Engine;
use futures::StreamExt;
use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Duration,
};

use crate::multimodal::video::{TempFile, is_video_file};

use super::types::ChatCompletionRequest;
use super::types::request::{InputAudio, VideoUrl};

pub(crate) async fn extract_chat_image_data(request: &ChatCompletionRequest) -> Vec<Vec<u8>> {
    collect_image_data(request.image_urls()).await
}

/// Environment variable holding the comma-separated list of canonical
/// directories permitted to be referenced via `file://` URIs and bare local
/// paths in `video_url` content blocks (issue #596).
///
/// Empty or unset = fail-closed: every local-filesystem video reference is
/// rejected. Operators must opt in explicitly to enable file uploads.
///
/// Operators are responsible for the directory permissions: a world- or
/// group-writable allowlist directory exposes the resolver to a TOCTOU race
/// (an attacker with write access can swap the file between
/// `extract_chat_video_paths_with_allowlist` returning and ffmpeg opening
/// the file by path). The startup helper [`scan_insecure_allowlist_dirs`]
/// detects this and emits a warning; restrict the directory to mode 0750
/// or stricter to clear it.
pub(crate) const VIDEO_DIR_ALLOWLIST_ENV: &str = "MLXCEL_VIDEO_DIR_ALLOWLIST";

/// Resolve `video_url` content blocks from a chat request to local file
/// paths (issue #553, wired up in #596). For `file://` and bare local paths
/// the path is canonicalised and validated against the directory allowlist
/// from `MLXCEL_VIDEO_DIR_ALLOWLIST`. For `data:video/...;base64,...` and
/// `http(s)://` we materialise the bytes to a temporary file because
/// `ffmpeg` needs an on-disk handle. Returned `(PathBuf, Option<f64>)`
/// tuples carry the optional per-video FPS override from
/// [`VideoUrl::fps`].
///
/// Returns `(paths, guards)` where `guards` owns RAII `TempFile` drop
/// guards for every caller-owned temp file (data-URI decode and HTTP fetch).
/// Pre-existing `file://` paths are not wrapped because the caller does not
/// own them. Stash the `guards` vector in a struct that lives for the full
/// duration of the request handler — when that struct drops, the temp files
/// are removed automatically.
pub(crate) async fn extract_chat_video_paths(
    request: &ChatCompletionRequest,
) -> (Vec<(PathBuf, Option<f64>)>, Vec<TempFile>) {
    let allowlist = video_dir_allowlist_from_env();
    extract_chat_video_paths_with_allowlist(request, &allowlist).await
}

/// Test/internal-friendly variant of [`extract_chat_video_paths`] that accepts
/// the allowlist directly. Production code reads the env var via the public
/// wrapper above; tests inject a controlled allowlist so they don't depend
/// on process-wide state.
///
/// # TOCTOU note
///
/// The allowlist guard runs `tokio::fs::canonicalize` and `tokio::fs::metadata`
/// on the resolved path before [`std::process::Command`] re-opens the file
/// inside `ffmpeg`. An attacker with write access to one of the allowlisted
/// directories could in principle race the canonicalise → ffmpeg-open
/// window by swapping a regular file for a symlink that points outside.
/// The full FD-passing fix is tracked separately; for now the operator-side
/// guard is the writability check fired at startup in `startup.rs`.
pub(crate) async fn extract_chat_video_paths_with_allowlist(
    request: &ChatCompletionRequest,
    allowlist: &[PathBuf],
) -> (Vec<(PathBuf, Option<f64>)>, Vec<TempFile>) {
    let mut paths = Vec::new();
    let mut guards = Vec::new();
    for video in request.video_urls() {
        if let Some((path, guard)) = resolve_video_url(&video, allowlist).await {
            paths.push((path, video.fps));
            if let Some(g) = guard {
                guards.push(g);
            }
        }
    }
    (paths, guards)
}

/// Read [`VIDEO_DIR_ALLOWLIST_ENV`] and canonicalise each entry. Entries
/// that fail to canonicalise (typo, missing directory) are dropped with a
/// warning rather than poisoning the whole list.
///
/// An empty/unset env yields an empty `Vec`, which is the deliberate
/// fail-closed default — no `file://` URI or bare local path can resolve
/// until an operator opts in.
///
/// Callable from sync contexts (used by [`crate::server::startup`] for the
/// startup-time writability check). The hot path on the request side reads
/// the result via the async [`extract_chat_video_paths`] wrapper which then
/// passes it down by reference, so the blocking `std::fs::canonicalize` here
/// runs once at startup and once per request handler — both off the Tokio
/// hot loop, so it is acceptable to keep this synchronous.
pub(crate) fn video_dir_allowlist_from_env() -> Vec<PathBuf> {
    let raw = std::env::var(VIDEO_DIR_ALLOWLIST_ENV).unwrap_or_default();
    if raw.trim().is_empty() {
        return Vec::new();
    }
    raw.split(',')
        .filter_map(|entry| {
            let trimmed = entry.trim();
            if trimmed.is_empty() {
                return None;
            }
            match std::fs::canonicalize(trimmed) {
                Ok(canonical) => Some(canonical),
                Err(err) => {
                    tracing::warn!(
                        "{VIDEO_DIR_ALLOWLIST_ENV} entry {trimmed:?} could not be canonicalised: \
                         {err}; dropping from allowlist"
                    );
                    None
                }
            }
        })
        .collect()
}

/// Walk every directory in `allowlist` and return the subset whose Unix
/// permissions allow group or world write access (mode bits `0o022`).
///
/// Used by the server startup hook to surface a `tracing::warn!` when an
/// operator opts into the video-URL feature with a loose-mode directory.
/// Because the resolver in this module canonicalises and stats the file,
/// then surrenders the path to ffmpeg, an attacker who can write inside
/// the directory could swap a benign mp4 for a symlink or another file
/// type between resolution and ffmpeg open. Restricting the directory to
/// the server uid (`0750`) closes that race for the common case.
///
/// On non-Unix targets the file mode is unavailable, so this function
/// returns an empty `Vec`.
///
/// Returning the offending directories rather than emitting the warning
/// directly keeps the helper unit-testable: tests can construct a
/// world-writable temp directory and assert the helper detects it.
#[must_use]
pub(crate) fn scan_insecure_allowlist_dirs(allowlist: &[PathBuf]) -> Vec<PathBuf> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut insecure = Vec::new();
        for dir in allowlist {
            match std::fs::metadata(dir) {
                Ok(meta) => {
                    let mode = meta.permissions().mode();
                    // Group or other write bits (0o020, 0o002) are the
                    // dangerous flags. Owner-only writability (0o200) is fine.
                    if mode & 0o022 != 0 {
                        insecure.push(dir.clone());
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        "Allowlist directory {dir:?} could not be stat'd for the security check: {err}",
                    );
                }
            }
        }
        insecure
    }
    #[cfg(not(unix))]
    {
        let _ = allowlist;
        Vec::new()
    }
}

/// Resolve a single [`VideoUrl`] to a `(path, guard)` pair.
///
/// The second element is `Some(TempFile)` when the resolver materialised the
/// content into a server-owned temp file (data-URI decode, HTTP fetch). For
/// `file://` and bare local paths the server does not own the file, so the
/// guard is `None` and the file is left in place after the request finishes.
async fn resolve_video_url(
    video: &VideoUrl,
    allowlist: &[PathBuf],
) -> Option<(PathBuf, Option<TempFile>)> {
    let url = &video.url;

    if url.starts_with("data:video/") {
        return decode_video_data_uri(url).await;
    }
    if let Some(path) = url.strip_prefix("file://") {
        return resolve_local_video_path(path, allowlist)
            .await
            .map(|p| (p, None));
    }
    if is_http_url(url) {
        return fetch_remote_video(url).await;
    }
    if let Some(path) = resolve_local_video_path(url, allowlist).await {
        return Some((path, None));
    }
    tracing::warn!("Unsupported video URL scheme or missing file: {}", url);
    None
}

/// Resolve a bare local path or the path component of a `file://` URI to a
/// canonical, allowlisted, regular video file (issue #596).
///
/// Async-friendly variant (#596 follow-up): uses `tokio::fs::canonicalize` and
/// `tokio::fs::metadata` so a slow disk or NFS mount cannot stall a Tokio
/// worker thread. Each `.await` boundary lets the runtime schedule other
/// requests while the lookup is in flight.
///
/// Rejection rules — every check is fail-closed:
/// * Empty allowlist → reject everything (operator opt-in required).
/// * `canonicalize` failure (missing file, broken symlink, permission) → reject.
/// * Canonical path not under any allowlist directory → reject. Symlinks pointing
///   outside the allowlist canonicalise to their target and are caught by this
///   check.
/// * Resolved target is not a regular file (directory, FIFO, device, socket) → reject.
/// * Filename extension is not in [`crate::multimodal::video::VIDEO_EXTENSIONS`] → reject.
async fn resolve_local_video_path(raw: &str, allowlist: &[PathBuf]) -> Option<PathBuf> {
    if allowlist.is_empty() {
        tracing::warn!(
            "{VIDEO_DIR_ALLOWLIST_ENV} is empty/unset; rejecting local video reference {raw:?}. \
             Set the env var to a comma-separated list of trusted directories to enable file uploads."
        );
        return None;
    }

    let path = Path::new(raw);

    let canonical = match tokio::fs::canonicalize(path).await {
        Ok(p) => p,
        Err(err) => {
            tracing::warn!("Failed to canonicalise video path {raw:?}: {err}");
            return None;
        }
    };

    if !allowlist.iter().any(|dir| canonical.starts_with(dir)) {
        tracing::warn!(
            "Video path {canonical:?} (from {raw:?}) is outside the allowlist; rejecting"
        );
        return None;
    }

    let metadata = match tokio::fs::metadata(&canonical).await {
        Ok(m) => m,
        Err(err) => {
            tracing::warn!("Failed to stat video path {canonical:?}: {err}");
            return None;
        }
    };
    if !metadata.is_file() {
        tracing::warn!("Video path {canonical:?} is not a regular file; rejecting");
        return None;
    }

    if !is_video_file(&canonical) {
        tracing::warn!(
            "Rejecting local video path with unsupported extension: {canonical:?} (from {raw:?})"
        );
        return None;
    }

    Some(canonical)
}

/// Maximum decoded video payload size: 1 GB. Mirrors the audio cap to
/// prevent OOM from extremely large base64 attachments.
pub(crate) const MAX_VIDEO_PAYLOAD_SIZE: usize = 1024 * 1024 * 1024;

async fn decode_video_data_uri(url: &str) -> Option<(PathBuf, Option<TempFile>)> {
    let Some((metadata, encoded_data)) = url.split_once(',') else {
        tracing::warn!("Invalid video data URI format");
        return None;
    };
    if !metadata.ends_with(";base64") {
        tracing::warn!("Unsupported video data URI encoding: {}", metadata);
        return None;
    }
    let bytes = match base64::engine::general_purpose::STANDARD.decode(encoded_data) {
        Ok(b) => b,
        Err(err) => {
            tracing::warn!("Failed to decode base64 video: {}", err);
            return None;
        }
    };
    if bytes.len() > MAX_VIDEO_PAYLOAD_SIZE {
        tracing::warn!(
            "Video payload too large ({} bytes, max {}); rejecting",
            bytes.len(),
            MAX_VIDEO_PAYLOAD_SIZE
        );
        return None;
    }
    write_video_temp_file(&bytes, infer_video_extension(metadata))
        .await
        .map(|(path, guard)| (path, Some(guard)))
}

/// Fetch a remote video URL with size enforcement applied **incrementally**
/// (issue #596 hardening).
///
/// Implementation note — buffer-then-check is a DoS vector. `response.bytes()`
/// would read the entire response body into memory before we could enforce
/// `MAX_VIDEO_PAYLOAD_SIZE`. A hostile server could advertise no
/// `Content-Length` (or lie about it) and stream until our process OOMs.
///
/// Instead we use `bytes_stream()` and accumulate into a `Vec<u8>`, checking
/// the length cap after every chunk. The moment the accumulated total exceeds
/// the cap we drop everything and return `None`. This keeps peak memory
/// bounded by `MAX_VIDEO_PAYLOAD_SIZE` regardless of what the remote server
/// does with the wire protocol.
///
/// Connect timeout, total request timeout, and redirect cap are configured
/// on the shared client (see [`http_image_client`]).
async fn fetch_remote_video(url: &str) -> Option<(PathBuf, Option<TempFile>)> {
    let response = match http_image_client().get(url).send().await {
        Ok(r) => r,
        Err(err) => {
            tracing::warn!("Failed to fetch video URL {}: {}", url, err);
            return None;
        }
    };
    let response = match response.error_for_status() {
        Ok(r) => r,
        Err(err) => {
            tracing::warn!("Video URL returned error status {}: {}", url, err);
            return None;
        }
    };

    // Streaming accumulation with per-chunk size enforcement.
    let mut accumulated: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk_result) = stream.next().await {
        let chunk = match chunk_result {
            Ok(c) => c,
            Err(err) => {
                tracing::warn!("Failed to read video response body {}: {}", url, err);
                return None;
            }
        };
        // Reject *before* extending the buffer if the new chunk would exceed
        // the cap; this keeps peak memory bounded by MAX_VIDEO_PAYLOAD_SIZE +
        // one chunk size rather than 2x cap.
        if accumulated.len().saturating_add(chunk.len()) > MAX_VIDEO_PAYLOAD_SIZE {
            tracing::warn!(
                "Remote video too large (>{} bytes after streaming chunk, max {}); rejecting",
                accumulated.len() + chunk.len(),
                MAX_VIDEO_PAYLOAD_SIZE
            );
            // Drop the partial buffer; do not retain partially fetched bytes.
            drop(accumulated);
            return None;
        }
        accumulated.extend_from_slice(&chunk);
    }

    let ext = url
        .rsplit_once('.')
        .map(|(_, ext)| ext.split('?').next().unwrap_or(ext).to_string())
        .unwrap_or_else(|| "mp4".to_string());
    write_video_temp_file(&accumulated, sanitize_video_extension(&ext))
        .await
        .map(|(path, guard)| (path, Some(guard)))
}

fn infer_video_extension(metadata: &str) -> &str {
    if metadata.contains("video/mp4") {
        "mp4"
    } else if metadata.contains("video/webm") {
        "webm"
    } else if metadata.contains("video/x-matroska") {
        "mkv"
    } else if metadata.contains("video/quicktime") {
        "mov"
    } else {
        "mp4"
    }
}

/// Write `bytes` to a fresh temp file and wrap the path in a [`TempFile`]
/// drop guard. The guard's `Drop` impl removes the file when the caller
/// drops it, so callers MUST keep the guard alive for as long as the file
/// must remain on disk (e.g., until ffmpeg has finished probing it).
///
/// Returning the guard alongside the path is the fix for the temp-file leak
/// reported during PR #600 review: every previous version returned a bare
/// `PathBuf` and relied on no-one to call cleanup explicitly. With the guard
/// in place a panic, an early-return error path, or a normal completion all
/// converge on the same cleanup behaviour.
async fn write_video_temp_file(bytes: &[u8], ext: &str) -> Option<(PathBuf, TempFile)> {
    let dir = std::env::temp_dir();
    let unique = uuid::Uuid::new_v4();
    let ext = sanitize_video_extension(ext);
    let path = dir.join(format!("mlxcel-video-{unique}.{ext}"));
    match tokio::fs::write(&path, bytes).await {
        Ok(()) => {
            let guard = TempFile::new(path.clone());
            Some((path, guard))
        }
        Err(err) => {
            tracing::warn!(
                "Failed to write video temp file {}: {}",
                path.display(),
                err
            );
            None
        }
    }
}

fn sanitize_video_extension(ext: &str) -> &'static str {
    match ext.trim_start_matches('.').to_ascii_lowercase().as_str() {
        "mp4" => "mp4",
        "webm" => "webm",
        "mkv" => "mkv",
        "mov" => "mov",
        "avi" => "avi",
        "m4v" => "m4v",
        "mpg" => "mpg",
        "mpeg" => "mpeg",
        _ => "mp4",
    }
}

/// Extract raw audio bytes from chat request audio inputs.
///
/// Supports base64-encoded inline data, `data:audio/...;base64,...` URIs,
/// `file://` paths, bare local paths, and `http(s)` URLs.
///
/// Only WAV format is currently supported; other formats are rejected early.
pub(crate) async fn extract_chat_audio_data(request: &ChatCompletionRequest) -> Vec<Vec<u8>> {
    let audio_inputs = request.audio_inputs();
    let mut audio_data = Vec::new();
    for input in &audio_inputs {
        if let Some(bytes) = read_audio_input(input).await {
            audio_data.push(bytes);
        }
    }
    audio_data
}

/// Maximum raw audio payload size after decoding: 500 MB.
/// This prevents OOM from extremely large base64 payloads before WAV
/// parsing can apply its own data-chunk limit.
const MAX_AUDIO_PAYLOAD_SIZE: usize = 500 * 1024 * 1024;

async fn read_audio_input(input: &InputAudio) -> Option<Vec<u8>> {
    // Validate format early -- only WAV is supported for now.
    if input.format != "wav" {
        tracing::warn!(
            "Unsupported audio format \'{}\'; only \'wav\' is currently supported",
            input.format
        );
        return None;
    }

    let data = &input.data;

    // data:audio/...;base64,... URI
    if data.starts_with("data:audio/") {
        return validate_audio_size(decode_data_uri(data));
    }

    // file:// prefix
    if let Some(path) = data.strip_prefix("file://") {
        return validate_audio_size(read_local_image(Path::new(path)).await);
    }

    // HTTP(S) URL
    if is_http_url(data) {
        return validate_audio_size(fetch_remote_image(data).await);
    }

    // Bare local path
    if Path::new(data).is_file() {
        return validate_audio_size(read_local_image(Path::new(data)).await);
    }

    // Try as raw base64 data
    match base64::engine::general_purpose::STANDARD.decode(data) {
        Ok(bytes) if !bytes.is_empty() => validate_audio_size(Some(bytes)),
        _ => {
            tracing::warn!("Could not decode audio input data");
            None
        }
    }
}

/// Reject audio payloads that exceed `MAX_AUDIO_PAYLOAD_SIZE`.
fn validate_audio_size(data: Option<Vec<u8>>) -> Option<Vec<u8>> {
    match data {
        Some(bytes) if bytes.len() > MAX_AUDIO_PAYLOAD_SIZE => {
            tracing::warn!(
                "Audio payload too large ({} bytes, max {}); rejecting",
                bytes.len(),
                MAX_AUDIO_PAYLOAD_SIZE
            );
            None
        }
        other => other,
    }
}

pub(crate) async fn collect_image_data<I, S>(urls: I) -> Vec<Vec<u8>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut images = Vec::new();

    for url in urls {
        if let Some(bytes) = read_image_url(url.as_ref()).await {
            images.push(bytes);
        }
    }

    images
}

pub(crate) async fn read_image_url(url: &str) -> Option<Vec<u8>> {
    if url.starts_with("data:image/") {
        return decode_data_uri(url);
    }

    if let Some(path) = url.strip_prefix("file://") {
        return read_local_image(Path::new(path)).await;
    }

    if is_http_url(url) {
        return fetch_remote_image(url).await;
    }

    if Path::new(url).is_file() {
        return read_local_image(Path::new(url)).await;
    }

    tracing::warn!("Unsupported image URL scheme: {}", url);
    None
}

fn decode_data_uri(url: &str) -> Option<Vec<u8>> {
    let Some((metadata, encoded_data)) = url.split_once(',') else {
        tracing::warn!("Invalid data URI format");
        return None;
    };

    if !metadata.ends_with(";base64") {
        tracing::warn!("Unsupported data URI encoding: {}", metadata);
        return None;
    }

    match base64::engine::general_purpose::STANDARD.decode(encoded_data) {
        Ok(bytes) => Some(bytes),
        Err(err) => {
            tracing::warn!("Failed to decode base64 image: {}", err);
            None
        }
    }
}

fn is_http_url(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

async fn fetch_remote_image(url: &str) -> Option<Vec<u8>> {
    let response = match http_image_client().get(url).send().await {
        Ok(response) => response,
        Err(err) => {
            tracing::warn!("Failed to fetch image URL {}: {}", url, err);
            return None;
        }
    };

    let response = match response.error_for_status() {
        Ok(response) => response,
        Err(err) => {
            tracing::warn!("Image URL returned error status {}: {}", url, err);
            return None;
        }
    };

    match response.bytes().await {
        Ok(bytes) => Some(bytes.to_vec()),
        Err(err) => {
            tracing::warn!("Failed to read image response body {}: {}", url, err);
            None
        }
    }
}

async fn read_local_image(path: &Path) -> Option<Vec<u8>> {
    match tokio::fs::read(path).await {
        Ok(bytes) => Some(bytes),
        Err(err) => {
            tracing::warn!("Failed to read image file {}: {}", path.display(), err);
            None
        }
    }
}

fn http_image_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            // Total deadline for any single fetch.
            .timeout(Duration::from_secs(10))
            // Cap the time spent dialling a hostile or unreachable origin so
            // the request as a whole cannot stall longer than necessary.
            .connect_timeout(Duration::from_secs(5))
            // Bound redirect chains so a malicious origin cannot bounce the
            // client through unbounded hops.
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .expect("server image client should build")
    })
}

#[cfg(test)]
#[path = "media_tests.rs"]
mod tests;
