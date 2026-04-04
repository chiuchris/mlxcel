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
use std::{path::Path, sync::OnceLock, time::Duration};

use super::types::ChatCompletionRequest;
use super::types::request::InputAudio;

pub(crate) async fn extract_chat_image_data(request: &ChatCompletionRequest) -> Vec<Vec<u8>> {
    collect_image_data(request.image_urls()).await
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
            .timeout(Duration::from_secs(10))
            .build()
            .expect("server image client should build")
    })
}

#[cfg(test)]
#[path = "media_tests.rs"]
mod tests;
