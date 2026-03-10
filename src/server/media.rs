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

pub(crate) async fn extract_chat_image_data(request: &ChatCompletionRequest) -> Vec<Vec<u8>> {
    collect_image_data(request.image_urls()).await
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
