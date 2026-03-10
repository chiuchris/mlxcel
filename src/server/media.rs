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
//! request formats without growing individual route handlers.

use base64::Engine;

use super::types::ChatCompletionRequest;

pub(crate) fn extract_chat_image_data(request: &ChatCompletionRequest) -> Vec<Vec<u8>> {
    collect_image_data(request.image_urls())
}

pub(crate) fn collect_image_data<I, S>(urls: I) -> Vec<Vec<u8>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    urls.into_iter()
        .filter_map(|url| read_image_url(url.as_ref()))
        .collect()
}

pub(crate) fn read_image_url(url: &str) -> Option<Vec<u8>> {
    if url.starts_with("data:") {
        return decode_data_uri(url);
    }

    if let Some(path) = url.strip_prefix("file://") {
        return read_local_image(path);
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

fn read_local_image(path: &str) -> Option<Vec<u8>> {
    match std::fs::read(path) {
        Ok(bytes) => Some(bytes),
        Err(err) => {
            tracing::warn!("Failed to read image file {}: {}", path, err);
            None
        }
    }
}

#[cfg(test)]
#[path = "media_tests.rs"]
mod tests;
