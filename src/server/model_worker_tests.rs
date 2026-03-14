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

use std::io::Cursor;

use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};

use super::{
    build_generation_result, decode_request_images, merge_config_stop_tokens, safe_emit_boundary,
};
use crate::SamplingConfig;

fn encode_png_bytes() -> Vec<u8> {
    let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(1, 1, Rgb([0, 0, 0])));
    let mut cursor = Cursor::new(Vec::new());
    image.write_to(&mut cursor, ImageFormat::Png).unwrap();
    cursor.into_inner()
}

#[test]
fn merge_config_stop_tokens_appends_only_missing_ids() {
    let sampling = SamplingConfig {
        stop_token_ids: vec![2, 3],
        ..SamplingConfig::greedy()
    };

    let merged = merge_config_stop_tokens(sampling, &[3, 4, 5]);
    assert_eq!(merged.stop_token_ids, vec![2, 3, 4, 5]);
}

#[test]
fn decode_request_images_keeps_valid_images_and_rejects_all_invalid_input() {
    let decoded = decode_request_images(&[encode_png_bytes(), vec![1, 2, 3]]).unwrap();
    assert_eq!(decoded.len(), 1);

    let err = decode_request_images(&[vec![1, 2, 3]]).unwrap_err();
    assert!(err.to_string().contains("Failed to decode any images"));
}

#[test]
fn build_generation_result_computes_finish_reason_and_generation_split() {
    let stop = build_generation_result("ok".to_string(), 10, 3, 120, 40, 8);
    assert_eq!(stop.finish_reason, "stop");
    assert_eq!(stop.generation_only_ms, 80);

    let length = build_generation_result("ok".to_string(), 10, 8, 50, 60, 8);
    assert_eq!(length.finish_reason, "length");
    assert_eq!(length.generation_only_ms, 0);
}

#[test]
fn safe_emit_boundary_stops_before_trailing_replacement_chars() {
    // ASCII string: boundary is at the end.
    assert_eq!(safe_emit_boundary("hello"), 5);

    // Replacement char at end: boundary stops before it.
    let with_replacement = "ok\u{FFFD}";
    let expected = "ok".len();
    assert_eq!(safe_emit_boundary(with_replacement), expected);

    // All replacement chars: boundary is 0.
    assert_eq!(safe_emit_boundary("\u{FFFD}\u{FFFD}"), 0);

    // Empty string: boundary is 0.
    assert_eq!(safe_emit_boundary(""), 0);

    // Multi-byte character followed by replacement char.
    let mixed = "\u{AC00}\u{FFFD}"; // Korean syllable + replacement
    assert_eq!(safe_emit_boundary(mixed), "\u{AC00}".len()); // 3 bytes
}
