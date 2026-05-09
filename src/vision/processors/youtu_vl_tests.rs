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

use super::*;
use image::{DynamicImage, RgbImage};

fn solid_image(h: u32, w: u32, rgb: [u8; 3]) -> DynamicImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.put_pixel(x, y, image::Rgb(rgb));
        }
    }
    DynamicImage::ImageRgb8(img)
}

fn synthetic_processor() -> YoutuVLProcessor {
    // patch_size=16, spatial_merge_size=2 → factor = 32.
    // Use very tight pixel bounds so a small synthetic image survives
    // smart_resize untouched.
    YoutuVLProcessor::new(16, 2)
        .with_pixel_bounds(32 * 32, 256 * 256)
        .with_norm([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
}

#[test]
fn smart_resize_aligns_to_patch_merge_factor() {
    let p = synthetic_processor();

    let cases = vec![
        // Inputs that are already multiples of 32 should pass through as-is.
        (64, 64, 64, 64),
        // Inputs slightly off should round to the nearest multiple.
        (60, 100, 64, 96),
        // Tiny inputs must be lifted to satisfy the min_pixels lower bound.
        (16, 16, 32, 32),
    ];
    for (h, w, exp_h, exp_w) in cases {
        let (rh, rw) = p.smart_resize(h, w);
        assert!(
            rh % 32 == 0 && rw % 32 == 0,
            "smart_resize output ({}, {}) not aligned to 32 for input ({}, {})",
            rh,
            rw,
            h,
            w
        );
        assert_eq!(rh, exp_h, "h mismatch for input ({h}, {w})");
        assert_eq!(rw, exp_w, "w mismatch for input ({h}, {w})");
    }
}

#[test]
fn preprocess_emits_expected_patch_shape() {
    let p = synthetic_processor();
    let img = solid_image(64, 96, [128, 128, 128]);
    let (pixel_values, spatial_shapes) = p.preprocess_with_spatial(&[img]);

    // 64x96 image at patch_size=16 → 4x6 = 24 patches.
    assert_eq!(spatial_shapes, vec![(4, 6)]);

    let shape = mlxcel_core::array_shape(&pixel_values);
    let expected_patches = 4 * 6;
    let expected_features = 16 * 16 * 3;
    assert_eq!(shape, vec![expected_patches, expected_features]);
}

#[test]
fn preprocess_concatenates_multi_image_batches() {
    let p = synthetic_processor();
    let img_a = solid_image(64, 64, [10, 20, 30]);
    let img_b = solid_image(32, 64, [200, 100, 50]);
    let (pixel_values, spatial_shapes) = p.preprocess_with_spatial(&[img_a, img_b]);

    // 64x64 → (4, 4) = 16 patches; 32x64 → (2, 4) = 8 patches.
    assert_eq!(spatial_shapes, vec![(4, 4), (2, 4)]);

    let total_patches = 16 + 8;
    let shape = mlxcel_core::array_shape(&pixel_values);
    assert_eq!(shape, vec![total_patches, 16 * 16 * 3]);
}

#[test]
fn normalization_matches_siglip_default() {
    let p = synthetic_processor();
    // A pure mid-gray image should normalize close to zero (val - 0.5)/0.5.
    let img = solid_image(64, 64, [128, 128, 128]);
    let (pixel_values, _) = p.preprocess_with_spatial(&[img]);
    mlxcel_core::eval(&pixel_values);

    let max_abs = mlxcel_core::max_all(&mlxcel_core::abs(&pixel_values));
    mlxcel_core::eval(&max_abs);
    // 128/255 ≈ 0.502; (0.502 - 0.5)/0.5 ≈ 0.0039
    assert!(
        mlxcel_core::item_f32(&max_abs) < 0.02,
        "mid-gray normalized values should sit close to 0; saw {}",
        mlxcel_core::item_f32(&max_abs)
    );
}
