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

//! Qwen2-VL Image Processor
//!
//! Handles dynamic resolution image preprocessing:
//! 1. Resize preserving aspect ratio within constraints
//! 2. Pad to multiples of patch_size * spatial_merge_size (= 28)
//! 3. Duplicate frame for temporal_patch_size (single image -> 2 frames)
//! 4. Flatten to patch format for vision encoder
//!
//! Used by: Qwen2-VL

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Qwen2VLProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Qwen2VLProcessor {
    /// Used by: Qwen2-VL, Qwen2.5-VL (CLIP normalization)
    pub fn new(patch_size: usize, temporal_patch_size: usize, spatial_merge_size: usize) -> Self {
        Self {
            patch_size,
            temporal_patch_size,
            spatial_merge_size,
            min_pixels: 4 * 28 * 28,     // 3136
            max_pixels: 16384 * 28 * 28, // large limit
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.261_302_6, 0.275_777_1],
        }
    }

    /// Used by: Qwen3-VL (simple 0.5/0.5 normalization)
    pub fn new_with_norm(
        patch_size: usize,
        temporal_patch_size: usize,
        spatial_merge_size: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Self {
        Self {
            patch_size,
            temporal_patch_size,
            spatial_merge_size,
            min_pixels: 4 * 28 * 28,
            max_pixels: 16384 * 28 * 28,
            mean,
            std,
        }
    }

    /// Compute target size that satisfies constraints
    /// Returns (height, width) padded to multiples of factor
    fn smart_resize(&self, orig_h: u32, orig_w: u32) -> (u32, u32) {
        let factor = (self.patch_size * self.spatial_merge_size) as u32; // 28

        // Start with original size, round to factor
        let mut h = ((orig_h as f64 / factor as f64).round() as u32).max(1) * factor;
        let mut w = ((orig_w as f64 / factor as f64).round() as u32).max(1) * factor;

        // Ensure within pixel limits
        let pixels = (h * w) as usize;
        if pixels > self.max_pixels {
            let scale = (self.max_pixels as f64 / pixels as f64).sqrt();
            h = ((h as f64 * scale / factor as f64).round() as u32).max(1) * factor;
            w = ((w as f64 * scale / factor as f64).round() as u32).max(1) * factor;
        }
        if (h * w) as usize > self.max_pixels {
            // Further reduce if needed
            let scale = (self.max_pixels as f64 / (h * w) as f64).sqrt();
            h = ((h as f64 * scale / factor as f64).floor() as u32).max(1) * factor;
            w = ((w as f64 * scale / factor as f64).floor() as u32).max(1) * factor;
        }
        let pixels = (h * w) as usize;
        if pixels < self.min_pixels {
            let scale = (self.min_pixels as f64 / pixels as f64).sqrt();
            h = ((h as f64 * scale / factor as f64).ceil() as u32).max(1) * factor;
            w = ((w as f64 * scale / factor as f64).ceil() as u32).max(1) * factor;
        }

        (h, w)
    }

    /// Compute grid_thw for a set of images
    /// Returns Vec of (temporal, h_patches, w_patches)
    pub fn compute_grid_thw(&self, images: &[image::DynamicImage]) -> Vec<(i32, i32, i32)> {
        images
            .iter()
            .map(|img| {
                let (h, w) = self.smart_resize(img.height(), img.width());
                let h_patches = h as i32 / self.patch_size as i32;
                let w_patches = w as i32 / self.patch_size as i32;
                (1i32, h_patches, w_patches) // temporal=1 for single images
            })
            .collect()
    }

    /// Preprocess images and return (pixel_values, grid_thw)
    pub fn preprocess_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32, i32)>) {
        let grid_thw = self.compute_grid_thw(images);
        let mut all_patches: Vec<f32> = Vec::new();
        let in_channels = 3usize;
        let patch_area = self.patch_size * self.patch_size;
        let features_per_pixel = in_channels * patch_area;

        for (img_idx, img) in images.iter().enumerate() {
            let (t, h_patches, w_patches) = grid_thw[img_idx];
            let target_h = h_patches as u32 * self.patch_size as u32;
            let target_w = w_patches as u32 * self.patch_size as u32;

            // Resize image
            let resized = img.resize_exact(target_w, target_h, FilterType::Lanczos3);
            let rgb = resized.to_rgb8();

            // Normalize: (pixel / 255.0 - mean) / std
            let h = target_h as usize;
            let w = target_w as usize;
            let mut normalized = vec![0f32; in_channels * h * w];
            for y in 0..h {
                for x in 0..w {
                    let pixel = rgb.get_pixel(x as u32, y as u32);
                    for c in 0..3 {
                        let val = pixel[c] as f32 / 255.0;
                        let norm_val = (val - self.mean[c]) / self.std[c];
                        normalized[c * h * w + y * w + x] = norm_val;
                    }
                }
            }

            // Convert to patch format:
            // For each spatial patch, extract [C, patch_size, patch_size] -> flatten to [C*P*P]
            // Then duplicate for temporal_patch_size (2 frames for single image)
            let total_patches = (h_patches * w_patches) as usize;
            let temporal = t as usize;

            for _t in 0..temporal {
                for patch_idx in 0..total_patches {
                    let py = patch_idx / w_patches as usize;
                    let px = patch_idx % w_patches as usize;
                    let y_start = py * self.patch_size;
                    let x_start = px * self.patch_size;

                    // For temporal_patch_size=2, we output 2 rows per spatial patch
                    for _tp in 0..self.temporal_patch_size {
                        let mut patch_data = Vec::with_capacity(features_per_pixel);
                        for c in 0..in_channels {
                            for dy in 0..self.patch_size {
                                for dx in 0..self.patch_size {
                                    let y = y_start + dy;
                                    let x = x_start + dx;
                                    patch_data.push(normalized[c * h * w + y * w + x]);
                                }
                            }
                        }
                        all_patches.extend_from_slice(&patch_data);
                    }
                }
            }
        }

        // Total rows: sum over all images of (t * h_patches * w_patches * temporal_patch_size)
        let total_rows: usize = grid_thw
            .iter()
            .map(|&(t, h, w)| (t as usize) * (h as usize) * (w as usize) * self.temporal_patch_size)
            .sum();

        let pixel_values = mlxcel_core::from_slice_f32(
            &all_patches,
            &[total_rows as i32, features_per_pixel as i32],
        );

        (pixel_values, grid_thw)
    }
}

impl ImageProcessor for Qwen2VLProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_grid(images);
        pixel_values
    }
}
