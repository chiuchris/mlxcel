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

//! Youtu-VL image processor.
//!
//! Mirrors the contract that upstream `references/mlx-vlm/mlx_vlm/models/
//! youtu_vl/vision.py::VisionModel.__call__` expects:
//! - `pixel_values`: `[total_patches, patch_size**2 * channels]` flattened
//!   patches, normalized with SigLIP statistics.
//! - `spatial_shapes`: `[N, 2]` of `(h_patches, w_patches)` per image (no
//!   temporal dimension — Youtu-VL is image-only at this scope).
//!
//! Why this is not the standard SigLIP processor: that processor emits
//! `[B, C, H, W]` because the existing SigLIP encoder uses a `Conv2d` patch
//! embedding. Youtu-VL uses a `Linear` patch embedding over flattened
//! patches and feeds them with `spatial_shapes` to a windowed-attention
//! tower, so the channel layout is fundamentally different. We therefore
//! own a small purpose-built processor instead of forcing reuse with
//! conditional code paths.
//!
//! Why this is not the Qwen2-VL processor either: Qwen2-VL duplicates each
//! spatial patch `temporal_patch_size` times and prepends a temporal axis to
//! mimic image-as-video; Youtu-VL's vision tower does not consume a
//! temporal index, so we emit one row per spatial patch and skip the
//! duplicate.
//!
//! Used by: `vision::youtu_vl::YoutuVLModel`.

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct YoutuVLProcessor {
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    /// Pre-pixel-area lower bound (in pixels). Falls back to upstream's
    /// SigLIP2 default if the HF processor json does not specify one.
    pub min_pixels: usize,
    /// Pre-pixel-area upper bound (in pixels). Same fallback rationale.
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl YoutuVLProcessor {
    /// SigLIP2 default normalization (mean=std=0.5 across all channels).
    pub fn new(patch_size: usize, spatial_merge_size: usize) -> Self {
        Self {
            patch_size,
            spatial_merge_size,
            // Match the SigLIP2 / Qwen2.5-VL defaults so a model whose
            // preprocessor.json was missing these keys still produces a
            // sensible patch grid.
            min_pixels: 4 * 28 * 28,
            max_pixels: 16384 * 28 * 28,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }

    pub fn with_norm(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.mean = mean;
        self.std = std;
        self
    }

    pub fn with_pixel_bounds(mut self, min_pixels: usize, max_pixels: usize) -> Self {
        self.min_pixels = min_pixels;
        self.max_pixels = max_pixels;
        self
    }

    /// Compute target (h, w) padded to multiples of `patch_size *
    /// spatial_merge_size` so the resulting patch grid is divisible by the
    /// spatial-merge factor used inside the encoder.
    fn smart_resize(&self, orig_h: u32, orig_w: u32) -> (u32, u32) {
        let factor = (self.patch_size * self.spatial_merge_size) as u32;

        let mut h = ((orig_h as f64 / factor as f64).round() as u32).max(1) * factor;
        let mut w = ((orig_w as f64 / factor as f64).round() as u32).max(1) * factor;

        let pixels = (h as usize) * (w as usize);
        if pixels > self.max_pixels {
            let scale = (self.max_pixels as f64 / pixels as f64).sqrt();
            h = ((h as f64 * scale / factor as f64).round() as u32).max(1) * factor;
            w = ((w as f64 * scale / factor as f64).round() as u32).max(1) * factor;
        }
        if (h as usize) * (w as usize) > self.max_pixels {
            let scale = (self.max_pixels as f64 / ((h as usize) * (w as usize)) as f64).sqrt();
            h = ((h as f64 * scale / factor as f64).floor() as u32).max(1) * factor;
            w = ((w as f64 * scale / factor as f64).floor() as u32).max(1) * factor;
        }
        let pixels = (h as usize) * (w as usize);
        if pixels < self.min_pixels {
            let scale = (self.min_pixels as f64 / pixels as f64).sqrt();
            h = ((h as f64 * scale / factor as f64).ceil() as u32).max(1) * factor;
            w = ((w as f64 * scale / factor as f64).ceil() as u32).max(1) * factor;
        }
        (h, w)
    }

    /// Compute `(h_patches, w_patches)` per image after resizing.
    pub fn compute_spatial_shapes(&self, images: &[image::DynamicImage]) -> Vec<(i32, i32)> {
        images
            .iter()
            .map(|img| {
                let (h, w) = self.smart_resize(img.height(), img.width());
                let h_patches = (h / self.patch_size as u32) as i32;
                let w_patches = (w / self.patch_size as u32) as i32;
                (h_patches, w_patches)
            })
            .collect()
    }

    /// Preprocess the input images and return `(pixel_values, spatial_shapes)`
    /// in the layout expected by `YoutuVLVisionEncoder::forward_with_spatial`.
    pub fn preprocess_with_spatial(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32)>) {
        let spatial_shapes = self.compute_spatial_shapes(images);

        let in_channels = 3usize;
        let patch_area = self.patch_size * self.patch_size;
        let features_per_patch = in_channels * patch_area;

        let total_patches: usize = spatial_shapes
            .iter()
            .map(|&(h, w)| (h as usize) * (w as usize))
            .sum();

        let mut all_patches = vec![0f32; total_patches * features_per_patch];
        let mut write_offset: usize = 0;

        for (img_idx, img) in images.iter().enumerate() {
            let (h_patches, w_patches) = spatial_shapes[img_idx];
            let target_h = (h_patches as u32) * (self.patch_size as u32);
            let target_w = (w_patches as u32) * (self.patch_size as u32);

            let resized = img.resize_exact(target_w, target_h, FilterType::Lanczos3);
            let rgb = resized.to_rgb8();

            let h = target_h as usize;
            let w = target_w as usize;
            let mut normalized = vec![0f32; in_channels * h * w];
            for y in 0..h {
                for x in 0..w {
                    let pixel = rgb.get_pixel(x as u32, y as u32);
                    for c in 0..in_channels {
                        let val = pixel[c] as f32 / 255.0;
                        let normed = (val - self.mean[c]) / self.std[c];
                        normalized[c * h * w + y * w + x] = normed;
                    }
                }
            }

            // Emit one row per spatial patch in the layout
            // `[h_patches * w_patches, channels * patch_size * patch_size]`,
            // with the inner ordering `(c, dy, dx)` to match how upstream
            // unfolds patches via `unfold` over the (C, H, W) image tensor.
            let total_patches_img = (h_patches as usize) * (w_patches as usize);
            for patch_idx in 0..total_patches_img {
                let py = patch_idx / w_patches as usize;
                let px = patch_idx % w_patches as usize;
                let y_start = py * self.patch_size;
                let x_start = px * self.patch_size;

                let row_start = (write_offset + patch_idx) * features_per_patch;
                let mut k = 0usize;
                for c in 0..in_channels {
                    for dy in 0..self.patch_size {
                        for dx in 0..self.patch_size {
                            let y = y_start + dy;
                            let x = x_start + dx;
                            all_patches[row_start + k] = normalized[c * h * w + y * w + x];
                            k += 1;
                        }
                    }
                }
            }

            write_offset += total_patches_img;
        }

        let pixel_values = mlxcel_core::from_slice_f32(
            &all_patches,
            &[total_patches as i32, features_per_patch as i32],
        );

        (pixel_values, spatial_shapes)
    }
}

impl ImageProcessor for YoutuVLProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_spatial(images);
        pixel_values
    }
}

#[cfg(test)]
#[path = "youtu_vl_tests.rs"]
mod tests;
