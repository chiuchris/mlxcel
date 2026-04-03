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

//! Gemma4 image preprocessor.
//!
//! Used by: Gemma4 VLM

use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Gemma4ImageInput {
    pub pixel_values: UniquePtr<MlxArray>, // [1, 3, H, W]
    pub patch_grid: (usize, usize),        // (patch_h, patch_w)
    pub num_soft_tokens: usize,
}

pub struct Gemma4Processor {
    pub patch_size: usize,
    pub max_soft_tokens: usize,
    pub pooling_kernel_size: usize,
    pub rescale_factor: f32,
}

impl Gemma4Processor {
    pub fn new(patch_size: usize, max_soft_tokens: usize, pooling_kernel_size: usize) -> Self {
        Self {
            patch_size,
            max_soft_tokens,
            pooling_kernel_size,
            rescale_factor: 1.0 / 255.0,
        }
    }

    pub fn preprocess(&self, images: &[DynamicImage]) -> Vec<Gemma4ImageInput> {
        images
            .iter()
            .map(|image| self.preprocess_single(image))
            .collect()
    }

    fn preprocess_single(&self, image: &DynamicImage) -> Gemma4ImageInput {
        let rgb = image.to_rgb8();
        let (target_h, target_w) =
            self.aspect_ratio_preserving_resize_dims(rgb.height() as usize, rgb.width() as usize);

        let resized = if rgb.height() as usize == target_h && rgb.width() as usize == target_w {
            rgb
        } else {
            DynamicImage::ImageRgb8(rgb)
                .resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom)
                .to_rgb8()
        };

        let channels = 3usize;
        let mut data = vec![0.0f32; channels * target_h * target_w];
        for y in 0..target_h {
            for x in 0..target_w {
                let pixel = resized.get_pixel(x as u32, y as u32);
                for c in 0..channels {
                    let dst = c * target_h * target_w + y * target_w + x;
                    data[dst] = pixel[c] as f32 * self.rescale_factor;
                }
            }
        }

        let patch_h = target_h / self.patch_size;
        let patch_w = target_w / self.patch_size;
        let num_soft_tokens = (patch_h * patch_w) / self.pooling_kernel_size.pow(2);

        Gemma4ImageInput {
            pixel_values: mlxcel_core::from_slice_f32(
                &data,
                &[1, channels as i32, target_h as i32, target_w as i32],
            ),
            patch_grid: (patch_h, patch_w),
            num_soft_tokens,
        }
    }

    fn aspect_ratio_preserving_resize_dims(
        &self,
        image_height: usize,
        image_width: usize,
    ) -> (usize, usize) {
        let max_patches = self.max_soft_tokens * self.pooling_kernel_size.pow(2);
        let target_px = max_patches as f64 * (self.patch_size * self.patch_size) as f64;
        let factor = (target_px / ((image_height * image_width).max(1) as f64)).sqrt();
        let side_mult = self.pooling_kernel_size * self.patch_size;

        let mut target_h = ((factor * image_height as f64 / side_mult as f64).floor() as usize)
            * side_mult;
        let mut target_w = ((factor * image_width as f64 / side_mult as f64).floor() as usize)
            * side_mult;

        let max_side_length =
            (max_patches / self.pooling_kernel_size.pow(2)).max(1) * side_mult;

        if target_h == 0 && target_w == 0 {
            target_h = side_mult;
            target_w = side_mult;
        } else if target_h == 0 {
            target_h = side_mult;
            target_w = ((image_width / image_height.max(1)).max(1) * side_mult).min(max_side_length);
        } else if target_w == 0 {
            target_w = side_mult;
            target_h = ((image_height / image_width.max(1)).max(1) * side_mult).min(max_side_length);
        }

        (target_h, target_w)
    }
}
