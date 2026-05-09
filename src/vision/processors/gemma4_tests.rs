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

//! Unit tests for the Gemma 4 image / video preprocessor (issue #553).
//!
//! These tests use synthetic in-memory frames so they run on any
//! platform without an actual video file or `ffmpeg` install.

use image::{DynamicImage, RgbImage};

use super::*;

fn synthetic_rgb(width: u32, height: u32, fill: u8) -> DynamicImage {
    let img = RgbImage::from_pixel(width, height, image::Rgb([fill, fill, fill]));
    DynamicImage::ImageRgb8(img)
}

fn frames(count: usize, width: u32, height: u32) -> Vec<DynamicImage> {
    (0..count)
        .map(|i| synthetic_rgb(width, height, (i * 7 % 200) as u8))
        .collect()
}

#[test]
fn process_videos_emits_one_features_block_per_video() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let videos = vec![frames(8, 320, 240), frames(12, 256, 192)];
    let features = processor.process_videos(&videos, None);
    assert_eq!(features.len(), 2);
    assert_eq!(features[0].num_frames(), 8);
    assert_eq!(features[1].num_frames(), 12);
}

#[test]
fn process_videos_clamps_long_inputs_to_video_num_frames() {
    let mut processor = Gemma4Processor::new(16, 280, 3);
    processor.set_video_config(70, 4).unwrap();
    let videos = vec![frames(20, 224, 224)];
    let features = processor.process_videos(&videos, None);
    assert_eq!(features.len(), 1);
    assert_eq!(
        features[0].num_frames(),
        4,
        "downsampled to video_num_frames"
    );
    assert_eq!(features[0].frame_timestamps.len(), 4);
}

#[test]
fn process_videos_pixel_values_have_expected_shape() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let videos = vec![frames(2, 192, 192)];
    let features = processor.process_videos(&videos, None);
    let first = &features[0].frames[0];
    let shape = mlxcel_core::array_shape(first.pixel_values.as_ref().unwrap());
    assert_eq!(shape.len(), 4, "pixel_values must be (1, C, H, W)");
    assert_eq!(shape[0], 1, "batch axis");
    assert_eq!(shape[1], 3, "channel axis is 3 (RGB)");
    let side_mult = (processor.pooling_kernel_size * processor.patch_size) as i32;
    assert!(
        shape[2] % side_mult == 0,
        "H must be divisible by side_mult"
    );
    assert!(
        shape[3] % side_mult == 0,
        "W must be divisible by side_mult"
    );
}

#[test]
fn process_videos_all_frames_share_video_dimensions() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let videos = vec![frames(6, 320, 240)];
    let features = processor.process_videos(&videos, None);
    let mut shapes = features[0]
        .frames
        .iter()
        .map(|f| mlxcel_core::array_shape(f.pixel_values.as_ref().unwrap()));
    let first = shapes.next().unwrap();
    for shape in shapes {
        assert_eq!(
            shape, first,
            "every frame in the same video must share H and W"
        );
    }
}

#[test]
fn process_videos_num_soft_tokens_within_budget() {
    let mut processor = Gemma4Processor::new(16, 280, 3);
    processor.set_video_config(70, 32).unwrap();
    let videos = vec![frames(2, 384, 288)];
    let features = processor.process_videos(&videos, None);
    let budget = processor.video_max_soft_tokens;
    assert!(
        features[0].num_soft_tokens_per_frame <= budget,
        "soft tokens per frame {} exceeds budget {}",
        features[0].num_soft_tokens_per_frame,
        budget
    );
    assert!(
        features[0].num_soft_tokens_per_frame > 0,
        "non-empty frames must produce at least one soft token"
    );
}

#[test]
fn process_videos_per_video_fps_drives_timestamps() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let videos = vec![frames(4, 224, 224), frames(4, 224, 224)];
    let features = processor.process_videos(&videos, Some(&[2.0, 1.0]));

    // First video sampled at 2 fps: t = [0, 0.5, 1.0, 1.5]
    assert!((features[0].frame_timestamps[1] - 0.5).abs() < 1e-6);
    // Second video sampled at 1 fps: t = [0, 1, 2, 3]
    assert!((features[1].frame_timestamps[1] - 1.0).abs() < 1e-6);
}

#[test]
fn process_videos_invalid_fps_falls_back_to_default() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let videos = vec![frames(2, 224, 224)];
    let features = processor.process_videos(&videos, Some(&[0.0]));
    // Default fps = 2.0 so timestamp[1] should be 0.5.
    assert!((features[0].frame_timestamps[1] - 0.5).abs() < 1e-6);
}

#[test]
fn process_videos_handles_empty_input() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let features = processor.process_videos(&[], None);
    assert!(features.is_empty());
}

#[test]
fn process_videos_handles_single_frame_video() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let videos = vec![frames(1, 224, 224)];
    let features = processor.process_videos(&videos, None);
    assert_eq!(features.len(), 1);
    assert_eq!(features[0].num_frames(), 1);
    assert_eq!(features[0].frame_timestamps, vec![0.0]);
}

#[test]
fn set_video_config_rejects_unsupported_soft_tokens() {
    let mut processor = Gemma4Processor::new(16, 280, 3);
    assert!(processor.set_video_config(99, 32).is_err());
    // Sanity: supported values pass.
    for &n in SUPPORTED_VIDEO_SOFT_TOKENS {
        let mut p = Gemma4Processor::new(16, 280, 3);
        assert!(p.set_video_config(n, 32).is_ok());
    }
}

#[test]
fn process_videos_existing_image_path_unchanged() {
    // Sanity: adding the video path must not alter the image path.
    // We confirm Gemma4ImageInput shapes match what Gemma 4 image
    // wiring already expects.
    let processor = Gemma4Processor::new(16, 280, 3);
    let images = vec![synthetic_rgb(224, 224, 128)];
    let processed = processor.preprocess(&images);
    assert_eq!(processed.len(), 1);
    assert!(processed[0].num_soft_tokens > 0);
    let shape = mlxcel_core::array_shape(processed[0].pixel_values.as_ref().unwrap());
    assert_eq!(shape.len(), 4);
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], 3);
}
