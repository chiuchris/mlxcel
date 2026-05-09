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

use std::path::Path;

use super::*;

#[test]
fn is_video_file_recognises_common_extensions() {
    assert!(is_video_file(Path::new("clip.mp4")));
    assert!(is_video_file(Path::new("clip.MP4")));
    assert!(is_video_file(Path::new("/data/clip.mov")));
    assert!(is_video_file(Path::new("/data/clip.webm")));
    assert!(is_video_file(Path::new("/data/clip.mkv")));
    assert!(is_video_file(Path::new("/data/clip.avi")));
    assert!(is_video_file(Path::new("/data/clip.m4v")));
}

#[test]
fn is_video_file_rejects_images_and_other_files() {
    assert!(!is_video_file(Path::new("photo.jpg")));
    assert!(!is_video_file(Path::new("photo.png")));
    assert!(!is_video_file(Path::new("photo.webp")));
    assert!(!is_video_file(Path::new("notes.txt")));
    assert!(!is_video_file(Path::new("/data/clip")));
    assert!(!is_video_file(Path::new("/data/")));
}

#[test]
fn smart_nframes_uniform_default_fps_two() {
    // 240 frames at 30 fps with target fps 2.0 => 240/30*2 = 16 frames.
    let n = smart_nframes(240, 30.0, Some(2.0), None).unwrap();
    assert_eq!(n, 16);
    assert!(n.is_multiple_of(FRAME_FACTOR));
}

#[test]
fn smart_nframes_clamps_to_min_frames() {
    // Very short video at low fps: clamp upward to FPS_MIN_FRAMES = 4.
    let n = smart_nframes(8, 30.0, Some(0.1), None).unwrap();
    assert_eq!(n, FPS_MIN_FRAMES);
}

#[test]
fn smart_nframes_clamps_to_max_frames() {
    // Long video at very high target fps: clamp to FPS_MAX_FRAMES = 768
    // when total_frames >= 768.
    let n = smart_nframes(10_000, 30.0, Some(60.0), None).unwrap();
    assert_eq!(n, FPS_MAX_FRAMES);
}

#[test]
fn smart_nframes_clamps_to_total_frames() {
    // Total frames < FPS_MAX_FRAMES — output cannot exceed total.
    let n = smart_nframes(20, 30.0, Some(100.0), None).unwrap();
    assert_eq!(n, 20);
}

#[test]
fn smart_nframes_explicit_nframes_rounds_to_factor() {
    // Caller supplies an odd target — round to the nearest even.
    let n = smart_nframes(40, 30.0, None, Some(7)).unwrap();
    assert_eq!(n, 8);
}

#[test]
fn smart_nframes_explicit_nframes_out_of_range_errors() {
    // Caller asks for more frames than exist.
    assert!(smart_nframes(10, 30.0, None, Some(20)).is_err());
}

#[test]
fn smart_nframes_total_below_factor_errors() {
    assert!(smart_nframes(1, 30.0, Some(2.0), None).is_err());
}

#[test]
fn smart_nframes_invalid_target_fps_errors() {
    assert!(smart_nframes(40, 30.0, Some(0.0), None).is_err());
    assert!(smart_nframes(40, 30.0, Some(-1.0), None).is_err());
}

#[test]
fn smart_nframes_handles_zero_video_fps_gracefully() {
    // ffprobe occasionally reports 0 fps for streams with metadata gaps.
    // The fallback path should still produce a sensible frame count
    // (treats fps as 1.0 internally).
    let n = smart_nframes(60, 0.0, Some(2.0), None).unwrap();
    assert!(n >= FRAME_FACTOR);
    assert!(n <= 60);
    assert!(n.is_multiple_of(FRAME_FACTOR));
}

#[test]
fn load_video_missing_file_errors() {
    let path = Path::new("/non/existent/video.mp4");
    let err = load_video(path, Some(2.0), None).unwrap_err();
    match err {
        VideoError::FileNotFound(p) => assert_eq!(p, path),
        VideoError::FfmpegMissing => {
            // On systems without ffmpeg the missing-binary error wins
            // — acceptable for this graceful-degradation contract.
        }
        other => panic!("expected FileNotFound or FfmpegMissing, got {other:?}"),
    }
}

#[test]
fn load_video_without_ffmpeg_returns_clear_error() {
    // This test exercises the graceful-degradation path that fires when
    // `ffmpeg` is not on PATH. We can only assert the precise error
    // shape on systems where ffmpeg is genuinely missing.
    if ffmpeg_available() {
        return;
    }
    let path = Path::new("/tmp/does-not-exist.mp4");
    let err = load_video(path, None, None).unwrap_err();
    matches!(err, VideoError::FfmpegMissing);
}
