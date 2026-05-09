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

//! Generic VLM video utilities (issue #553).
//!
//! Translates the upstream Python `mlx_vlm/video_generate.py` pipeline into
//! Rust so that any vision-language model can accept `--video` inputs through
//! the same code path used by `--image`. The module is intentionally
//! decoder-agnostic at the Rust level: frame extraction is delegated to the
//! `ffmpeg` system binary via a subprocess. We chose subprocess over the
//! `ffmpeg-next` Rust binding because:
//!
//! - `ffmpeg-next` requires `libavcodec` / `libavformat` dev headers at build
//!   time, which complicates Linux + CUDA CI containers and introduces a
//!   non-trivial system dependency.
//! - The runtime binary `ffmpeg` ships on every macOS Apple Silicon
//!   developer machine via Homebrew and on every standard Linux GPU image,
//!   so the deployment surface is already covered.
//! - We only need uniform frame sampling (no per-codec hot path), so the
//!   per-call subprocess fork cost is negligible compared to the model
//!   forward pass.
//!
//! The module surfaces three primitives:
//!
//! - [`smart_nframes`] — uniform-sample target frame count per upstream
//!   [`smart_nframes`](https://github.com/Blaizzy/mlx-vlm) heuristics (FPS
//!   in × target FPS, clamped to `[FPS_MIN_FRAMES, FPS_MAX_FRAMES]`,
//!   rounded down to a multiple of `FRAME_FACTOR = 2`).
//! - [`load_video`] — decode a video file, sample frames, return a
//!   `Vec<DynamicImage>` ready for the existing image-tower preprocess.
//! - [`load_videos`] — multi-video convenience wrapper.
//! - [`is_video_file`] — extension-based detection (mp4/mov/webm/mkv/...).
//!
//! When `ffmpeg` is not present on the runtime PATH, [`load_video`] returns
//! a clear error so callers can degrade gracefully or surface a
//! configuration message to the user. Tests that exercise the subprocess
//! path are gated by [`ffmpeg_available`] and skip cleanly on machines
//! without `ffmpeg`.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use image::DynamicImage;

/// Default FPS target when callers do not specify `--fps`. Mirrors the
/// upstream Python `FPS = 2.0` constant.
pub const DEFAULT_FPS: f64 = 2.0;

/// Frame count must be an even multiple — vision towers downstream pair
/// frames. Mirrors upstream `FRAME_FACTOR = 2`.
pub const FRAME_FACTOR: usize = 2;

/// Lower bound on sampled frame count (mirrors upstream `FPS_MIN_FRAMES`).
pub const FPS_MIN_FRAMES: usize = 4;

/// Upper bound on sampled frame count (mirrors upstream `FPS_MAX_FRAMES`).
pub const FPS_MAX_FRAMES: usize = 768;

/// Recognised video extensions for [`is_video_file`]. Lowercased.
const VIDEO_EXTENSIONS: &[&str] = &[
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg",
];

/// Errors surfaced by the video pipeline.
#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    /// `ffmpeg` is not on PATH. Callers can fall back gracefully.
    #[error("ffmpeg binary not found on PATH; install ffmpeg to enable video inputs")]
    FfmpegMissing,
    /// `ffprobe` is not on PATH (ships with ffmpeg).
    #[error("ffprobe binary not found on PATH; install ffmpeg to enable video inputs")]
    FfprobeMissing,
    /// Video file does not exist on disk.
    #[error("video file not found: {0}")]
    FileNotFound(PathBuf),
    /// `ffprobe` could not parse the video container.
    #[error("ffprobe failed for {path}: {message}")]
    Probe { path: PathBuf, message: String },
    /// `ffmpeg` failed to extract the requested frame.
    #[error("ffmpeg frame extraction failed for {path}: {message}")]
    Extract { path: PathBuf, message: String },
    /// No frames were decoded (corrupt file or empty stream).
    #[error("no frames decoded from video {0}")]
    EmptyVideo(PathBuf),
    /// Generic I/O failure surfacing from an underlying `Command`.
    #[error("video I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Decoded a frame but the bytes did not parse as a valid image.
    #[error("decoded frame from {path} is not a valid image: {source}")]
    Decode {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
}

/// Round `value` to the nearest multiple of `factor`. Mirrors
/// `round_by_factor` from `video_generate.py`.
#[inline]
fn round_by_factor(value: f64, factor: usize) -> usize {
    let factor_f = factor as f64;
    ((value / factor_f).round() as usize) * factor
}

/// Floor `value` to the nearest multiple of `factor` from below. Mirrors
/// `floor_by_factor`.
#[inline]
fn floor_by_factor(value: f64, factor: usize) -> usize {
    let factor_f = factor as f64;
    ((value / factor_f).floor() as usize) * factor
}

/// Ceil `value` to the nearest multiple of `factor` from above. Mirrors
/// `ceil_by_factor`.
#[inline]
fn ceil_by_factor(value: f64, factor: usize) -> usize {
    let factor_f = factor as f64;
    ((value / factor_f).ceil() as usize) * factor
}

/// Return true when `path` has a video file extension.
///
/// Detection is purely extension-based and does not open the file. Used by
/// the CLI to dispatch between image and video preprocessing without a
/// container probe round-trip.
#[must_use]
pub fn is_video_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let lower = name.to_lowercase();
    VIDEO_EXTENSIONS.iter().any(|ext| lower.ends_with(ext))
}

/// Compute the number of frames to sample given the source video's frame
/// count and FPS, with optional caller-supplied target FPS / explicit
/// frame count overrides. Mirrors upstream `smart_nframes`.
///
/// Behavior:
/// - When `target_nframes == Some(n)`, returns `round_by_factor(n,
///   FRAME_FACTOR)` clamped to `[FRAME_FACTOR, total_frames]`.
/// - Otherwise computes `total_frames / video_fps * target_fps`, clamps
///   to `[FPS_MIN_FRAMES, min(FPS_MAX_FRAMES, total_frames)]`, then
///   floors to a multiple of `FRAME_FACTOR`.
///
/// # Errors
/// Returns `Err` when:
/// - `total_frames < FRAME_FACTOR` (can't sample at least 2 frames).
/// - The clamped result still does not fit in `[FRAME_FACTOR, total_frames]`.
pub fn smart_nframes(
    total_frames: usize,
    video_fps: f64,
    target_fps: Option<f64>,
    target_nframes: Option<usize>,
) -> Result<usize, VideoError> {
    if total_frames < FRAME_FACTOR {
        return Err(VideoError::Extract {
            path: PathBuf::new(),
            message: format!(
                "video has only {total_frames} frame(s); need at least {FRAME_FACTOR}"
            ),
        });
    }

    if let Some(n) = target_nframes {
        let nframes = round_by_factor(n as f64, FRAME_FACTOR);
        if nframes < FRAME_FACTOR || nframes > total_frames {
            return Err(VideoError::Extract {
                path: PathBuf::new(),
                message: format!("nframes={nframes} out of range [{FRAME_FACTOR}, {total_frames}]"),
            });
        }
        return Ok(nframes);
    }

    let fps = target_fps.unwrap_or(DEFAULT_FPS);
    if fps <= 0.0 {
        return Err(VideoError::Extract {
            path: PathBuf::new(),
            message: format!("target fps must be > 0; got {fps}"),
        });
    }
    let video_fps_safe = if video_fps > 0.0 { video_fps } else { 1.0 };

    let raw = total_frames as f64 / video_fps_safe * fps;
    let max_cap = ceil_by_factor(FPS_MIN_FRAMES as f64, FRAME_FACTOR).max(FRAME_FACTOR);
    let upper = floor_by_factor(FPS_MAX_FRAMES.min(total_frames) as f64, FRAME_FACTOR);
    let upper = upper.max(max_cap);

    let bounded = raw
        .max(max_cap as f64)
        .min(upper as f64)
        .min(total_frames as f64);
    let nframes = floor_by_factor(bounded, FRAME_FACTOR);
    let nframes = nframes.max(FRAME_FACTOR).min(total_frames);
    if !(FRAME_FACTOR..=total_frames).contains(&nframes) {
        return Err(VideoError::Extract {
            path: PathBuf::new(),
            message: format!("nframes={nframes} out of range [{FRAME_FACTOR}, {total_frames}]"),
        });
    }
    Ok(nframes)
}

/// Return true if `ffmpeg` and `ffprobe` are both invokable. Cached with
/// `OnceLock` so repeated calls do not re-fork.
pub fn ffmpeg_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let ff = Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        let fp = Command::new("ffprobe")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        ff && fp
    })
}

/// Probe a video container with `ffprobe`, returning `(total_frames,
/// average_fps)`. The two values feed [`smart_nframes`].
fn probe_video(path: &Path) -> Result<(usize, f64), VideoError> {
    if !path.exists() {
        return Err(VideoError::FileNotFound(path.to_path_buf()));
    }
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,avg_frame_rate,r_frame_rate,duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                VideoError::FfprobeMissing
            } else {
                VideoError::Io {
                    path: path.to_path_buf(),
                    source: err,
                }
            }
        })?;
    if !output.status.success() {
        return Err(VideoError::Probe {
            path: path.to_path_buf(),
            message: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    let text = String::from_utf8_lossy(&output.stdout);

    let mut nb_frames: Option<usize> = None;
    let mut avg_fps: Option<f64> = None;
    let mut r_fps: Option<f64> = None;
    let mut duration: Option<f64> = None;
    for line in text.lines() {
        let Some((key, raw_value)) = line.split_once('=') else {
            continue;
        };
        let value = raw_value.trim();
        match key.trim() {
            "nb_frames" => {
                if let Ok(n) = value.parse::<usize>() {
                    nb_frames = Some(n);
                }
            }
            "avg_frame_rate" => avg_fps = parse_rational(value),
            "r_frame_rate" => r_fps = parse_rational(value),
            "duration" => {
                if let Ok(d) = value.parse::<f64>() {
                    duration = Some(d);
                }
            }
            _ => {}
        }
    }

    let fps = avg_fps
        .or(r_fps)
        .filter(|f| f.is_finite() && *f > 0.0)
        .unwrap_or(1.0);

    // Some containers return nb_frames="N/A". Fall back to duration*fps.
    let total = if let Some(n) = nb_frames.filter(|n| *n > 0) {
        n
    } else if let Some(d) = duration.filter(|d| d.is_finite() && *d > 0.0) {
        ((d * fps).round() as usize).max(1)
    } else {
        return Err(VideoError::Probe {
            path: path.to_path_buf(),
            message: "could not determine frame count from ffprobe".to_string(),
        });
    };
    Ok((total, fps))
}

/// Parse `"30000/1001"` style rationals returned by ffprobe.
fn parse_rational(value: &str) -> Option<f64> {
    if let Some((num, den)) = value.split_once('/') {
        let num: f64 = num.trim().parse().ok()?;
        let den: f64 = den.trim().parse().ok()?;
        if den.abs() < f64::EPSILON {
            None
        } else {
            Some(num / den)
        }
    } else {
        value.parse().ok()
    }
}

/// Decode a video file and return uniformly-sampled frames as RGB images.
///
/// Sampling matches the upstream Python pipeline:
/// 1. Probe `total_frames` and `video_fps` via `ffprobe`.
/// 2. Compute `nframes = smart_nframes(total_frames, video_fps,
///    target_fps, ..)`.
/// 3. Build `nframes` evenly-spaced indices across `[0, total_frames-1]`.
/// 4. Extract each frame as PNG via a single `ffmpeg` invocation per
///    frame and decode with the existing `image` crate.
///
/// # Errors
/// Returns [`VideoError`] when `ffmpeg`/`ffprobe` is missing, the file
/// does not exist, the container cannot be probed, or no frames can be
/// decoded.
pub fn load_video(
    path: &Path,
    target_fps: Option<f64>,
    target_nframes: Option<usize>,
) -> Result<Vec<DynamicImage>, VideoError> {
    if !ffmpeg_available() {
        return Err(VideoError::FfmpegMissing);
    }
    let (total_frames, video_fps) = probe_video(path)?;
    let nframes = smart_nframes(total_frames, video_fps, target_fps, target_nframes).map_err(
        |err| match err {
            VideoError::Extract { message, .. } => VideoError::Extract {
                path: path.to_path_buf(),
                message,
            },
            other => other,
        },
    )?;

    let indices = uniform_indices(total_frames, nframes);
    let mut frames = Vec::with_capacity(indices.len());
    for idx in indices {
        let png_bytes = extract_frame_png(path, idx, video_fps)?;
        let img = image::load_from_memory(&png_bytes).map_err(|err| VideoError::Decode {
            path: path.to_path_buf(),
            source: err,
        })?;
        frames.push(img);
    }
    if frames.is_empty() {
        return Err(VideoError::EmptyVideo(path.to_path_buf()));
    }
    Ok(frames)
}

/// Convenience: load multiple videos, returning one frame vector per video.
///
/// Errors short-circuit on the first failure. Callers that need
/// per-video error tolerance should call [`load_video`] in a loop and
/// handle the [`Result`] themselves.
pub fn load_videos(
    paths: &[PathBuf],
    target_fps: Option<f64>,
    target_nframes: Option<usize>,
) -> Result<Vec<Vec<DynamicImage>>, VideoError> {
    let mut all = Vec::with_capacity(paths.len());
    for path in paths {
        all.push(load_video(path, target_fps, target_nframes)?);
    }
    Ok(all)
}

/// Compute `nframes` evenly-spaced frame indices across
/// `[0, total_frames - 1]`. Mirrors the Python `np.linspace(...).round()
/// .astype(int)` call used in `load_video`.
fn uniform_indices(total_frames: usize, nframes: usize) -> Vec<usize> {
    if nframes <= 1 {
        return vec![0];
    }
    let last = total_frames.saturating_sub(1) as f64;
    let step = last / (nframes as f64 - 1.0);
    (0..nframes)
        .map(|i| (i as f64 * step).round() as usize)
        .map(|i| i.min(total_frames.saturating_sub(1)))
        .collect()
}

/// Extract a single frame at `frame_idx` (0-based) as PNG bytes via
/// `ffmpeg -ss ... -frames:v 1 -f image2 -c:v png -`.
///
/// We seek by timestamp instead of frame number because ffmpeg's frame-
/// number seeking is filter-graph-side (`-vf select=eq(n,i)`) and runs
/// the entire decoder up to that index, which is O(n) per frame. The
/// timestamp approach (input `-ss`) is O(1) — close enough since we
/// already chose `frame_idx` uniformly.
fn extract_frame_png(path: &Path, frame_idx: usize, video_fps: f64) -> Result<Vec<u8>, VideoError> {
    let timestamp = frame_idx as f64 / video_fps.max(1e-3);
    let output = Command::new("ffmpeg")
        .args([
            "-loglevel",
            "error",
            "-ss",
            &format!("{timestamp:.6}"),
            "-i",
        ])
        .arg(path)
        .args(["-frames:v", "1", "-f", "image2", "-c:v", "png", "-"])
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .output()
        .map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                VideoError::FfmpegMissing
            } else {
                VideoError::Io {
                    path: path.to_path_buf(),
                    source: err,
                }
            }
        })?;
    if !output.status.success() {
        return Err(VideoError::Extract {
            path: path.to_path_buf(),
            message: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    if output.stdout.is_empty() {
        return Err(VideoError::Extract {
            path: path.to_path_buf(),
            message: format!("ffmpeg produced empty PNG for frame {frame_idx}"),
        });
    }
    Ok(output.stdout)
}

#[cfg(test)]
#[path = "video_tests.rs"]
mod tests;
