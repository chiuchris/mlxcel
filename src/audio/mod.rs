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

//! Gemma4 audio encoder and feature extraction.
//!
//! This module implements the Conformer-based audio encoder for Gemma 4's
//! audio modality, including mel spectrogram feature extraction and the
//! full encoder pipeline.
//!
//! Used by: Gemma4 VLM (audio modality)

mod attention;
pub mod config;
pub mod encoder;
pub mod feature_extractor;

pub use config::AudioConfig;
pub use encoder::AudioEncoder;
pub use feature_extractor::{
    AudioFeatureExtractor, AudioFeatureExtractorConfig, compute_audio_num_tokens, load_wav_file,
};
