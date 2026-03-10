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

//! Shared multimodal prompt and runtime helpers.
//!
//! This layer sits between model-specific vision implementations and the CLI /
//! server entry points. The goal is to keep prompt rewriting, image-token block
//! expansion, and prepared-embedding plumbing reusable across frontends.
//!
//! Modules:
//! - `phi4_siglip_prompt`: Phi4-SigLIP `<image>` placeholder handling
//! - `minicpmo_prompt`: MiniCPM-o image placeholder expansion and bounds
//! - `qwen_vl`: Qwen-VL token insertion rules
//! - `phi3v_prompt`: Phi3V image-tag normalization
//! - `vlm_prompt`: generic image-token block expansion
//! - `vlm_runtime`: image preprocessing and embedding preparation shared by CLI/server

pub mod minicpmo_prompt;
pub mod phi3v_prompt;
pub mod phi4_siglip_prompt;
pub mod qwen_vl;
pub mod vlm_prompt;
pub mod vlm_runtime;
