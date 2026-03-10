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

//! mlxcel - High-performance LLM inference on Apple Silicon
//!
//! This crate provides efficient inference for Large Language Models using
//! direct MLX C++ bindings via mlxcel-core.

pub mod execution;
pub mod lora;
pub mod models;
pub mod multimodal;
pub mod server;
pub mod tokenizer;
pub mod vision;

mod loaded_model;
mod loaded_model_capabilities;
mod loading;
mod model_metadata;

#[cfg(test)]
#[path = "model_metadata_tests.rs"]
mod model_metadata_tests;

// Re-export mlxcel-core generate module
pub use execution::runtime::{RuntimeDevice, RuntimeSetup, initialize_runtime};
pub use execution::sampling;
pub use mlxcel_core::generate;
pub use mlxcel_core::generate::{CxxGenerator, GenerationStats, LanguageModel, SamplingConfig};
pub use mlxcel_core::speculative::SpeculativeGenerator;
pub use multimodal::{
    minicpmo_prompt, moondream3_prompt, phi3v_prompt, phi4_siglip_prompt, phi4mm_prompt, qwen_vl,
    vlm_prompt, vlm_runtime,
};

// Re-export split modules
pub use loaded_model::LoadedModel;
pub use loaded_model_capabilities::VlmRuntimeRef;
pub use loading::{load_model, load_model_with_adapter, read_eos_token_ids};
