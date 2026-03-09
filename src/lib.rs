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
mod loading;

// Re-export mlxcel-core generate module
pub use execution::runtime::{RuntimeDevice, RuntimeSetup, initialize_runtime};
pub use execution::sampling;
pub use mlxcel_core::generate;
pub use mlxcel_core::generate::{CxxGenerator, GenerationStats, LanguageModel, SamplingConfig};
pub use mlxcel_core::speculative::SpeculativeGenerator;
pub use multimodal::{phi3v_prompt, qwen_vl, vlm_prompt, vlm_runtime};

// Re-export split modules
pub use loaded_model::{LoadedModel, VlmRuntimeRef};
pub use loading::{load_model, load_model_with_adapter, read_eos_token_ids};
