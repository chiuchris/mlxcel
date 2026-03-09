//! mlxcel - High-performance LLM inference on Apple Silicon
//!
//! This crate provides efficient inference for Large Language Models using
//! direct MLX C++ bindings via mlxcel-core.

pub mod lora;
pub mod models;
pub mod phi3v_prompt;
pub mod qwen_vl;
pub mod server;
pub mod tokenizer;
pub mod vision;
pub mod vlm_prompt;
pub mod vlm_runtime;

mod loaded_model;
mod loading;
mod runtime;

// Re-export mlxcel-core generate module
pub use mlxcel_core::generate;
pub use mlxcel_core::generate::{CxxGenerator, GenerationStats, LanguageModel, SamplingConfig};
pub use mlxcel_core::speculative::SpeculativeGenerator;

// Re-export split modules
pub use loaded_model::LoadedModel;
pub use loading::{load_model, load_model_with_adapter, read_eos_token_ids};
pub use runtime::{RuntimeDevice, RuntimeSetup, initialize_runtime};
