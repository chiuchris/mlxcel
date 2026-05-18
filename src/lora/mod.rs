//! LoRA (Low-Rank Adaptation) adapter loading support
//!
//! This module provides functionality to load and apply LoRA adapters
//! to base models. LoRA enables efficient fine-tuning by adding low-rank
//! update matrices to frozen base weights.
//!
//! Uses mlxcel-core types (WeightMap) for weight storage.

mod config;
mod loader;

pub use config::{AdapterConfig, LoRAParameters};
pub use loader::{apply_lora_adapters, fuse_lora_weights};
