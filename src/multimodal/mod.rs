//! Shared multimodal prompt and runtime helpers.
//!
//! This layer sits between model-specific vision implementations and the CLI /
//! server entry points. The goal is to keep prompt rewriting, image-token block
//! expansion, and prepared-embedding plumbing reusable across frontends.
//!
//! Modules:
//! - `qwen_vl`: Qwen-VL token insertion rules
//! - `phi3v_prompt`: Phi3V image-tag normalization
//! - `vlm_prompt`: generic image-token block expansion
//! - `vlm_runtime`: image preprocessing and embedding preparation shared by CLI/server

pub mod phi3v_prompt;
pub mod qwen_vl;
pub mod vlm_prompt;
pub mod vlm_runtime;
