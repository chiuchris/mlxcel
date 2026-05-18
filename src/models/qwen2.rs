//! Qwen2 model implementation using mlxcel-core
//!
//! Qwen2 uses the same architecture as Llama with minor differences:
//! - attention_bias support (handled via config)
//! - tie_word_embeddings support (handled via config)
//!
//! For simplicity, we re-export the Llama3 implementation which handles
//! these configuration options correctly.

// Re-export Llama3 components directly
pub use super::llama3::{
    Attention, Llama3Model as Qwen2Model, MLP, ModelArgs, Quantization, RopeScaling,
    TransformerBlock,
};

// Type aliases for clarity
pub type Model = Qwen2Model;
