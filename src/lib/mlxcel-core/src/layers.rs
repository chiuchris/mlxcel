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

//! High-level model layer implementations using mlx-cxx
//!
//! This module provides Rust wrappers for common neural network layers
//! using the mlx-cxx bindings for optimal performance.
//!
//! Cache state machines now live in `crate::cache`; this module re-exports
//! them so existing model code can keep importing cache types through
//! `mlxcel_core::layers`.

use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;

pub use crate::cache::{ChunkedKVCache, KVCache, RotatingKVCache};

/// Quantized weight structure for 4-bit/8-bit quantized layers
/// Supports affine, mxfp4, nvfp4, and mxfp8 quantization modes.
/// For mxfp4/nvfp4/mxfp8 modes, biases is None.
pub struct QuantizedWeight {
    pub weight: UniquePtr<MlxArray>,
    pub scales: UniquePtr<MlxArray>,
    pub biases: Option<UniquePtr<MlxArray>>,
    pub group_size: i32,
    pub bits: i32,
    pub mode: String,
}

impl QuantizedWeight {
    /// Create a new quantized weight from raw components (affine mode)
    pub fn new(
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            weight,
            scales,
            biases: Some(biases),
            group_size,
            bits,
            mode: "affine".to_string(),
        }
    }

    /// Create a new quantized weight with explicit mode
    pub fn new_with_mode(
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: Option<UniquePtr<MlxArray>>,
        group_size: i32,
        bits: i32,
        mode: String,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
            mode,
        }
    }

    /// Get raw pointer to biases (null if not present, e.g. mxfp4/nvfp4/mxfp8)
    pub fn biases_ptr(&self) -> *const MlxArray {
        match &self.biases {
            Some(b) => b.as_ref().unwrap() as *const MlxArray,
            None => std::ptr::null(),
        }
    }
}

// Embedding Layers.
/// Non-quantized embedding layer
pub struct Embedding {
    pub weight: UniquePtr<MlxArray>,
}

impl Embedding {
    /// Create a new embedding layer
    pub fn new(weight: UniquePtr<MlxArray>) -> Self {
        Self { weight }
    }

    /// Load from weight map
    pub fn from_weights(weights: &crate::weights::WeightMap, prefix: &str) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let weight = weights
            .get(&weight_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
        Ok(Self { weight })
    }

    /// Embedding lookup: indices -> embeddings
    pub fn forward(&self, indices: &MlxArray) -> UniquePtr<MlxArray> {
        ffi::embedding(&self.weight, indices)
    }

    /// Use embedding as linear projection (for tied embeddings/lm_head)
    /// y = x @ W (no transpose needed since embedding weight is [vocab, dim])
    pub fn as_linear(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let wt = ffi::transpose(&self.weight);
        ffi::matmul(x, &wt)
    }
}

/// Quantized embedding layer (4-bit/8-bit)
/// Supports affine, mxfp4, nvfp4, and mxfp8 quantization modes.
pub struct QuantizedEmbedding {
    pub weight: UniquePtr<MlxArray>,
    pub scales: UniquePtr<MlxArray>,
    pub biases: Option<UniquePtr<MlxArray>>,
    pub group_size: i32,
    pub bits: i32,
    pub mode: String,
}

impl QuantizedEmbedding {
    /// Create a new quantized embedding layer (affine mode)
    pub fn new(
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            weight,
            scales,
            biases: Some(biases),
            group_size,
            bits,
            mode: "affine".to_string(),
        }
    }

    /// Load from weight map
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Self::from_weights_with_mode(weights, prefix, group_size, bits, "affine")
    }

    /// Load from weight map with explicit quantization mode
    pub fn from_weights_with_mode(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let scales_name = format!("{}.scales", prefix);
        let biases_name = format!("{}.biases", prefix);

        let weight = weights
            .get(&weight_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
        let scales = weights
            .get(&scales_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Scales not found: {}", scales_name))?;
        // biases may not exist for mxfp4/nvfp4/mxfp8 modes
        let biases = weights.get(&biases_name).map(|w| ffi::copy(w));

        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
            mode: mode.to_string(),
        })
    }

    fn biases_ptr(&self) -> *const MlxArray {
        match &self.biases {
            Some(b) => b.as_ref().unwrap() as *const MlxArray,
            None => std::ptr::null(),
        }
    }

    /// Quantized embedding lookup with dequantization
    pub fn forward(&self, indices: &MlxArray) -> UniquePtr<MlxArray> {
        unsafe {
            ffi::quantized_embedding(
                &self.weight,
                &self.scales,
                self.biases_ptr(),
                indices,
                self.group_size,
                self.bits,
                &self.mode,
            )
        }
    }

    /// Use embedding as linear projection (for tied embeddings/lm_head)
    pub fn as_linear(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        unsafe {
            ffi::quantized_linear_forward(
                x,
                &self.weight,
                &self.scales,
                self.biases_ptr(),
                std::ptr::null(),
                self.group_size,
                self.bits,
                &self.mode,
            )
        }
    }
}

/// Unified embedding that auto-detects quantization
pub enum UnifiedEmbedding {
    Quantized(QuantizedEmbedding),
    Regular(Embedding),
}

impl UnifiedEmbedding {
    /// Load from weight map, auto-detecting quantization
    ///
    /// Detects quantization by checking for `.scales` key in weights.
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let scales_name = format!("{}.scales", prefix);

        if weights.contains_key(&scales_name) {
            // Quantized embedding
            Ok(Self::Quantized(QuantizedEmbedding::from_weights(
                weights, prefix, group_size, bits,
            )?))
        } else {
            // Regular embedding
            Ok(Self::Regular(Embedding::from_weights(weights, prefix)?))
        }
    }

    /// Embedding lookup
    pub fn forward(&self, indices: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized(e) => e.forward(indices),
            Self::Regular(e) => e.forward(indices),
        }
    }

    /// Use embedding as linear projection
    pub fn as_linear(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized(e) => e.as_linear(x),
            Self::Regular(e) => e.as_linear(x),
        }
    }

    /// Check if this is a quantized embedding
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }
}

/// RMS Normalization layer
pub struct RMSNorm {
    pub weight: UniquePtr<MlxArray>,
    pub eps: f32,
}

impl RMSNorm {
    /// Create a new RMS norm layer
    pub fn new(weight: UniquePtr<MlxArray>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Forward pass using fast RMS norm kernel
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        ffi::fast_rms_norm(x, &self.weight, self.eps)
    }
}

/// Gemma-style RMS Normalization layer with (1 + weight) pattern
/// Gemma-style RMS normalization with (1 + weight) adjustment.
/// Used by: Gemma, Gemma2, Gemma3, Gemma3n
pub struct GemmaRMSNorm {
    pub weight: UniquePtr<MlxArray>,
    /// Pre-computed (1 + weight) to avoid per-forward allocation
    adjusted_weight: UniquePtr<MlxArray>,
    pub eps: f32,
}

impl GemmaRMSNorm {
    /// Create a new Gemma RMS norm layer.
    /// Pre-computes (1 + weight) once at construction time.
    pub fn new(weight: UniquePtr<MlxArray>, eps: f32) -> Self {
        let ones = ffi::ones(
            &[ffi::array_shape(&weight)[0]],
            ffi::array_dtype(&weight),
        );
        let adjusted_weight = ffi::add(&ones, &weight);
        Self {
            weight,
            adjusted_weight,
            eps,
        }
    }

    /// Forward pass using fast RMS norm kernel with pre-computed (1 + weight)
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        ffi::fast_rms_norm(x, &self.adjusted_weight, self.eps)
    }
}

/// Layer Normalization layer (standard LayerNorm with weight and optional bias)
pub struct LayerNorm {
    pub weight: UniquePtr<MlxArray>,
    pub bias: Option<UniquePtr<MlxArray>>,
    pub eps: f32,
}

impl LayerNorm {
    /// Create a new layer norm
    pub fn new(weight: UniquePtr<MlxArray>, bias: Option<UniquePtr<MlxArray>>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Forward pass using fast layer norm kernel
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let weight_ptr = self.weight.as_ref().unwrap() as *const MlxArray;
        let bias_ptr = self
            .bias
            .as_ref()
            .map(|b| b.as_ref().unwrap() as *const MlxArray)
            .unwrap_or(std::ptr::null());

        unsafe { ffi::fast_layer_norm(x, weight_ptr, bias_ptr, self.eps) }
    }
}

/// Optional LoRA weights for runtime on-the-fly application.
/// Used by: Phi4MM VLM (vision LoRA active during prefill only)
pub struct LoRAWeights {
    /// LoRA A weight: [rank, in_features]
    pub a: UniquePtr<MlxArray>,
    /// LoRA B weight: [out_features, rank]
    pub b: UniquePtr<MlxArray>,
    pub scale: f32,
    /// Whether LoRA is currently active. Uses Cell for interior mutability
    /// so that after_prefill(&self) can toggle it without &mut self.
    active: std::cell::Cell<bool>,
}

/// Regular (non-quantized) Linear layer
pub struct Linear {
    pub weight: UniquePtr<MlxArray>,
    pub bias: Option<UniquePtr<MlxArray>>,
    /// Optional runtime LoRA (applied on-the-fly, not fused into weight)
    lora: Option<LoRAWeights>,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(weight: UniquePtr<MlxArray>, bias: Option<UniquePtr<MlxArray>>) -> Self {
        Self {
            weight,
            bias,
            lora: None,
        }
    }

    /// Load from weight map
    pub fn from_weights(weights: &crate::weights::WeightMap, prefix: &str) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);

        let weight = weights
            .get(&weight_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;

        // Check for optional bias
        let bias_name = format!("{}.bias", prefix);
        let bias = weights.get(&bias_name).map(|w| ffi::copy(w));

        Ok(Self {
            weight,
            bias,
            lora: None,
        })
    }

    /// Set runtime LoRA weights. Starts active.
    pub fn set_lora(&mut self, a: UniquePtr<MlxArray>, b: UniquePtr<MlxArray>, scale: f32) {
        self.lora = Some(LoRAWeights {
            a,
            b,
            scale,
            active: std::cell::Cell::new(true),
        });
    }

    /// Toggle LoRA active state (no-op if no LoRA weights set)
    pub fn set_lora_active(&self, active: bool) {
        if let Some(ref lora) = self.lora {
            lora.active.set(active);
        }
    }

    /// Forward pass: y = x @ W.T + bias [+ LoRA if active]
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Transpose weight from [out, in] to [in, out]
        let wt = ffi::transpose(&self.weight);
        let mut result = ffi::matmul(x, &wt);

        // Apply runtime LoRA: result += (x @ A.T) @ B.T * scale
        if let Some(ref lora) = self.lora {
            if lora.active.get() {
                let at = ffi::transpose(&lora.a);
                let bt = ffi::transpose(&lora.b);
                let lora_out = ffi::matmul(&ffi::matmul(x, &at), &bt);
                let scaled = crate::multiply_scalar(&lora_out, lora.scale);
                result = ffi::add(&result, &scaled);
            }
        }

        match &self.bias {
            Some(b) => ffi::add(&result, b),
            None => result,
        }
    }
}

/// Unified Linear layer that auto-detects quantization
///
/// Checks for `.scales` key in weight map to determine whether to use
/// quantized or regular linear operations. Replaces both the old
/// `QuantizedLinear` and `UnifiedLinear` types.
///
/// Used by: all text/VLM model implementations
pub enum UnifiedLinear {
    Quantized {
        weight: QuantizedWeight,
        bias: Option<UniquePtr<MlxArray>>,
    },
    Regular(Linear),
}

/// Backward-compatible alias
pub type QuantizedLinear = UnifiedLinear;

impl UnifiedLinear {
    /// Create a new quantized linear layer (explicit construction)
    pub fn new(weight: QuantizedWeight, bias: Option<UniquePtr<MlxArray>>) -> Self {
        Self::Quantized { weight, bias }
    }

    /// Load from weight map, auto-detecting quantization (affine mode)
    ///
    /// Detects quantization by checking for `.scales` key in weights.
    /// Falls back to regular Linear if scales are absent.
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Self::from_weights_with_mode(weights, prefix, group_size, bits, "affine")
    }

    /// Load from weight map with explicit quantization mode
    ///
    /// Detects quantization by checking for `.scales` key in weights.
    /// Falls back to regular Linear if scales are absent.
    /// For mxfp4/nvfp4/mxfp8 modes, biases are optional.
    pub fn from_weights_with_mode(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<Self, String> {
        let scales_name = format!("{}.scales", prefix);

        if weights.contains_key(&scales_name) {
            // Quantized path
            let weight_name = format!("{}.weight", prefix);
            let biases_name = format!("{}.biases", prefix);

            let weight = weights
                .get(&weight_name)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
            let scales = weights
                .get(&scales_name)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Scales not found: {}", scales_name))?;
            // biases may not exist for mxfp4/nvfp4/mxfp8 modes
            let biases = weights.get(&biases_name).map(|w| ffi::copy(w));

            let qweight = QuantizedWeight {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode: mode.to_string(),
            };

            let bias_name = format!("{}.bias", prefix);
            let bias = weights.get(&bias_name).map(|w| ffi::copy(w));

            Ok(Self::Quantized {
                weight: qweight,
                bias,
            })
        } else {
            // Fallback to regular linear (non-quantized model)
            Ok(Self::Regular(Linear::from_weights(weights, prefix)?))
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized { weight, bias } => {
                let bias_ptr = bias
                    .as_ref()
                    .map(|b| b.as_ref().unwrap() as *const MlxArray)
                    .unwrap_or(std::ptr::null());

                unsafe {
                    ffi::quantized_linear_forward(
                        x,
                        &weight.weight,
                        &weight.scales,
                        weight.biases_ptr(),
                        bias_ptr,
                        weight.group_size,
                        weight.bits,
                        &weight.mode,
                    )
                }
            }
            Self::Regular(linear) => linear.forward(x),
        }
    }

    /// Set runtime LoRA weights (only for Regular variant)
    pub fn set_lora(&mut self, a: UniquePtr<MlxArray>, b: UniquePtr<MlxArray>, scale: f32) {
        if let Self::Regular(linear) = self {
            linear.set_lora(a, b, scale);
        }
    }

    /// Toggle LoRA active state
    pub fn set_lora_active(&self, active: bool) {
        if let Self::Regular(linear) = self {
            linear.set_lora_active(active);
        }
    }

    /// Check if this is a quantized linear layer
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }

    /// Get a reference to the inner QuantizedWeight (if quantized)
    /// Used by compiled MLP operations that need direct access to weight/scales/biases
    pub fn quantized_weight(&self) -> Option<&QuantizedWeight> {
        match self {
            Self::Quantized { weight, .. } => Some(weight),
            Self::Regular(_) => None,
        }
    }
}

/// Quantized per-head linear layer for MLA (Multi-head Latent Attention)
/// Weight shape: [num_heads, output_dim, input_dim_packed]
/// Used in GLM4 MoE Lite, DeepSeek-V2, etc.
pub struct QuantizedMultiLinear {
    pub weight: UniquePtr<MlxArray>,
    pub scales: UniquePtr<MlxArray>,
    pub biases: Option<UniquePtr<MlxArray>>,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantizedMultiLinear {
    /// Create a new quantized multi-linear layer
    pub fn new(
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: Option<UniquePtr<MlxArray>>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        }
    }

    /// Load from weight map
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let scales_name = format!("{}.scales", prefix);
        let biases_name = format!("{}.biases", prefix);

        let weight = weights
            .get(&weight_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
        let scales = weights
            .get(&scales_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Scales not found: {}", scales_name))?;
        let biases = weights.get(&biases_name).map(|w| ffi::copy(w));

        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    /// Forward pass: per-head linear projection
    /// x: [batch, heads, seq, input_dim]
    /// Returns: [batch, heads, seq, output_dim]
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let biases_ptr: *const MlxArray = match &self.biases {
            Some(b) => b.as_ref().unwrap() as *const MlxArray,
            None => std::ptr::null(),
        };

        unsafe {
            ffi::quantized_matmul(
                x,
                &self.weight,
                &self.scales,
                biases_ptr,
                true, // transpose
                self.group_size,
                self.bits,
                "affine",
            )
        }
    }

    /// Forward pass without transpose: x @ weight
    /// Used by MLA embed_q(kv_latent, transpose=False) for projecting latent to K
    pub fn forward_no_transpose(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let biases_ptr: *const MlxArray = match &self.biases {
            Some(b) => b.as_ref().unwrap() as *const MlxArray,
            None => std::ptr::null(),
        };

        unsafe {
            ffi::quantized_matmul(
                x,
                &self.weight,
                &self.scales,
                biases_ptr,
                false, // no transpose
                self.group_size,
                self.bits,
                "affine",
            )
        }
    }

    /// Dequantize weights to full precision
    /// Returns: [num_heads, output_dim, input_dim]
    pub fn dequantize(&self) -> UniquePtr<MlxArray> {
        let biases_ptr: *const MlxArray = match &self.biases {
            Some(b) => b.as_ref().unwrap() as *const MlxArray,
            None => std::ptr::null(),
        };

        unsafe {
            ffi::dequantize(
                &self.weight,
                &self.scales,
                biases_ptr,
                self.group_size,
                self.bits,
                "affine",
            )
        }
    }
}

/// SwiGLU MLP layer with optional compilation for kernel fusion
pub struct SwiGLUMLP {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
    pub use_compiled: bool,
}

impl SwiGLUMLP {
    /// Create a new SwiGLU MLP layer
    pub fn new(
        gate_proj: QuantizedWeight,
        up_proj: QuantizedWeight,
        down_proj: QuantizedWeight,
        use_compiled: bool,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
            use_compiled,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if self.use_compiled {
            // Use compiled version with kernel fusion
            // Falls back to non-compiled for non-affine modes inside C++
            unsafe {
                ffi::compiled_moe_expert_forward(
                    x,
                    &self.gate_proj.weight,
                    &self.gate_proj.scales,
                    self.gate_proj.biases_ptr(),
                    &self.up_proj.weight,
                    &self.up_proj.scales,
                    self.up_proj.biases_ptr(),
                    &self.down_proj.weight,
                    &self.down_proj.scales,
                    self.down_proj.biases_ptr(),
                    self.gate_proj.group_size,
                    self.gate_proj.bits,
                    &self.gate_proj.mode,
                )
            }
        } else {
            // Non-compiled version
            let gate = unsafe {
                ffi::quantized_linear_forward(
                    x,
                    &self.gate_proj.weight,
                    &self.gate_proj.scales,
                    self.gate_proj.biases_ptr(),
                    std::ptr::null(),
                    self.gate_proj.group_size,
                    self.gate_proj.bits,
                    &self.gate_proj.mode,
                )
            };

            let up = unsafe {
                ffi::quantized_linear_forward(
                    x,
                    &self.up_proj.weight,
                    &self.up_proj.scales,
                    self.up_proj.biases_ptr(),
                    std::ptr::null(),
                    self.up_proj.group_size,
                    self.up_proj.bits,
                    &self.up_proj.mode,
                )
            };

            let silu_gate = ffi::silu(&gate);
            let activated = ffi::multiply(&silu_gate, &up);

            unsafe {
                ffi::quantized_linear_forward(
                    &activated,
                    &self.down_proj.weight,
                    &self.down_proj.scales,
                    self.down_proj.biases_ptr(),
                    std::ptr::null(),
                    self.down_proj.group_size,
                    self.down_proj.bits,
                    &self.down_proj.mode,
                )
            }
        }
    }
}

/// MoE Switch layer using gather_qmm for efficient expert routing
pub struct MoESwitch {
    /// Expert weights: [num_experts, ...]
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
    pub num_experts: i32,
}

impl MoESwitch {
    /// Create a new MoE switch layer
    pub fn new(
        gate_proj: QuantizedWeight,
        up_proj: QuantizedWeight,
        down_proj: QuantizedWeight,
        num_experts: i32,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
            num_experts,
        }
    }

    /// Forward pass with expert indices
    /// x: [batch, seq_len, hidden_dim]
    /// indices: [batch, seq_len] - expert index for each token
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = unsafe {
            ffi::gather_qmm(
                x,
                &self.gate_proj.weight,
                &self.gate_proj.scales,
                self.gate_proj.biases_ptr(),
                std::ptr::null(),    // lhs_indices
                indices as *const _, // rhs_indices
                true,                // transpose
                self.gate_proj.group_size,
                self.gate_proj.bits,
                false, // sorted_indices
                &self.gate_proj.mode,
            )
        };

        let up = unsafe {
            ffi::gather_qmm(
                x,
                &self.up_proj.weight,
                &self.up_proj.scales,
                self.up_proj.biases_ptr(),
                std::ptr::null(),
                indices as *const _,
                true,
                self.up_proj.group_size,
                self.up_proj.bits,
                false,
                &self.up_proj.mode,
            )
        };

        let activated = ffi::compiled_swiglu_activation(&gate, &up);

        unsafe {
            ffi::gather_qmm(
                &activated,
                &self.down_proj.weight,
                &self.down_proj.scales,
                self.down_proj.biases_ptr(),
                std::ptr::null(),
                indices as *const _,
                true,
                self.down_proj.group_size,
                self.down_proj.bits,
                false,
                &self.down_proj.mode,
            )
        }
    }
}

/// Attention layer with RoPE and KV cache
pub struct Attention {
    pub q_proj: QuantizedWeight,
    pub k_proj: QuantizedWeight,
    pub v_proj: QuantizedWeight,
    pub o_proj: QuantizedWeight,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub rope_scale: f32,
}

impl Attention {
    /// Create a new attention layer
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj: QuantizedWeight,
        k_proj: QuantizedWeight,
        v_proj: QuantizedWeight,
        o_proj: QuantizedWeight,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rope_dims: i32,
        rope_base: f32,
        rope_scale: f32,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dims,
            rope_base,
            rope_scale,
        }
    }

    /// Forward pass
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(x);
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V
        let q = unsafe {
            ffi::quantized_linear_forward(
                x,
                &self.q_proj.weight,
                &self.q_proj.scales,
                self.q_proj.biases_ptr(),
                std::ptr::null(),
                self.q_proj.group_size,
                self.q_proj.bits,
                &self.q_proj.mode,
            )
        };

        let k = unsafe {
            ffi::quantized_linear_forward(
                x,
                &self.k_proj.weight,
                &self.k_proj.scales,
                self.k_proj.biases_ptr(),
                std::ptr::null(),
                self.k_proj.group_size,
                self.k_proj.bits,
                &self.k_proj.mode,
            )
        };

        let v = unsafe {
            ffi::quantized_linear_forward(
                x,
                &self.v_proj.weight,
                &self.v_proj.scales,
                self.v_proj.biases_ptr(),
                std::ptr::null(),
                self.v_proj.group_size,
                self.v_proj.bits,
                &self.v_proj.mode,
            )
        };

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = ffi::reshape(&q, &[batch_size, seq_len, self.n_heads, self.head_dim]);
        let k = ffi::reshape(&k, &[batch_size, seq_len, self.n_kv_heads, self.head_dim]);
        let v = ffi::reshape(&v, &[batch_size, seq_len, self.n_kv_heads, self.head_dim]);

        // Apply RoPE
        let offset = cache.offset;
        let q = ffi::fast_rope(
            &q,
            self.rope_dims,
            false,
            self.rope_base,
            self.rope_scale,
            offset,
        );
        let k = ffi::fast_rope(
            &k,
            self.rope_dims,
            false,
            self.rope_base,
            self.rope_scale,
            offset,
        );

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = ffi::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = ffi::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = ffi::transpose_axes(&v, &[0, 2, 1, 3]);

        // Update KV cache and get sliced views
        let (k, v) = cache.update_and_fetch(k, v);

        // Compute attention scale
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Scaled dot-product attention
        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        let attn_out =
            unsafe { ffi::fast_scaled_dot_product_attention(&q, &k, &v, scale, mask_ptr) };

        // Transpose back and reshape
        let attn_out = ffi::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = ffi::reshape(
            &attn_out,
            &[batch_size, seq_len, self.n_heads * self.head_dim],
        );

        // Output projection
        unsafe {
            ffi::quantized_linear_forward(
                &attn_out,
                &self.o_proj.weight,
                &self.o_proj.scales,
                self.o_proj.biases_ptr(),
                std::ptr::null(),
                self.o_proj.group_size,
                self.o_proj.bits,
                &self.o_proj.mode,
            )
        }
    }
}

// MultiLinear Layer (for MLA attention: embed_q, unembed_out).
/// MultiLinear layer for per-head linear projections.
///
/// Weight shape: `[num_heads, output_dims, input_dims]`
///
/// Used by MLA attention (DeepSeek V3/V3.2, GLM4 MoE Lite) for:
/// - `embed_q`: projects Q_nope into KV latent space
/// - `unembed_out`: projects attention output from latent space to V dimensions
///
/// Supports both quantized and non-quantized weights.
/// Used by: DeepSeek V3, DeepSeek V3.2, GLM4 MoE Lite
pub enum MultiLinear {
    Quantized(QuantizedMultiLinear),
    Regular(RegularMultiLinear),
}

/// Non-quantized multi-head linear layer.
/// Weight shape: `[num_heads, output_dims, input_dims]`
pub struct RegularMultiLinear {
    pub weight: UniquePtr<MlxArray>,
}

impl MultiLinear {
    /// Load from weight map, auto-detecting quantization.
    ///
    /// Checks for `.scales` key to determine if weights are quantized.
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let scales_name = format!("{}.scales", prefix);

        if weights.contains_key(&scales_name) {
            // Quantized: use existing QuantizedMultiLinear
            Ok(MultiLinear::Quantized(QuantizedMultiLinear::from_weights(
                weights, prefix, group_size, bits,
            )?))
        } else {
            // Non-quantized: regular weight
            let weight = weights
                .get(&weight_name)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
            Ok(MultiLinear::Regular(RegularMultiLinear { weight }))
        }
    }

    /// Forward pass with transpose (default behavior).
    ///
    /// Computes `x @ weight.swapaxes(-1, -2)`.
    /// Input x: `[..., num_heads, seq_len, input_dims]`
    /// Output: `[..., num_heads, seq_len, output_dims]`
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MultiLinear::Quantized(q) => q.forward(x),
            MultiLinear::Regular(r) => {
                // weight is [num_heads, output_dims, input_dims]
                // swapaxes(-1, -2) → [num_heads, input_dims, output_dims]
                let wt = ffi::transpose_axes(&r.weight, &[0, 2, 1]);
                ffi::matmul(x, &wt)
            }
        }
    }

    /// Forward pass without transpose.
    ///
    /// Computes `x @ weight`.
    /// Input x: `[..., 1_or_heads, seq_len, output_dims]`
    /// Output: `[..., num_heads, seq_len, input_dims]`
    pub fn forward_no_transpose(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MultiLinear::Quantized(q) => q.forward_no_transpose(x),
            MultiLinear::Regular(r) => ffi::matmul(x, &r.weight),
        }
    }
}
