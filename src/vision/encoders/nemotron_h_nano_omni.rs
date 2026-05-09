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

//! Nemotron H Nano Omni vision tower (RADIO v2.5-H).
//!
//! Faithful Rust port of
//! `references/mlx-vlm/mlx_vlm/models/nemotron_h_nano_omni/vision.py`
//! (issue #554, vision-only scope). The encoder is a Vision Transformer
//! with NVIDIA's RADIO patch generator: a learned `[CLS]` token (with
//! optional teacher-tied registers), a 1D learned positional embedding,
//! and a stack of pre-norm transformer blocks. Each block uses standard
//! multi-head self-attention with bias and a 2-layer MLP with `gelu`.
//!
//! Output structure mirrors upstream `RadioOutput { summary, features }`:
//! `summary` is the leading `num_cls_tokens` slot(s) flattened, and
//! `features` are the trailing patch positions (post-skip).
//!
//! Used by: Nemotron H Nano Omni VLM
//!
//! Weight names are kept identical to upstream so HuggingFace mlx-community
//! checkpoints load directly without remapping. The full weight tree is:
//! ```text
//! vision_model.radio_model.input_conditioner.norm_mean
//! vision_model.radio_model.input_conditioner.norm_std
//! vision_model.radio_model.model.patch_generator.cls_token.token
//! vision_model.radio_model.model.patch_generator.embedder.weight
//! vision_model.radio_model.model.patch_generator.video_embedder.weight
//! vision_model.radio_model.model.patch_generator.pos_embed
//! vision_model.radio_model.model.blocks.{i}.norm1.{weight,bias}
//! vision_model.radio_model.model.blocks.{i}.attn.qkv.{weight,bias}
//! vision_model.radio_model.model.blocks.{i}.attn.proj.{weight,bias}
//! vision_model.radio_model.model.blocks.{i}.norm2.{weight,bias}
//! vision_model.radio_model.model.blocks.{i}.mlp.fc1.{weight,bias}
//! vision_model.radio_model.model.blocks.{i}.mlp.fc2.{weight,bias}
//! ```

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

/// Output of the RADIO vision tower.
///
/// Mirrors upstream `RadioOutput` in `vision.py`. `summary` is the
/// concatenation of class-token slots (one per teacher when
/// `cls_token_per_teacher=True`), flattened to `[batch, num_cls * embed]`.
/// `features` are the patch tokens (after stripping cls + register
/// tokens), shape `[batch, num_patches, embed_dim]`.
pub struct NemotronHNanoOmniRadioOutput {
    pub summary: UniquePtr<MlxArray>,
    pub features: UniquePtr<MlxArray>,
}

/// Optional `args` block inside the upstream `VisionConfig`.
///
/// Only the fields that influence weight shapes or topology are surfaced
/// here. The rest of the upstream `args` dict is ignored — the loader
/// only forwards what is shape-affecting.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct NemotronHNanoOmniVisionArgs {
    /// Override for `max_resolution` when computing the position-embed
    /// table size. Upstream key: `cpe_max_size`.
    #[serde(default)]
    pub cpe_max_size: Option<usize>,
    /// When true and `teachers` is non-empty, the encoder allocates one
    /// `[CLS]` slot per distinct teacher name; otherwise a single
    /// `[CLS]` slot is used. Upstream default is `True`.
    #[serde(default = "default_cls_token_per_teacher")]
    pub cls_token_per_teacher: bool,
    /// Round-up multiple for register tokens that pad the cls-token
    /// region to a hardware-friendly count. `None` → no registers
    /// (matches upstream `register_multiple=None`).
    #[serde(default)]
    pub register_multiple: Option<usize>,
    /// Number of distinct cls slots (one per teacher name).
    /// The upstream code derives this from a list of dicts at runtime;
    /// the loader pre-computes it and stores the count here so the Rust
    /// port does not need to parse a structured teacher list.
    #[serde(default)]
    pub num_distinct_teachers: Option<usize>,
}

fn default_cls_token_per_teacher() -> bool {
    true
}

/// Vision-tower configuration mirrored from the upstream `VisionConfig`
/// dataclass. Only fields used by the Rust port are kept; the rest of
/// the upstream config is ignored (parsed and dropped) to keep the port
/// future-compatible without lockstep-tracking every new flag.
///
/// Defaults match upstream RADIO v2.5-H so omitting fields in
/// `config.json` reproduces the released model.
#[derive(Debug, Clone, Deserialize)]
pub struct NemotronHNanoOmniVisionConfig {
    #[serde(default)]
    pub args: Option<NemotronHNanoOmniVisionArgs>,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_max_resolution")]
    pub max_resolution: usize,
    #[serde(default = "default_video_temporal_patch_size")]
    pub video_temporal_patch_size: usize,
}

fn default_hidden_size() -> usize {
    1280
}
fn default_num_hidden_layers() -> usize {
    32
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_intermediate_size() -> usize {
    5120
}
fn default_image_size() -> usize {
    224
}
fn default_patch_size() -> usize {
    16
}
fn default_max_resolution() -> usize {
    2048
}
fn default_video_temporal_patch_size() -> usize {
    2
}

impl Default for NemotronHNanoOmniVisionConfig {
    fn default() -> Self {
        Self {
            args: None,
            hidden_size: default_hidden_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            intermediate_size: default_intermediate_size(),
            image_size: default_image_size(),
            patch_size: default_patch_size(),
            max_resolution: default_max_resolution(),
            video_temporal_patch_size: default_video_temporal_patch_size(),
        }
    }
}

impl NemotronHNanoOmniVisionConfig {
    fn cpe_max_size(&self) -> usize {
        self.args
            .as_ref()
            .and_then(|a| a.cpe_max_size)
            .unwrap_or(self.max_resolution)
    }

    fn cls_token_per_teacher(&self) -> bool {
        self.args
            .as_ref()
            .map(|a| a.cls_token_per_teacher)
            .unwrap_or(true)
    }

    fn register_multiple(&self) -> Option<usize> {
        self.args.as_ref().and_then(|a| a.register_multiple)
    }

    fn num_distinct_teachers(&self) -> usize {
        self.args
            .as_ref()
            .and_then(|a| a.num_distinct_teachers)
            .unwrap_or(0)
    }

    /// Number of class-token slots emitted by the patch generator.
    pub fn num_cls_tokens(&self) -> usize {
        let teachers = self.num_distinct_teachers();
        if self.cls_token_per_teacher() && teachers > 0 {
            teachers
        } else {
            1
        }
    }

    /// Number of register tokens added to round the cls region up to
    /// `register_multiple`. Mirrors upstream `ClsToken.__init__`.
    pub fn num_registers(&self) -> usize {
        if let Some(register_multiple) = self.register_multiple() {
            let n_tokens = self.num_cls_tokens();
            if register_multiple == 0 {
                0
            } else {
                register_multiple - (n_tokens % register_multiple)
            }
        } else {
            0
        }
    }

    /// Total number of leading "skip" tokens (cls + registers) before the
    /// patch features begin.
    pub fn num_skip(&self) -> usize {
        self.num_cls_tokens() + self.num_registers()
    }
}

fn copy_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {key}"))
}

/// Optional pixel normalization that mirrors `InputConditioner` upstream.
///
/// Implements `(x - norm_mean) / norm_std` with parameter tensors of
/// shape `[3, 1, 1]`. The image processor already performs the
/// HuggingFace-style normalization, so this layer is effectively the
/// identity in the released checkpoint (mean=0, std=1) but is kept for
/// faithfulness with the upstream weight tree.
struct InputConditioner {
    norm_mean: UniquePtr<MlxArray>,
    norm_std: UniquePtr<MlxArray>,
}

impl InputConditioner {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let norm_mean = weights
            .get(&format!("{prefix}.norm_mean"))
            .map(|w| mlxcel_core::copy(w))
            .unwrap_or_else(|| mlxcel_core::zeros(&[3, 1, 1], mlxcel_core::dtype::FLOAT32));
        let norm_std = weights
            .get(&format!("{prefix}.norm_std"))
            .map(|w| mlxcel_core::copy(w))
            .unwrap_or_else(|| mlxcel_core::ones(&[3, 1, 1], mlxcel_core::dtype::FLOAT32));
        Ok(Self {
            norm_mean,
            norm_std,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let centered = mlxcel_core::subtract(x, &self.norm_mean);
        mlxcel_core::divide(&centered, &self.norm_std)
    }
}

/// `[CLS]` (and optional register) prefix prepended to the patch sequence.
struct ClsToken {
    /// Learned token table of shape `[num_cls + num_registers, embed_dim]`.
    token: UniquePtr<MlxArray>,
    /// Cached `(num_cls + num_registers, embed_dim)` so we can build a
    /// broadcast view without re-introspecting the array on every call.
    num_total: i32,
    embed_dim: i32,
}

impl ClsToken {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronHNanoOmniVisionConfig,
    ) -> Result<Self, String> {
        let token = copy_weight(weights, &format!("{prefix}.token"))?;
        let shape = mlxcel_core::array_shape(&token);
        if shape.len() != 2 {
            return Err(format!("{prefix}.token must be 2D, got shape {shape:?}"));
        }
        let _ = config;
        Ok(Self {
            token,
            num_total: shape[0],
            embed_dim: shape[1],
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let dtype = mlxcel_core::array_dtype(x);
        let batch = mlxcel_core::array_shape(x)[0];
        let token = mlxcel_core::expand_dims(&self.token, 0); // [1, N, D]
        let token = mlxcel_core::broadcast_to(&token, &[batch, self.num_total, self.embed_dim]);
        let token = mlxcel_core::astype(&token, dtype);
        mlxcel_core::concatenate(&token, x, 1)
    }
}

/// Patch generator that converts a channels-first image tensor into a
/// sequence of patch embeddings, then prepends `[CLS]` tokens.
///
/// Mirrors upstream `ViTPatchGenerator`. Position embeddings are looked
/// up from a learned 1D table sized for `(max_resolution / patch_size)
/// ** 2` patches; for the default 224x224 input the lookup is a direct
/// slice and no resize is needed.
struct ViTPatchGenerator {
    embedder: UnifiedLinear,
    video_embedder: Option<UnifiedLinear>,
    cls_token: ClsToken,
    pos_embed: UniquePtr<MlxArray>, // [1, num_patches, embed_dim]
    patch_size: usize,
    num_rows: usize,
    num_cols: usize,
    /// Cached "no-resize" input dims used by the released checkpoint.
    /// When the runtime passes the same `(patch_h, patch_w)` we slice the
    /// position-embed table directly. Different sizes fall back to a
    /// bilinear-interpolated lookup.
    input_rows: usize,
    input_cols: usize,
}

impl ViTPatchGenerator {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronHNanoOmniVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let patch_size = config.patch_size;
        let max_input_dims = config.cpe_max_size();
        let num_rows = max_input_dims / patch_size;
        let num_cols = max_input_dims / patch_size;
        let input_rows = config.image_size / patch_size;
        let input_cols = config.image_size / patch_size;

        let embedder =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.embedder"), group_size, bits)?;

        // The video embedder weight is optional in non-video-capable
        // checkpoints. Probe for either the regular or quantized weight
        // keys before constructing.
        let video_embedder_present = weights
            .contains_key(&format!("{prefix}.video_embedder.weight"))
            || weights.contains_key(&format!("{prefix}.video_embedder.scales"));
        let video_embedder = if video_embedder_present {
            Some(UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.video_embedder"),
                group_size,
                bits,
            )?)
        } else {
            None
        };

        let cls_token = ClsToken::from_weights(weights, &format!("{prefix}.cls_token"), config)?;
        let pos_embed = copy_weight(weights, &format!("{prefix}.pos_embed"))?;

        Ok(Self {
            embedder,
            video_embedder,
            cls_token,
            pos_embed,
            patch_size,
            num_rows,
            num_cols,
            input_rows,
            input_cols,
        })
    }

    /// Channels-first `[B, C, H, W]` -> patch tokens `[B, N, C*p*p]`.
    fn im_to_patches(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let patch = self.patch_size as i32;
        let patch_h = height / patch;
        let patch_w = width / patch;
        let reshaped = mlxcel_core::reshape(x, &[batch, channels, patch_h, patch, patch_w, patch]);
        let permuted = mlxcel_core::transpose_axes(&reshaped, &[0, 2, 4, 1, 3, 5]);
        mlxcel_core::reshape(
            &permuted,
            &[batch, patch_h * patch_w, channels * patch * patch],
        )
    }

    /// Slice / broadcast the learned position-embed table to the
    /// runtime's `(patch_h, patch_w)` grid. Returns shape
    /// `[batch, patch_h * patch_w, embed_dim]`.
    fn get_pos_embeddings(
        &self,
        batch_size: i32,
        patch_h: usize,
        patch_w: usize,
    ) -> UniquePtr<MlxArray> {
        // Fast path: runtime grid matches the table's stored grid; just
        // broadcast without slicing.
        if patch_h == self.num_rows && patch_w == self.num_cols {
            let shape = mlxcel_core::array_shape(&self.pos_embed);
            return mlxcel_core::broadcast_to(&self.pos_embed, &[batch_size, shape[1], shape[2]]);
        }

        // Released-checkpoint path: image_size matches the configured
        // input size, so the relevant tile lives at the top-left corner
        // of the position-embed table. We reshape to a 2D grid and slice.
        let shape = mlxcel_core::array_shape(&self.pos_embed);
        let embed_dim = shape[2];
        if patch_h <= self.num_rows && patch_w <= self.num_cols {
            let grid = mlxcel_core::reshape(
                &self.pos_embed,
                &[1, self.num_rows as i32, self.num_cols as i32, embed_dim],
            );
            let sliced = mlxcel_core::slice(
                &grid,
                &[0, 0, 0, 0],
                &[1, patch_h as i32, patch_w as i32, embed_dim],
            );
            let flat = mlxcel_core::reshape(&sliced, &[1, (patch_h * patch_w) as i32, embed_dim]);
            return mlxcel_core::broadcast_to(
                &flat,
                &[batch_size, (patch_h * patch_w) as i32, embed_dim],
            );
        }

        // Larger grid than the stored table — fall back to the broadcast
        // of the full table. This is only reached for resolutions above
        // `cpe_max_size`, which the checkpoint metadata explicitly
        // forbids; we keep the path defensive rather than panicking.
        mlxcel_core::broadcast_to(
            &self.pos_embed,
            &[
                batch_size,
                (self.num_rows * self.num_cols) as i32,
                embed_dim,
            ],
        )
    }

    fn forward(&self, x: &MlxArray, use_video_embedder: bool) -> UniquePtr<MlxArray> {
        let patches = self.im_to_patches(x);
        let projected = if use_video_embedder {
            self.video_embedder
                .as_ref()
                .map(|emb| emb.forward(&patches))
                .unwrap_or_else(|| self.embedder.forward(&patches))
        } else {
            self.embedder.forward(&patches)
        };

        let in_shape = mlxcel_core::array_shape(x);
        let patch_h = (in_shape[2] / self.patch_size as i32) as usize;
        let patch_w = (in_shape[3] / self.patch_size as i32) as usize;
        let pos = self.get_pos_embeddings(in_shape[0], patch_h, patch_w);
        let pos = mlxcel_core::astype(&pos, mlxcel_core::array_dtype(&projected));
        let with_pos = mlxcel_core::add(&projected, &pos);
        self.cls_token.forward(&with_pos)
    }

    /// Reuse the released image-size grid to drive default-shape tests.
    pub fn default_input_dims(&self) -> (usize, usize) {
        (self.input_rows, self.input_cols)
    }
}

/// Standard pre-norm Transformer attention block used by RADIO.
///
/// Matches upstream `Attention` exactly: a single fused QKV projection
/// followed by output projection. Self-attention is full bidirectional
/// (no causal mask, no padding mask) since vision tokens are arranged
/// spatially rather than sequentially.
struct VisionAttention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronHNanoOmniVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let qkv = UnifiedLinear::from_weights(weights, &format!("{prefix}.qkv"), group_size, bits)?;
        let proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.proj"), group_size, bits)?;

        if config.num_attention_heads == 0 {
            return Err("num_attention_heads must be > 0".to_string());
        }
        let head_dim = config.hidden_size / config.num_attention_heads;
        let scale = (head_dim as f32).powf(-0.5);

        Ok(Self {
            qkv,
            proj,
            num_heads: config.num_attention_heads as i32,
            head_dim: head_dim as i32,
            scale,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let length = shape[1];
        let dim = shape[2];

        // Fused QKV projection followed by reshape into `[B, L, 3, H, D]`
        // and a permutation to `[3, B, H, L, D]` so we can split with
        // simple slices. Matches upstream `qkv = qkv.transpose(2, 0, 3, 1, 4)`.
        let qkv = self.qkv.forward(x);
        let qkv = mlxcel_core::reshape(&qkv, &[batch, length, 3, self.num_heads, self.head_dim]);
        let qkv = mlxcel_core::transpose_axes(&qkv, &[2, 0, 3, 1, 4]);

        let q = mlxcel_core::slice(
            &qkv,
            &[0, 0, 0, 0, 0],
            &[1, batch, self.num_heads, length, self.head_dim],
        );
        let k = mlxcel_core::slice(
            &qkv,
            &[1, 0, 0, 0, 0],
            &[2, batch, self.num_heads, length, self.head_dim],
        );
        let v = mlxcel_core::slice(
            &qkv,
            &[2, 0, 0, 0, 0],
            &[3, batch, self.num_heads, length, self.head_dim],
        );
        let q = mlxcel_core::squeeze_axis(&q, 0);
        let k = mlxcel_core::squeeze_axis(&k, 0);
        let v = mlxcel_core::squeeze_axis(&v, 0);

        // Bidirectional attention — no mask, no offset.
        let attn = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &q,
                &k,
                &v,
                self.scale,
                std::ptr::null(),
                0.0,
                0,
            )
        };
        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[batch, length, dim]);
        self.proj.forward(&attn)
    }
}

/// 2-layer MLP with `gelu` activation. Mirrors upstream `MLP`.
struct VisionMLP {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl VisionMLP {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let fc1 = UnifiedLinear::from_weights(weights, &format!("{prefix}.fc1"), group_size, bits)?;
        let fc2 = UnifiedLinear::from_weights(weights, &format!("{prefix}.fc2"), group_size, bits)?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.fc1.forward(x);
        let h = mlxcel_core::gelu(&h);
        self.fc2.forward(&h)
    }
}

/// Pre-norm transformer block. Matches upstream `Block`.
struct VisionBlock {
    norm1: LayerNorm,
    attn: VisionAttention,
    norm2: LayerNorm,
    mlp: VisionMLP,
}

impl VisionBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronHNanoOmniVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let norm1 = load_layer_norm(weights, &format!("{prefix}.norm1"))?;
        let norm2 = load_layer_norm(weights, &format!("{prefix}.norm2"))?;
        let attn = VisionAttention::from_weights(
            weights,
            &format!("{prefix}.attn"),
            config,
            group_size,
            bits,
        )?;
        let mlp = VisionMLP::from_weights(weights, &format!("{prefix}.mlp"), group_size, bits)?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.norm1.forward(x);
        let attn = self.attn.forward(&normed);
        let h = mlxcel_core::add(x, &attn);
        let normed = self.norm2.forward(&h);
        let ffw = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &ffw)
    }
}

fn load_layer_norm(weights: &WeightMap, prefix: &str) -> Result<LayerNorm, String> {
    let weight = copy_weight(weights, &format!("{prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|w| mlxcel_core::copy(w));
    Ok(LayerNorm::new(weight, bias, 1e-6))
}

/// Top-level Nemotron H Nano Omni vision tower.
///
/// Matches the upstream wrapper chain `VisionModel -> RadioModel ->
/// RadioBackbone`: the outer struct owns the input conditioner and the
/// patch generator + transformer stack. `forward` returns the
/// summary/features split that the multimodal projector consumes.
pub struct NemotronHNanoOmniVisionModel {
    config: NemotronHNanoOmniVisionConfig,
    input_conditioner: InputConditioner,
    patch_generator: ViTPatchGenerator,
    blocks: Vec<VisionBlock>,
    num_cls_tokens: i32,
    num_skip: i32,
}

impl NemotronHNanoOmniVisionModel {
    /// Construct the vision tower from a weight map. `prefix` is the
    /// HuggingFace tree prefix (typically `vision_model.radio_model`).
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronHNanoOmniVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let input_conditioner =
            InputConditioner::from_weights(weights, &format!("{prefix}.input_conditioner"))?;
        let patch_generator = ViTPatchGenerator::from_weights(
            weights,
            &format!("{prefix}.model.patch_generator"),
            config,
            group_size,
            bits,
        )?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            blocks.push(VisionBlock::from_weights(
                weights,
                &format!("{prefix}.model.blocks.{layer_idx}"),
                config,
                group_size,
                bits,
            )?);
        }

        let num_cls_tokens = config.num_cls_tokens() as i32;
        let num_skip = config.num_skip() as i32;
        Ok(Self {
            config: config.clone(),
            input_conditioner,
            patch_generator,
            blocks,
            num_cls_tokens,
            num_skip,
        })
    }

    /// Run the vision tower on a `[B, C, H, W]` channels-first input
    /// tensor and return `RadioOutput { summary, features }`.
    pub fn forward(
        &self,
        pixel_values: &MlxArray,
        use_video_embedder: bool,
    ) -> NemotronHNanoOmniRadioOutput {
        let normalized = self.input_conditioner.forward(pixel_values);
        let mut hidden = self
            .patch_generator
            .forward(&normalized, use_video_embedder);
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }

        let shape = mlxcel_core::array_shape(&hidden);
        let batch = shape[0];
        let total = shape[1];
        let embed = shape[2];

        // `summary = y[:, :num_cls]` flattened to `[batch, num_cls *
        // embed]`. `features = y[:, num_skip:]`. Mirrors upstream.
        let summary_slice =
            mlxcel_core::slice(&hidden, &[0, 0, 0], &[batch, self.num_cls_tokens, embed]);
        let summary = mlxcel_core::reshape(&summary_slice, &[batch, self.num_cls_tokens * embed]);

        let features = mlxcel_core::slice(&hidden, &[0, self.num_skip, 0], &[batch, total, embed]);

        NemotronHNanoOmniRadioOutput { summary, features }
    }

    /// Patch size (pixels per patch side). Used by the top-level VLM
    /// to compute `(patch_h, patch_w)` from the input image dimensions.
    pub fn patch_size(&self) -> usize {
        self.config.patch_size
    }

    /// Embed dimension of the encoder output. The multimodal projector
    /// expects this as the per-token feature width.
    pub fn embed_dim(&self) -> usize {
        self.config.hidden_size
    }

    /// Number of cls tokens in the leading slice. Unused outside tests
    /// but worth exposing for assertions.
    pub fn num_cls_tokens(&self) -> usize {
        self.num_cls_tokens as usize
    }

    /// Number of leading skip tokens (cls + registers).
    pub fn num_skip(&self) -> usize {
        self.num_skip as usize
    }

    /// Default input grid `(patch_h, patch_w)` derived from
    /// `image_size / patch_size`. Useful when constructing test inputs
    /// or sanity checks against the released checkpoint shape.
    pub fn default_input_dims(&self) -> (usize, usize) {
        self.patch_generator.default_input_dims()
    }
}

#[cfg(test)]
#[path = "nemotron_h_nano_omni_tests.rs"]
mod tests;
