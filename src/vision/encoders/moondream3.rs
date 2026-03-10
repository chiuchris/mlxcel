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

//! Moondream3 vision tower.
//!
//! The Rust port mirrors the ViT backbone and projection MLP from the shipped
//! reference code. Local crop fusion is currently reduced by averaging the
//! local-crop feature grids before the projection stage.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Moondream3VisionConfig {
    #[serde(default = "default_enc_dim")]
    pub enc_dim: usize,
    #[serde(default = "default_enc_patch_size")]
    pub enc_patch_size: usize,
    #[serde(default = "default_enc_n_layers")]
    pub enc_n_layers: usize,
    #[serde(default = "default_enc_ff_dim")]
    pub enc_ff_dim: usize,
    #[serde(default = "default_enc_n_heads")]
    pub enc_n_heads: usize,
    #[serde(default = "default_proj_out_dim")]
    pub proj_out_dim: usize,
    #[serde(default = "default_crop_size")]
    pub crop_size: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_max_crops")]
    pub max_crops: usize,
    #[serde(default = "default_overlap_margin")]
    pub overlap_margin: usize,
    #[serde(default = "default_proj_inner_dim")]
    pub proj_inner_dim: usize,
}

fn default_enc_dim() -> usize {
    1152
}

fn default_enc_patch_size() -> usize {
    14
}

fn default_enc_n_layers() -> usize {
    27
}

fn default_enc_ff_dim() -> usize {
    4304
}

fn default_enc_n_heads() -> usize {
    16
}

fn default_proj_out_dim() -> usize {
    2048
}

fn default_crop_size() -> usize {
    378
}

fn default_in_channels() -> usize {
    3
}

fn default_max_crops() -> usize {
    12
}

fn default_overlap_margin() -> usize {
    4
}

fn default_proj_inner_dim() -> usize {
    8192
}

struct Moondream3VisionAttention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    num_heads: i32,
    scale: f32,
}

impl Moondream3VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        dims: usize,
        num_heads: usize,
    ) -> Result<Self, String> {
        let qkv = UnifiedLinear::from_weights(weights, &format!("{}.qkv", prefix), 64, 4)?;
        let proj = UnifiedLinear::from_weights(weights, &format!("{}.proj", prefix), 64, 4)?;
        let head_dim = dims / num_heads;
        Ok(Self {
            qkv,
            proj,
            num_heads: num_heads as i32,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];
        let qkv = self.qkv.forward(x);
        let qkv_dim = mlxcel_core::array_shape(&qkv)[2];
        let chunk = qkv_dim / 3;
        let q = mlxcel_core::slice_last_dim(&qkv, 0, chunk);
        let k = mlxcel_core::slice_last_dim(&qkv, chunk, chunk * 2);
        let v = mlxcel_core::slice_last_dim(&qkv, chunk * 2, chunk * 3);
        let head_dim = chunk / self.num_heads;

        let q = mlxcel_core::reshape(&q, &[batch, seq_len, self.num_heads, head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::reshape(&k, &[batch, seq_len, self.num_heads, head_dim]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::reshape(&v, &[batch, seq_len, self.num_heads, head_dim]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let attn = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(&q, &k, &v, self.scale, std::ptr::null())
        };
        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[batch, seq_len, self.num_heads * head_dim]);
        self.proj.forward(&attn)
    }
}

struct Moondream3VisionMlp {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl Moondream3VisionMlp {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            fc1: UnifiedLinear::from_weights(weights, &format!("{}.fc1", prefix), 64, 4)?,
            fc2: UnifiedLinear::from_weights(weights, &format!("{}.fc2", prefix), 64, 4)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let hidden = self.fc1.forward(x);
        let hidden = mlxcel_core::gelu_approx(&hidden);
        self.fc2.forward(&hidden)
    }
}

struct Moondream3VisionBlock {
    ln1: LayerNorm,
    attn: Moondream3VisionAttention,
    ln2: LayerNorm,
    mlp: Moondream3VisionMlp,
}

impl Moondream3VisionBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Moondream3VisionConfig,
    ) -> Result<Self, String> {
        Ok(Self {
            ln1: load_layer_norm(weights, &format!("{}.ln1", prefix), 1e-5)?,
            attn: Moondream3VisionAttention::from_weights(
                weights,
                &format!("{}.attn", prefix),
                config.enc_dim,
                config.enc_n_heads,
            )?,
            ln2: load_layer_norm(weights, &format!("{}.ln2", prefix), 1e-5)?,
            mlp: Moondream3VisionMlp::from_weights(weights, &format!("{}.mlp", prefix))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let attn = self.attn.forward(&self.ln1.forward(x));
        let hidden = mlxcel_core::add(x, &attn);
        let mlp = self.mlp.forward(&self.ln2.forward(&hidden));
        mlxcel_core::add(&hidden, &mlp)
    }
}

pub struct Moondream3VisionModel {
    patch_emb: UnifiedLinear,
    pos_emb: UniquePtr<MlxArray>,
    blocks: Vec<Moondream3VisionBlock>,
    post_ln: LayerNorm,
    proj_fc1: UnifiedLinear,
    proj_fc2: UnifiedLinear,
    config: Moondream3VisionConfig,
}

impl Moondream3VisionModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Moondream3VisionConfig,
    ) -> Result<Self, String> {
        let mut blocks = Vec::with_capacity(config.enc_n_layers);
        for idx in 0..config.enc_n_layers {
            blocks.push(Moondream3VisionBlock::from_weights(
                weights,
                &format!("vision.blocks.{}", idx),
                config,
            )?);
        }

        Ok(Self {
            patch_emb: UnifiedLinear::from_weights(weights, "vision.patch_emb", 64, 4)?,
            pos_emb: get_weight_copy(weights, "vision.pos_emb")?,
            blocks,
            post_ln: load_layer_norm(weights, "vision.post_ln", 1e-5)?,
            proj_fc1: UnifiedLinear::from_weights(weights, "vision.proj_mlp.fc1", 64, 4)?,
            proj_fc2: UnifiedLinear::from_weights(weights, "vision.proj_mlp.fc2", 64, 4)?,
            config: config.clone(),
        })
    }

    fn create_patches(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let patch = self.config.enc_patch_size as i32;

        let x = mlxcel_core::reshape(
            x,
            &[batch, channels, height / patch, patch, width / patch, patch],
        );
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 4, 1, 3, 5]);
        mlxcel_core::reshape(
            &x,
            &[
                batch,
                (height / patch) * (width / patch),
                channels * patch * patch,
            ],
        )
    }

    fn encode_crops(&self, pixel_values: &MlxArray) -> UniquePtr<MlxArray> {
        let mut hidden = self.patch_emb.forward(&self.create_patches(pixel_values));
        hidden = mlxcel_core::add(&hidden, &self.pos_emb);
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }
        self.post_ln.forward(&hidden)
    }

    pub fn encode_image_embeddings(&self, pixel_values: &MlxArray) -> UniquePtr<MlxArray> {
        let crops = self.encode_crops(pixel_values);
        let crop_count = mlxcel_core::array_shape(&crops)[0];
        let global = mlxcel_core::slice(&crops, &[0, 0, 0], &[1, 729, self.config.enc_dim as i32]);
        let local_mean = if crop_count > 1 {
            let local = mlxcel_core::slice(
                &crops,
                &[1, 0, 0],
                &[crop_count, 729, self.config.enc_dim as i32],
            );
            mlxcel_core::mean_axis(&local, 0, true)
        } else {
            mlxcel_core::copy(&global)
        };
        let merged = mlxcel_core::concatenate(&global, &local_mean, 2);
        let projected = self.proj_fc1.forward(&merged);
        let projected = mlxcel_core::gelu_approx(&projected);
        self.proj_fc2.forward(&projected)
    }

    pub fn output_token_count(&self) -> usize {
        (self.config.crop_size / self.config.enc_patch_size).pow(2)
    }
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|weight| mlxcel_core::copy(weight))
        .ok_or_else(|| format!("Missing weight: {}", name))
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = get_weight_copy(weights, &format!("{}.weight", prefix))?;
    let bias = weights
        .get(&format!("{}.bias", prefix))
        .map(|value| mlxcel_core::copy(value));
    Ok(LayerNorm::new(weight, bias, eps))
}
