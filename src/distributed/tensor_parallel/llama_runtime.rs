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

//! In-process tensor-parallel runtime for dense Llama/Qwen-family models.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::{self, ModelType};

use super::{
    EmbeddingMode, ModelShardPlan, ShardConfig, ShardSpec, ShardStrategy,
    TensorParallelPlanSummary, compute_shard_spec, resolve_model_shard_plan,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorParallelRuntimeKind {
    LlamaStyle,
    Qwen3,
    Gemma3,
    Ernie45,
    HunyuanV1Dense,
}

#[derive(Debug, Clone)]
pub struct TensorParallelRuntimeSupport {
    pub kind: TensorParallelRuntimeKind,
    pub summary: TensorParallelPlanSummary,
    pub force_no_batch: bool,
}

pub struct TensorParallelLlamaModel {
    ranks: Vec<crate::models::Llama3Model>,
    num_layers_per_rank: usize,
    prefer_precise_reduction: bool,
}

impl TensorParallelLlamaModel {
    pub fn from_model_dir(model_dir: &Path, shard_config: ShardConfig) -> Result<Self> {
        let support = validate_supported_runtime(model_dir, shard_config, None)?;
        ensure!(
            support.kind == TensorParallelRuntimeKind::LlamaStyle,
            "tensor-parallel Llama-style runtime cannot load {:?}",
            support.summary.model_type
        );
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: crate::models::llama3::ModelArgs =
            serde_json::from_str(&config_str).context("failed to parse llama config")?;
        let weights = models::load_and_sanitize_weights(model_dir).map_err(anyhow::Error::msg)?;
        Self::from_full_weights(&args, &weights, &support.summary.plan)
    }

    fn from_full_weights(
        args: &crate::models::llama3::ModelArgs,
        weights: &WeightMap,
        plan: &ModelShardPlan,
    ) -> Result<Self> {
        let mut ranks = Vec::with_capacity(plan.tp_size);
        for rank in 0..plan.tp_size {
            let rank_weights = shard_weight_map(weights, plan, rank)?;
            let rank_args = local_llama_args(args, plan)?;
            let rank_model = crate::models::Llama3Model::from_weights(&rank_weights, &rank_args)
                .map_err(anyhow::Error::msg)?;
            ranks.push(rank_model);
        }

        Ok(Self {
            ranks,
            num_layers_per_rank: args.num_hidden_layers,
            prefer_precise_reduction: args.tie_word_embeddings,
        })
    }
}

impl LanguageModel for TensorParallelLlamaModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut rank_caches = split_rank_caches(caches, self.num_layers_per_rank, self.ranks.len())
            .expect("tensor-parallel caches must match num_layers");
        let mut h = self.ranks[0].embed_tokens.forward(input_ids);

        for layer_idx in 0..self.num_layers_per_rank {
            let attn_norm = self.ranks[0].layers[layer_idx].input_layernorm.forward(&h);
            let attn_parts: Vec<_> = self
                .ranks
                .iter()
                .zip(rank_caches.iter_mut())
                .map(|(rank, caches)| {
                    rank.layers[layer_idx].self_attn.forward(
                        &attn_norm,
                        &mut caches[layer_idx],
                        mask,
                    )
                })
                .collect();
            let attn_out = if self.prefer_precise_reduction {
                reduce_sum_f32(attn_parts)
            } else {
                reduce_sum(attn_parts)
            };
            h = mlxcel_core::add(&h, &attn_out);

            let ffn_norm = self.ranks[0].layers[layer_idx]
                .post_attention_layernorm
                .forward(&h);
            let ffn_parts: Vec<_> = self
                .ranks
                .iter()
                .map(|rank| rank.layers[layer_idx].mlp.forward(&ffn_norm))
                .collect();
            let ff_out = if self.prefer_precise_reduction {
                reduce_sum_f32(ffn_parts)
            } else {
                reduce_sum(ffn_parts)
            };
            h = mlxcel_core::add(&h, &ff_out);
        }

        let h = self.ranks[0].norm.forward(&h);
        self.ranks[0].lm_head.forward(&h)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut rank_caches = split_rank_caches(caches, self.num_layers_per_rank, self.ranks.len())
            .expect("tensor-parallel caches must match num_layers");
        let mut h = match input_embeddings {
            Some(embeddings) => mlxcel_core::copy(embeddings),
            None => self.ranks[0].embed_tokens.forward(input_ids),
        };

        for layer_idx in 0..self.num_layers_per_rank {
            let attn_norm = self.ranks[0].layers[layer_idx].input_layernorm.forward(&h);
            let attn_parts: Vec<_> = self
                .ranks
                .iter()
                .zip(rank_caches.iter_mut())
                .map(|(rank, caches)| {
                    rank.layers[layer_idx].self_attn.forward(
                        &attn_norm,
                        &mut caches[layer_idx],
                        mask,
                    )
                })
                .collect();
            let attn_out = if self.prefer_precise_reduction {
                reduce_sum_f32(attn_parts)
            } else {
                reduce_sum(attn_parts)
            };
            h = mlxcel_core::add(&h, &attn_out);

            let ffn_norm = self.ranks[0].layers[layer_idx]
                .post_attention_layernorm
                .forward(&h);
            let ffn_parts: Vec<_> = self
                .ranks
                .iter()
                .map(|rank| rank.layers[layer_idx].mlp.forward(&ffn_norm))
                .collect();
            let ff_out = if self.prefer_precise_reduction {
                reduce_sum_f32(ffn_parts)
            } else {
                reduce_sum(ffn_parts)
            };
            h = mlxcel_core::add(&h, &ff_out);
        }

        let h = self.ranks[0].norm.forward(&h);
        self.ranks[0].lm_head.forward(&h)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.ranks[0].embed_tokens.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.num_layers()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.num_layers_per_rank * self.ranks.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.ranks[0].eos_token_ids()
    }

    fn supports_batching(&self) -> bool {
        false
    }

    fn supports_batched_prefill(&self) -> bool {
        false
    }
}

pub struct TensorParallelQwen3Model {
    ranks: Vec<crate::models::Qwen3Model>,
    num_layers_per_rank: usize,
}

impl TensorParallelQwen3Model {
    pub fn from_model_dir(model_dir: &Path, shard_config: ShardConfig) -> Result<Self> {
        let support = validate_supported_runtime(model_dir, shard_config, None)?;
        ensure!(
            support.kind == TensorParallelRuntimeKind::Qwen3,
            "tensor-parallel Qwen3 runtime cannot load {:?}",
            support.summary.model_type
        );
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: crate::models::qwen3::ModelArgs =
            serde_json::from_str(&config_str).context("failed to parse qwen3 config")?;
        let weights = models::load_and_sanitize_weights(model_dir).map_err(anyhow::Error::msg)?;
        Self::from_full_weights(&args, &weights, &support.summary.plan)
    }

    fn from_full_weights(
        args: &crate::models::qwen3::ModelArgs,
        weights: &WeightMap,
        plan: &ModelShardPlan,
    ) -> Result<Self> {
        let mut ranks = Vec::with_capacity(plan.tp_size);
        for rank in 0..plan.tp_size {
            let rank_weights = shard_weight_map(weights, plan, rank)?;
            let rank_args = local_qwen3_args(args, plan)?;
            let rank_model = crate::models::Qwen3Model::from_weights(&rank_weights, &rank_args)
                .map_err(anyhow::Error::msg)?;
            ranks.push(rank_model);
        }

        Ok(Self {
            ranks,
            num_layers_per_rank: args.num_hidden_layers,
        })
    }

    fn final_logits(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        if let Some(ref lm_head) = self.ranks[0].lm_head {
            lm_head.forward(hidden)
        } else {
            self.ranks[0].embed_tokens.as_linear(hidden)
        }
    }
}

pub struct TensorParallelGemma3Model {
    ranks: Vec<crate::models::Gemma3Model>,
    rank_caches: RefCell<Vec<Vec<crate::models::gemma3::Cache>>>,
    num_layers_per_rank: usize,
}

impl TensorParallelGemma3Model {
    pub fn from_model_dir(model_dir: &Path, shard_config: ShardConfig) -> Result<Self> {
        let support = validate_supported_runtime(model_dir, shard_config, None)?;
        ensure!(
            support.kind == TensorParallelRuntimeKind::Gemma3,
            "tensor-parallel Gemma3 runtime cannot load {:?}",
            support.summary.model_type
        );
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: crate::models::gemma3::ModelArgs =
            serde_json::from_str(&config_str).context("failed to parse gemma3 config")?;
        let weights = models::load_and_sanitize_weights(model_dir).map_err(anyhow::Error::msg)?;
        Self::from_full_weights(&args, &weights, &support.summary.plan)
    }

    fn from_full_weights(
        args: &crate::models::gemma3::ModelArgs,
        weights: &WeightMap,
        plan: &ModelShardPlan,
    ) -> Result<Self> {
        let mut ranks = Vec::with_capacity(plan.tp_size);
        for rank in 0..plan.tp_size {
            let rank_weights = shard_gemma3_weight_map(weights, plan, rank, args)?;
            let rank_args = local_gemma3_args(args, plan)?;
            let rank_model = crate::models::Gemma3Model::from_weights(&rank_weights, &rank_args)
                .map_err(anyhow::Error::msg)?;
            ranks.push(rank_model);
        }

        let rank_caches = ranks
            .iter()
            .map(crate::models::Gemma3Model::make_caches)
            .collect();

        Ok(Self {
            ranks,
            rank_caches: RefCell::new(rank_caches),
            num_layers_per_rank: args.num_hidden_layers,
        })
    }

    fn reset_caches(&self) {
        *self.rank_caches.borrow_mut() = self
            .ranks
            .iter()
            .map(crate::models::Gemma3Model::make_caches)
            .collect();
    }
}

impl LanguageModel for TensorParallelGemma3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut rank_caches = self.rank_caches.borrow_mut();
        let mut h = self.ranks[0].get_embed_tokens(input_ids);
        let seq_len = mlxcel_core::array_shape(input_ids)[1];

        let (global_mask, sliding_mask) =
            gemma3_masks(mask, seq_len, &mut rank_caches[0], &self.ranks[0]);

        for layer_idx in 0..self.num_layers_per_rank {
            let attn_norm = self.ranks[0].layers[layer_idx].input_layernorm.forward(&h);
            let layer_mask = gemma3_layer_mask(
                layer_idx,
                self.ranks[0].sliding_window_pattern,
                global_mask.as_deref(),
                sliding_mask.as_deref(),
            );
            let attn_parts: Vec<_> = self
                .ranks
                .iter()
                .zip(rank_caches.iter_mut())
                .map(|(rank, caches)| {
                    rank.layers[layer_idx].self_attn.forward(
                        &attn_norm,
                        gemma3_cache_interface(&mut caches[layer_idx]),
                        layer_mask,
                    )
                })
                .collect();
            let attn_out = reduce_sum_f32(attn_parts);
            let post_attn = self.ranks[0].layers[layer_idx]
                .post_attention_layernorm
                .forward(&attn_out);
            h = mlxcel_core::compiled_clip_residual(&h, &post_attn);

            let ffn_norm = self.ranks[0].layers[layer_idx]
                .pre_feedforward_layernorm
                .forward(&h);
            let ffn_parts: Vec<_> = self
                .ranks
                .iter()
                .map(|rank| rank.layers[layer_idx].mlp.forward(&ffn_norm))
                .collect();
            let ff_out = reduce_sum_f32(ffn_parts);
            let post_ff = self.ranks[0].layers[layer_idx]
                .post_feedforward_layernorm
                .forward(&ff_out);
            h = mlxcel_core::compiled_clip_residual(&h, &post_ff);
        }

        let h = self.ranks[0].norm.forward(&h);
        self.ranks[0].lm_head.forward(&h)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut rank_caches = self.rank_caches.borrow_mut();
        let mut h = if let Some(embeddings) = input_embeddings {
            let h = mlxcel_core::copy(embeddings);
            mlxcel_core::multiply_scalar(&h, (self.ranks[0].hidden_size as f32).sqrt())
        } else {
            self.ranks[0].get_embed_tokens(input_ids)
        };
        let seq_len = mlxcel_core::array_shape(input_ids)[1];

        let (global_mask, sliding_mask) =
            gemma3_masks(mask, seq_len, &mut rank_caches[0], &self.ranks[0]);

        for layer_idx in 0..self.num_layers_per_rank {
            let attn_norm = self.ranks[0].layers[layer_idx].input_layernorm.forward(&h);
            let layer_mask = gemma3_layer_mask(
                layer_idx,
                self.ranks[0].sliding_window_pattern,
                global_mask.as_deref(),
                sliding_mask.as_deref(),
            );
            let attn_parts: Vec<_> = self
                .ranks
                .iter()
                .zip(rank_caches.iter_mut())
                .map(|(rank, caches)| {
                    rank.layers[layer_idx].self_attn.forward(
                        &attn_norm,
                        gemma3_cache_interface(&mut caches[layer_idx]),
                        layer_mask,
                    )
                })
                .collect();
            let attn_out = reduce_sum_f32(attn_parts);
            let post_attn = self.ranks[0].layers[layer_idx]
                .post_attention_layernorm
                .forward(&attn_out);
            h = mlxcel_core::compiled_clip_residual(&h, &post_attn);

            let ffn_norm = self.ranks[0].layers[layer_idx]
                .pre_feedforward_layernorm
                .forward(&h);
            let ffn_parts: Vec<_> = self
                .ranks
                .iter()
                .map(|rank| rank.layers[layer_idx].mlp.forward(&ffn_norm))
                .collect();
            let ff_out = reduce_sum_f32(ffn_parts);
            let post_ff = self.ranks[0].layers[layer_idx]
                .post_feedforward_layernorm
                .forward(&ff_out);
            h = mlxcel_core::compiled_clip_residual(&h, &post_ff);
        }

        let h = self.ranks[0].norm.forward(&h);
        self.ranks[0].lm_head.forward(&h)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.ranks[0].embed_tokens.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.reset_caches();
        (0..self.num_layers_per_rank)
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.num_layers_per_rank
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![0, 1, 106]
    }

    fn supports_batching(&self) -> bool {
        false
    }

    fn supports_batched_prefill(&self) -> bool {
        false
    }
}

impl LanguageModel for TensorParallelQwen3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut rank_caches = split_rank_caches(caches, self.num_layers_per_rank, self.ranks.len())
            .expect("tensor-parallel caches must match num_layers");
        let mut h = self.ranks[0].embed_tokens.forward(input_ids);

        for layer_idx in 0..self.num_layers_per_rank {
            let attn_norm = self.ranks[0].layers[layer_idx].input_layernorm.forward(&h);
            let attn_parts: Vec<_> = self
                .ranks
                .iter()
                .zip(rank_caches.iter_mut())
                .map(|(rank, caches)| {
                    rank.layers[layer_idx].self_attn.forward(
                        &attn_norm,
                        &mut caches[layer_idx],
                        mask,
                    )
                })
                .collect();
            let attn_out = reduce_sum(attn_parts);
            h = mlxcel_core::add(&h, &attn_out);

            let ffn_norm = self.ranks[0].layers[layer_idx]
                .post_attention_layernorm
                .forward(&h);
            let ffn_parts: Vec<_> = self
                .ranks
                .iter()
                .map(|rank| rank.layers[layer_idx].mlp.forward(&ffn_norm))
                .collect();
            let ff_out = reduce_sum(ffn_parts);
            h = mlxcel_core::add(&h, &ff_out);
        }

        let h = self.ranks[0].norm.forward(&h);
        self.final_logits(&h)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut rank_caches = split_rank_caches(caches, self.num_layers_per_rank, self.ranks.len())
            .expect("tensor-parallel caches must match num_layers");
        let mut h = match input_embeddings {
            Some(embeddings) => mlxcel_core::copy(embeddings),
            None => self.ranks[0].embed_tokens.forward(input_ids),
        };

        for layer_idx in 0..self.num_layers_per_rank {
            let attn_norm = self.ranks[0].layers[layer_idx].input_layernorm.forward(&h);
            let attn_parts: Vec<_> = self
                .ranks
                .iter()
                .zip(rank_caches.iter_mut())
                .map(|(rank, caches)| {
                    rank.layers[layer_idx].self_attn.forward(
                        &attn_norm,
                        &mut caches[layer_idx],
                        mask,
                    )
                })
                .collect();
            let attn_out = reduce_sum(attn_parts);
            h = mlxcel_core::add(&h, &attn_out);

            let ffn_norm = self.ranks[0].layers[layer_idx]
                .post_attention_layernorm
                .forward(&h);
            let ffn_parts: Vec<_> = self
                .ranks
                .iter()
                .map(|rank| rank.layers[layer_idx].mlp.forward(&ffn_norm))
                .collect();
            let ff_out = reduce_sum(ffn_parts);
            h = mlxcel_core::add(&h, &ff_out);
        }

        let h = self.ranks[0].norm.forward(&h);
        self.final_logits(&h)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.ranks[0].get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.num_layers()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.num_layers_per_rank * self.ranks.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.ranks[0].eos_token_ids()
    }

    fn supports_batching(&self) -> bool {
        false
    }

    fn supports_batched_prefill(&self) -> bool {
        false
    }
}

pub struct TensorParallelErnie45Model {
    ranks: Vec<crate::models::Ernie45Model>,
    num_layers_per_rank: usize,
}

impl TensorParallelErnie45Model {
    pub fn from_model_dir(model_dir: &Path, shard_config: ShardConfig) -> Result<Self> {
        let support = validate_supported_runtime(model_dir, shard_config, None)?;
        ensure!(
            support.kind == TensorParallelRuntimeKind::Ernie45,
            "tensor-parallel Ernie45 runtime cannot load {:?}",
            support.summary.model_type
        );
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: crate::models::ernie4_5::ModelArgs =
            serde_json::from_str(&config_str).context("failed to parse ernie4_5 config")?;
        let weights = models::load_and_sanitize_weights(model_dir).map_err(anyhow::Error::msg)?;
        Self::from_full_weights(&args, &weights, &support.summary.plan)
    }

    fn from_full_weights(
        args: &crate::models::ernie4_5::ModelArgs,
        weights: &WeightMap,
        plan: &ModelShardPlan,
    ) -> Result<Self> {
        let mut ranks = Vec::with_capacity(plan.tp_size);
        for rank in 0..plan.tp_size {
            let rank_weights = shard_ernie45_weight_map(weights, plan, rank, args)?;
            let rank_args = local_ernie45_args(args, plan)?;
            let rank_model = crate::models::Ernie45Model::from_weights(&rank_weights, &rank_args)
                .map_err(anyhow::Error::msg)?;
            ranks.push(rank_model);
        }

        Ok(Self {
            ranks,
            num_layers_per_rank: args.num_hidden_layers,
        })
    }

    fn final_logits(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        if let Some(ref lm_head) = self.ranks[0].lm_head {
            lm_head.forward(hidden)
        } else {
            self.ranks[0].embed_tokens.as_linear(hidden)
        }
    }
}

impl LanguageModel for TensorParallelErnie45Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut rank_caches = split_rank_caches(caches, self.num_layers_per_rank, self.ranks.len())
            .expect("tensor-parallel caches must match num_layers");
        let mut h = self.ranks[0].embed_tokens.forward(input_ids);

        for layer_idx in 0..self.num_layers_per_rank {
            let attn_norm = self.ranks[0].layers[layer_idx].input_layernorm.forward(&h);
            let attn_parts: Vec<_> = self
                .ranks
                .iter()
                .zip(rank_caches.iter_mut())
                .map(|(rank, caches)| {
                    rank.layers[layer_idx].self_attn.forward(
                        &attn_norm,
                        &mut caches[layer_idx],
                        mask,
                    )
                })
                .collect();
            let attn_out = reduce_sum_f32(attn_parts);
            h = mlxcel_core::add(&h, &attn_out);

            let ffn_norm = self.ranks[0].layers[layer_idx]
                .post_attention_layernorm
                .forward(&h);
            let ffn_parts: Vec<_> = self
                .ranks
                .iter()
                .map(|rank| rank.layers[layer_idx].mlp.forward(&ffn_norm))
                .collect();
            let ff_out = reduce_sum_f32(ffn_parts);
            h = mlxcel_core::add(&h, &ff_out);
        }

        let h = self.ranks[0].norm.forward(&h);
        self.final_logits(&h)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.ranks[0].embed_tokens.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.num_layers()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.num_layers_per_rank * self.ranks.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.ranks[0].eos_token_ids()
    }

    fn supports_batching(&self) -> bool {
        false
    }

    fn supports_batched_prefill(&self) -> bool {
        false
    }
}

pub struct TensorParallelHunyuanV1DenseModel {
    ranks: Vec<crate::models::HunyuanV1DenseModel>,
    num_layers_per_rank: usize,
}

impl TensorParallelHunyuanV1DenseModel {
    pub fn from_model_dir(model_dir: &Path, shard_config: ShardConfig) -> Result<Self> {
        let support = validate_supported_runtime(model_dir, shard_config, None)?;
        ensure!(
            support.kind == TensorParallelRuntimeKind::HunyuanV1Dense,
            "tensor-parallel Hunyuan v1 dense runtime cannot load {:?}",
            support.summary.model_type
        );
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: crate::models::hunyuan_v1_dense::ModelArgs =
            serde_json::from_str(&config_str).context("failed to parse hunyuan_v1_dense config")?;
        let weights = models::load_and_sanitize_weights(model_dir).map_err(anyhow::Error::msg)?;
        Self::from_full_weights(&args, &weights, &support.summary.plan)
    }

    fn from_full_weights(
        args: &crate::models::hunyuan_v1_dense::ModelArgs,
        weights: &WeightMap,
        plan: &ModelShardPlan,
    ) -> Result<Self> {
        let mut ranks = Vec::with_capacity(plan.tp_size);
        for rank in 0..plan.tp_size {
            let rank_weights = shard_weight_map(weights, plan, rank)?;
            let rank_args = local_hunyuan_v1_dense_args(args, plan)?;
            let rank_model =
                crate::models::HunyuanV1DenseModel::from_weights(&rank_weights, &rank_args)
                    .map_err(anyhow::Error::msg)?;
            ranks.push(rank_model);
        }

        Ok(Self {
            ranks,
            num_layers_per_rank: args.num_hidden_layers,
        })
    }

    fn final_logits(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        if let Some(ref lm_head) = self.ranks[0].lm_head {
            lm_head.forward(hidden)
        } else {
            self.ranks[0].embed_tokens.as_linear(hidden)
        }
    }
}

impl LanguageModel for TensorParallelHunyuanV1DenseModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut rank_caches = split_rank_caches(caches, self.num_layers_per_rank, self.ranks.len())
            .expect("tensor-parallel caches must match num_layers");
        let mut h = self.ranks[0].embed_tokens.forward(input_ids);

        for layer_idx in 0..self.num_layers_per_rank {
            let attn_norm = self.ranks[0].layers[layer_idx].input_layernorm.forward(&h);
            let attn_parts: Vec<_> = self
                .ranks
                .iter()
                .zip(rank_caches.iter_mut())
                .map(|(rank, caches)| {
                    rank.layers[layer_idx].self_attn.forward(
                        &attn_norm,
                        &mut caches[layer_idx],
                        mask,
                    )
                })
                .collect();
            let attn_out = reduce_sum_f32(attn_parts);
            h = mlxcel_core::add(&h, &attn_out);

            let ffn_norm = self.ranks[0].layers[layer_idx]
                .post_attention_layernorm
                .forward(&h);
            let ffn_parts: Vec<_> = self
                .ranks
                .iter()
                .map(|rank| rank.layers[layer_idx].mlp.forward(&ffn_norm))
                .collect();
            let ff_out = reduce_sum_f32(ffn_parts);
            h = mlxcel_core::add(&h, &ff_out);
        }

        let h = self.ranks[0].norm.forward(&h);
        self.final_logits(&h)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.ranks[0].embed_tokens.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.num_layers()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.num_layers_per_rank * self.ranks.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.ranks[0].eos_token_ids()
    }

    fn supports_batching(&self) -> bool {
        false
    }

    fn supports_batched_prefill(&self) -> bool {
        false
    }
}

pub fn validate_supported_runtime(
    model_path: &Path,
    shard_config: ShardConfig,
    adapter_path: Option<&Path>,
) -> Result<TensorParallelRuntimeSupport> {
    let summary = resolve_model_shard_plan(model_path, shard_config)?;
    let kind = runtime_kind_for(&summary);

    if summary.shard_config.tp_size == 1 {
        return Ok(TensorParallelRuntimeSupport {
            kind: kind.unwrap_or(TensorParallelRuntimeKind::LlamaStyle),
            summary,
            force_no_batch: false,
        });
    }

    if adapter_path.is_some() {
        bail!("tensor-parallel runtime does not support LoRA adapters yet");
    }

    ensure!(
        summary.shard_config.embedding_mode == EmbeddingMode::Replicated,
        "tensor-parallel runtime currently requires --tp-embedding-mode replicated"
    );
    ensure!(
        summary.shard_config.lm_head_mode == EmbeddingMode::Replicated,
        "tensor-parallel runtime currently requires --tp-lm-head-mode replicated"
    );
    let kind = kind.ok_or_else(|| {
        anyhow::anyhow!(
            "tensor-parallel runtime currently supports only dense Llama/Qwen2/Qwen3/Gemma3/ERNIE/Hunyuan models, got {:?} ({})",
            summary.model_type,
            summary.architecture
        )
    })?;

    Ok(TensorParallelRuntimeSupport {
        kind,
        summary,
        force_no_batch: true,
    })
}

fn local_llama_args(
    args: &crate::models::llama3::ModelArgs,
    plan: &ModelShardPlan,
) -> Result<crate::models::llama3::ModelArgs> {
    ensure!(
        args.num_attention_heads.is_multiple_of(plan.tp_size),
        "num_attention_heads ({}) must be divisible by tp_size ({})",
        args.num_attention_heads,
        plan.tp_size
    );
    ensure!(
        args.intermediate_size.is_multiple_of(plan.tp_size),
        "intermediate_size ({}) must be divisible by tp_size ({})",
        args.intermediate_size,
        plan.tp_size
    );

    let num_kv_heads = args.num_kv_heads();
    ensure!(
        num_kv_heads >= plan.tp_size && num_kv_heads.is_multiple_of(plan.tp_size),
        "num_key_value_heads ({num_kv_heads}) must be divisible by tp_size ({}) for the current tensor-parallel runtime",
        plan.tp_size
    );
    ensure!(
        !args.attention_bias && !args.mlp_bias,
        "tensor-parallel runtime currently supports bias-free dense llama models only"
    );

    let mut local = args.clone();
    local.head_dim = Some(args.head_dim());
    local.num_attention_heads /= plan.tp_size;
    local.num_key_value_heads = Some(num_kv_heads / plan.tp_size);
    local.intermediate_size /= plan.tp_size;
    // TP ranks load a replicated lm_head copy even for tied-embedding checkpoints.
    // `load_and_sanitize_weights()` guarantees `lm_head.*` exists by copying from
    // `model.embed_tokens.*` when the original checkpoint ties embeddings.
    local.tie_word_embeddings = false;
    Ok(local)
}

fn local_qwen3_args(
    args: &crate::models::qwen3::ModelArgs,
    plan: &ModelShardPlan,
) -> Result<crate::models::qwen3::ModelArgs> {
    ensure!(
        args.num_attention_heads.is_multiple_of(plan.tp_size),
        "num_attention_heads ({}) must be divisible by tp_size ({})",
        args.num_attention_heads,
        plan.tp_size
    );
    ensure!(
        args.intermediate_size.is_multiple_of(plan.tp_size),
        "intermediate_size ({}) must be divisible by tp_size ({})",
        args.intermediate_size,
        plan.tp_size
    );
    ensure!(
        args.num_key_value_heads >= plan.tp_size
            && args.num_key_value_heads.is_multiple_of(plan.tp_size),
        "num_key_value_heads ({}) must be divisible by tp_size ({}) for the current tensor-parallel runtime",
        args.num_key_value_heads,
        plan.tp_size
    );

    let mut local = args.clone();
    local.num_attention_heads /= plan.tp_size;
    local.num_key_value_heads /= plan.tp_size;
    local.intermediate_size /= plan.tp_size;
    Ok(local)
}

fn local_gemma3_args(
    args: &crate::models::gemma3::ModelArgs,
    plan: &ModelShardPlan,
) -> Result<crate::models::gemma3::ModelArgs> {
    ensure!(
        args.num_attention_heads.is_multiple_of(plan.tp_size),
        "num_attention_heads ({}) must be divisible by tp_size ({})",
        args.num_attention_heads,
        plan.tp_size
    );
    ensure!(
        args.intermediate_size.is_multiple_of(plan.tp_size),
        "intermediate_size ({}) must be divisible by tp_size ({})",
        args.intermediate_size,
        plan.tp_size
    );
    ensure!(
        args.num_key_value_heads > 0,
        "num_key_value_heads must be greater than zero for Gemma3 tensor parallelism"
    );

    let mut local = args.clone();
    local.num_attention_heads /= plan.tp_size;
    if args.num_key_value_heads >= plan.tp_size
        && args.num_key_value_heads.is_multiple_of(plan.tp_size)
    {
        local.num_key_value_heads /= plan.tp_size;
    }
    local.intermediate_size /= plan.tp_size;
    Ok(local)
}

fn local_ernie45_args(
    args: &crate::models::ernie4_5::ModelArgs,
    plan: &ModelShardPlan,
) -> Result<crate::models::ernie4_5::ModelArgs> {
    ensure!(
        args.num_attention_heads.is_multiple_of(plan.tp_size),
        "num_attention_heads ({}) must be divisible by tp_size ({})",
        args.num_attention_heads,
        plan.tp_size
    );
    ensure!(
        args.intermediate_size.is_multiple_of(plan.tp_size),
        "intermediate_size ({}) must be divisible by tp_size ({})",
        args.intermediate_size,
        plan.tp_size
    );
    let num_kv_heads = args.num_kv_heads();
    ensure!(
        num_kv_heads > 0,
        "num_key_value_heads must be greater than zero for ERNIE 4.5 tensor parallelism"
    );
    ensure!(
        if num_kv_heads < plan.tp_size {
            plan.tp_size.is_multiple_of(num_kv_heads)
        } else {
            num_kv_heads.is_multiple_of(plan.tp_size)
        },
        "num_key_value_heads ({num_kv_heads}) must divide tp_size ({}) or be divisible by it for ERNIE 4.5 tensor parallelism",
        plan.tp_size
    );
    ensure!(
        !args.use_bias,
        "tensor-parallel runtime currently supports bias-free ERNIE 4.5 models only"
    );

    let mut local = args.clone();
    local.head_dim = Some(args.head_dim());
    local.num_attention_heads /= plan.tp_size;
    local.num_key_value_heads = Some(if num_kv_heads < plan.tp_size {
        1
    } else {
        num_kv_heads / plan.tp_size
    });
    local.intermediate_size /= plan.tp_size;
    Ok(local)
}

fn local_hunyuan_v1_dense_args(
    args: &crate::models::hunyuan_v1_dense::ModelArgs,
    plan: &ModelShardPlan,
) -> Result<crate::models::hunyuan_v1_dense::ModelArgs> {
    ensure!(
        args.num_attention_heads.is_multiple_of(plan.tp_size),
        "num_attention_heads ({}) must be divisible by tp_size ({})",
        args.num_attention_heads,
        plan.tp_size
    );
    ensure!(
        args.intermediate_size.is_multiple_of(plan.tp_size),
        "intermediate_size ({}) must be divisible by tp_size ({})",
        args.intermediate_size,
        plan.tp_size
    );
    ensure!(
        args.num_key_value_heads >= plan.tp_size
            && args.num_key_value_heads.is_multiple_of(plan.tp_size),
        "num_key_value_heads ({}) must be divisible by tp_size ({}) for the current tensor-parallel runtime",
        args.num_key_value_heads,
        plan.tp_size
    );
    ensure!(
        !args.attention_bias,
        "tensor-parallel runtime currently supports bias-free Hunyuan v1 dense models only"
    );

    let mut local = args.clone();
    local.head_dim = Some(args.head_dim());
    local.num_attention_heads /= plan.tp_size;
    local.num_key_value_heads /= plan.tp_size;
    local.intermediate_size /= plan.tp_size;
    Ok(local)
}

fn runtime_kind_for(summary: &TensorParallelPlanSummary) -> Option<TensorParallelRuntimeKind> {
    match summary.model_type {
        ModelType::Llama if is_llama_style_architecture(&summary.architecture) => {
            Some(TensorParallelRuntimeKind::LlamaStyle)
        }
        ModelType::Qwen2 if is_qwen2_architecture(&summary.architecture) => {
            Some(TensorParallelRuntimeKind::LlamaStyle)
        }
        ModelType::Qwen3 if is_qwen3_architecture(&summary.architecture) => {
            Some(TensorParallelRuntimeKind::Qwen3)
        }
        ModelType::Gemma3 if is_gemma3_architecture(&summary.architecture) => {
            Some(TensorParallelRuntimeKind::Gemma3)
        }
        ModelType::Ernie45 if summary.architecture == "ernie4_5" => {
            Some(TensorParallelRuntimeKind::Ernie45)
        }
        ModelType::HunyuanV1Dense if summary.architecture == "hunyuan_v1_dense" => {
            Some(TensorParallelRuntimeKind::HunyuanV1Dense)
        }
        _ => None,
    }
}

fn is_llama_style_architecture(architecture: &str) -> bool {
    matches!(
        architecture,
        "llama" | "llama3" | "mistral" | "yi" | "tinyllama" | "vicuna"
    )
}

fn is_qwen2_architecture(architecture: &str) -> bool {
    matches!(architecture, "qwen2" | "qwen2.5")
}

fn is_qwen3_architecture(architecture: &str) -> bool {
    architecture == "qwen3"
}

fn is_gemma3_architecture(architecture: &str) -> bool {
    matches!(architecture, "gemma3" | "gemma3_text")
}

fn split_rank_caches<'a>(
    caches: &'a mut [KVCache],
    num_layers_per_rank: usize,
    num_ranks: usize,
) -> Result<Vec<&'a mut [KVCache]>> {
    let expected = num_layers_per_rank * num_ranks;
    ensure!(
        caches.len() == expected,
        "tensor-parallel cache count mismatch: expected {expected}, got {}",
        caches.len()
    );

    let mut remaining = caches;
    let mut per_rank = Vec::with_capacity(num_ranks);
    for _ in 0..num_ranks {
        let (rank_caches, rest) = remaining.split_at_mut(num_layers_per_rank);
        per_rank.push(rank_caches);
        remaining = rest;
    }
    Ok(per_rank)
}

fn reduce_sum(parts: Vec<UniquePtr<MlxArray>>) -> UniquePtr<MlxArray> {
    let mut parts = parts.into_iter();
    let mut acc = parts.next().expect("reduce_sum requires at least one part");
    let dtype = mlxcel_core::array_dtype(&acc);
    if matches!(
        dtype,
        mlxcel_core::dtype::FLOAT16 | mlxcel_core::dtype::BFLOAT16
    ) {
        let mut acc_f32 = mlxcel_core::astype(&acc, mlxcel_core::dtype::FLOAT32);
        for part in parts {
            let part_f32 = mlxcel_core::astype(&part, mlxcel_core::dtype::FLOAT32);
            acc_f32 = mlxcel_core::add(&acc_f32, &part_f32);
        }
        acc = mlxcel_core::astype(&acc_f32, dtype);
    } else {
        for part in parts {
            acc = mlxcel_core::add(&acc, &part);
        }
    }
    acc
}

fn reduce_sum_f32(parts: Vec<UniquePtr<MlxArray>>) -> UniquePtr<MlxArray> {
    let mut parts = parts.into_iter();
    let first = parts
        .next()
        .expect("reduce_sum_f32 requires at least one part");
    let mut acc = mlxcel_core::astype(&first, mlxcel_core::dtype::FLOAT32);
    for part in parts {
        let part_f32 = mlxcel_core::astype(&part, mlxcel_core::dtype::FLOAT32);
        acc = mlxcel_core::add(&acc, &part_f32);
    }
    acc
}

fn shard_weight_map(weights: &WeightMap, plan: &ModelShardPlan, rank: usize) -> Result<WeightMap> {
    let mut sharded = HashMap::with_capacity(weights.len());
    for (name, tensor) in weights {
        let logical_name = logical_weight_name(name);
        let shape = mlxcel_core::array_shape(tensor);
        let shape: Vec<usize> = shape
            .into_iter()
            .map(|dim| usize::try_from(dim).context("negative tensor dimension"))
            .collect::<Result<_>>()?;
        let spec = compute_shard_spec(&logical_name, &shape, plan, rank)?;
        let sharded_tensor = shard_tensor(tensor, &spec)?;
        sharded.insert(name.clone(), sharded_tensor);
    }
    Ok(sharded)
}

fn shard_gemma3_weight_map(
    weights: &WeightMap,
    plan: &ModelShardPlan,
    rank: usize,
    args: &crate::models::gemma3::ModelArgs,
) -> Result<WeightMap> {
    let mut sharded = HashMap::with_capacity(weights.len());
    let replicate_kv = args.num_key_value_heads < plan.tp_size;

    for (name, tensor) in weights {
        let logical_name = logical_weight_name(name);
        let shape = mlxcel_core::array_shape(tensor);
        let shape: Vec<usize> = shape
            .into_iter()
            .map(|dim| usize::try_from(dim).context("negative tensor dimension"))
            .collect::<Result<_>>()?;

        let spec = if replicate_kv
            && (logical_name.ends_with(".self_attn.k_proj.weight")
                || logical_name.ends_with(".self_attn.v_proj.weight"))
        {
            ShardSpec {
                rank,
                tp_size: plan.tp_size,
                shard_axis: 0,
                start_index: 0,
                end_index: shape.first().copied().unwrap_or(0),
                padded: false,
                pad_count: 0,
                strategy: ShardStrategy::Replicated,
            }
        } else {
            compute_shard_spec(&logical_name, &shape, plan, rank)?
        };

        let sharded_tensor = shard_tensor(tensor, &spec)?;
        sharded.insert(name.clone(), sharded_tensor);
    }

    Ok(sharded)
}

fn shard_ernie45_weight_map(
    weights: &WeightMap,
    plan: &ModelShardPlan,
    rank: usize,
    args: &crate::models::ernie4_5::ModelArgs,
) -> Result<WeightMap> {
    let mut sharded = HashMap::with_capacity(weights.len());
    let replicate_kv = args.num_kv_heads() < plan.tp_size;
    let head_dim = args.head_dim();
    let total_kv_heads = args.num_kv_heads();
    let ranks_per_kv = if replicate_kv {
        plan.tp_size / total_kv_heads
    } else {
        1
    };

    for (name, tensor) in weights {
        let logical_name = logical_weight_name(name);
        let shape = mlxcel_core::array_shape(tensor);
        let shape: Vec<usize> = shape
            .into_iter()
            .map(|dim| usize::try_from(dim).context("negative tensor dimension"))
            .collect::<Result<_>>()?;

        let spec = if replicate_kv
            && (logical_name.ends_with(".self_attn.k_proj.weight")
                || logical_name.ends_with(".self_attn.v_proj.weight"))
        {
            let kv_index = rank / ranks_per_kv;
            let start_index = kv_index * head_dim;
            let end_index = start_index + head_dim;
            ShardSpec {
                rank,
                tp_size: plan.tp_size,
                shard_axis: 0,
                start_index,
                end_index,
                padded: false,
                pad_count: 0,
                strategy: ShardStrategy::ColumnParallel,
            }
        } else {
            compute_shard_spec(&logical_name, &shape, plan, rank)?
        };

        let sharded_tensor = shard_tensor(tensor, &spec)?;
        sharded.insert(name.clone(), sharded_tensor);
    }

    Ok(sharded)
}

fn gemma3_masks(
    mask: Option<&MlxArray>,
    seq_len: i32,
    rank0_caches: &mut [crate::models::gemma3::Cache],
    rank0: &crate::models::Gemma3Model,
) -> (Option<UniquePtr<MlxArray>>, Option<UniquePtr<MlxArray>>) {
    if let Some(mask) = mask {
        return (Some(mlxcel_core::copy(mask)), Some(mlxcel_core::copy(mask)));
    }
    if seq_len == 1 {
        return (None, None);
    }

    let global_idx = rank0.sliding_window_pattern - 1;
    let global_offset = gemma3_cache_offset(&mut rank0_caches[global_idx]);
    let global_mask = Some(mlxcel_core::utils::create_causal_mask(
        seq_len,
        global_offset,
    ));
    let sliding_mask = if rank0.sliding_window_pattern > 1 {
        let sliding_offset = gemma3_cache_offset(&mut rank0_caches[0]);
        let max_cache = rank0.sliding_window as i32;
        let effective_offset = sliding_offset.min((max_cache - seq_len).max(0));
        Some(mlxcel_core::utils::create_causal_mask_with_window(
            seq_len,
            effective_offset,
            Some(max_cache),
        ))
    } else {
        None
    };
    (global_mask, sliding_mask)
}

fn gemma3_layer_mask<'a>(
    layer_idx: usize,
    sliding_window_pattern: usize,
    global_mask: Option<&'a MlxArray>,
    sliding_mask: Option<&'a MlxArray>,
) -> Option<&'a MlxArray> {
    if sliding_window_pattern <= 1 {
        return global_mask;
    }
    let is_global = (layer_idx % sliding_window_pattern) == (sliding_window_pattern - 1);
    if is_global { global_mask } else { sliding_mask }
}

fn gemma3_cache_offset(cache: &mut crate::models::gemma3::Cache) -> i32 {
    match cache {
        crate::models::gemma3::Cache::Standard(cache) => cache.offset,
        crate::models::gemma3::Cache::Rotating(cache) => cache.offset,
    }
}

fn gemma3_cache_interface(
    cache: &mut crate::models::gemma3::Cache,
) -> &mut dyn crate::models::gemma3::CacheInterface {
    match cache {
        crate::models::gemma3::Cache::Standard(cache) => cache,
        crate::models::gemma3::Cache::Rotating(cache) => cache,
    }
}

fn logical_weight_name(name: &str) -> String {
    for suffix in [".scales", ".biases", ".bias"] {
        if let Some(prefix) = name.strip_suffix(suffix) {
            return format!("{prefix}.weight");
        }
    }
    name.to_string()
}

fn shard_tensor(tensor: &MlxArray, spec: &ShardSpec) -> Result<UniquePtr<MlxArray>> {
    if spec.strategy == ShardStrategy::Replicated {
        return Ok(mlxcel_core::copy(tensor));
    }

    let shape = mlxcel_core::array_shape(tensor);
    ensure!(
        spec.shard_axis < shape.len(),
        "cannot shard tensor with shape {:?} on axis {}",
        shape,
        spec.shard_axis
    );

    let mut starts = vec![0; shape.len()];
    let mut stops = shape.clone();
    starts[spec.shard_axis] = i32::try_from(spec.start_index).context("shard start overflow")?;
    stops[spec.shard_axis] = i32::try_from(spec.end_index).context("shard stop overflow")?;
    Ok(mlxcel_core::slice(tensor, &starts, &stops))
}

#[cfg(test)]
#[path = "llama_runtime_tests.rs"]
mod tests;
