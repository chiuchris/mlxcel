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

//! Stage-local execution for pipeline-parallel inference.
//!
//! This module bridges the existing pipeline partition and partial-loading
//! helpers to a real execution seam:
//! - build a stage-local model from a [`StageAssignment`]
//! - run only the assigned layer range
//! - accept either token IDs (entry stage) or hidden states (middle/last stage)
//! - return either hidden states (non-final stage) or logits (final stage)
//!
//! Phase 0 intentionally keeps the adapter surface narrow. The common executor
//! API is generic, while the first concrete adapter targets the standard Llama
//! family so later phases can reuse the same worker/runtime path.
//!
//! Used by: pipeline worker loop, CLI pipeline runtime, server pipeline runtime

use std::path::Path;

use anyhow::{Result, anyhow, bail, ensure};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::pipeline_hint;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, copy};

use crate::models;
use crate::models::{ModelType, get_model_type, sanitize_config_json};

use super::partial_loading::{LayerFilter, filter_weight_map};
use super::partition::StageAssignment;

/// Input payload for a single stage-local forward.
pub enum StageExecutionInput<'a> {
    /// Entry-stage execution from token IDs.
    TokenIds(&'a MlxArray),
    /// Middle/last-stage execution from upstream hidden states.
    HiddenStates(&'a MlxArray),
}

/// Output payload from a stage-local forward.
pub enum StageExecutionOutput {
    /// Hidden states to be forwarded to the next stage.
    HiddenStates(UniquePtr<MlxArray>),
    /// Final logits produced by the last stage.
    Logits(UniquePtr<MlxArray>),
}

impl StageExecutionOutput {
    pub fn into_hidden_states(self) -> Result<UniquePtr<MlxArray>> {
        match self {
            Self::HiddenStates(hidden) => Ok(hidden),
            Self::Logits(_) => bail!("expected hidden states but stage produced logits"),
        }
    }

    pub fn into_logits(self) -> Result<UniquePtr<MlxArray>> {
        match self {
            Self::Logits(logits) => Ok(logits),
            Self::HiddenStates(_) => bail!("expected logits but stage produced hidden states"),
        }
    }
}

/// Common execution interface for one pipeline stage.
pub trait StageExecutor {
    fn stage_assignment(&self) -> &StageAssignment;
    fn layer_filter(&self) -> &LayerFilter;
    fn make_caches(&self) -> Vec<KVCache>;
    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput>;
}

/// A stage-local executor loaded from model weights.
pub struct LoadedStageExecutor {
    stage: StageAssignment,
    filter: LayerFilter,
    backend: StageExecutorBackend,
}

enum StageExecutorBackend {
    Llama(LlamaStageModel),
}

impl LoadedStageExecutor {
    /// Load a stage-local executor from a model directory and assignment.
    ///
    /// This constructor deliberately routes through the existing partial-load
    /// filter so later phases can replace the "load then filter" path with
    /// shard-selective I/O without changing the runtime API.
    pub fn load(model_dir: &Path, stage: &StageAssignment) -> Result<Self> {
        ensure!(
            stage.layer_range.end > stage.layer_range.start,
            "stage {} has an empty layer range",
            stage.stage_index
        );

        let filter = LayerFilter::from_stage(stage);
        let model_type = get_model_type(model_dir)?;
        let backend = match model_type {
            ModelType::Llama => StageExecutorBackend::Llama(LlamaStageModel::load(
                model_dir,
                &filter,
                stage.stage_index,
            )?),
            other => bail!(
                "pipeline stage executor is not implemented for model type {:?} yet",
                other
            ),
        };

        Ok(Self {
            stage: stage.clone(),
            filter,
            backend,
        })
    }
}

impl StageExecutor for LoadedStageExecutor {
    fn stage_assignment(&self) -> &StageAssignment {
        &self.stage
    }

    fn layer_filter(&self) -> &LayerFilter {
        &self.filter
    }

    fn make_caches(&self) -> Vec<KVCache> {
        match &self.backend {
            StageExecutorBackend::Llama(model) => model.make_caches(),
        }
    }

    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        match &self.backend {
            StageExecutorBackend::Llama(model) => model.execute(input, caches, mask),
        }
    }
}

struct LlamaStageModel {
    filter: LayerFilter,
    embed_tokens: Option<UnifiedEmbedding>,
    layers: Vec<models::llama3::TransformerBlock>,
    norm: Option<RMSNorm>,
    lm_head: Option<UnifiedLinear>,
}

impl LlamaStageModel {
    fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|err| anyhow!("failed to read {}: {}", config_path.display(), err))?;
        let config_str = sanitize_config_json(&config_str);
        let args: models::llama3::ModelArgs = serde_json::from_str(&config_str)
            .map_err(|err| anyhow!("failed to parse {}: {}", config_path.display(), err))?;

        let mut weights =
            models::load_and_sanitize_weights(model_dir).map_err(anyhow::Error::msg)?;
        let mut effective_filter = filter.clone();
        if args.tie_word_embeddings && filter.has_lm_head {
            effective_filter.has_embedding = true;
        }
        filter_weight_map(&mut weights, &effective_filter);

        Self::from_filtered_weights(&weights, &args, filter, stage_index)
    }

    fn from_filtered_weights(
        weights: &WeightMap,
        args: &models::llama3::ModelArgs,
        filter: &LayerFilter,
        stage_index: usize,
    ) -> Result<Self> {
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens = if filter.has_embedding {
            Some(
                UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)
                    .map_err(anyhow::Error::msg)?,
            )
        } else {
            None
        };

        let mut layers = Vec::with_capacity(filter.num_layers());
        for layer_idx in filter.layer_range.clone() {
            let layer = models::llama3::TransformerBlock::from_weights(weights, args, layer_idx)
                .map_err(anyhow::Error::msg)?;
            layers.push(layer);
        }

        let (norm, lm_head) = if filter.has_lm_head {
            let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
            let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);
            let lm_head = if args.tie_word_embeddings {
                UnifiedLinear::from_weights(weights, "model.embed_tokens", group_size, bits)
                    .map_err(anyhow::Error::msg)?
            } else {
                UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)
                    .map_err(anyhow::Error::msg)?
            };
            (Some(norm), Some(lm_head))
        } else {
            (None, None)
        };

        ensure!(
            !layers.is_empty(),
            "stage {} did not load any layers from range {}..{}",
            stage_index,
            filter.layer_range.start,
            filter.layer_range.end
        );

        Ok(Self {
            filter: filter.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        ensure!(
            caches.len() == self.layers.len(),
            "stage cache count mismatch: expected {}, got {}",
            self.layers.len(),
            caches.len()
        );

        let mut hidden = match input {
            StageExecutionInput::TokenIds(input_ids) => self
                .embed_tokens
                .as_ref()
                .ok_or_else(|| {
                    anyhow!("stage does not host embeddings; hidden-state input required")
                })?
                .forward(input_ids),
            StageExecutionInput::HiddenStates(hidden_states) => {
                if self.filter.has_embedding {
                    bail!("entry stage expects token IDs, not hidden states");
                }
                copy(hidden_states)
            }
        };

        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &mut caches[i], mask);
            pipeline_hint(&hidden, i, n);
        }

        match (&self.norm, &self.lm_head) {
            (Some(norm), Some(lm_head)) => {
                let hidden = norm.forward(&hidden);
                Ok(StageExecutionOutput::Logits(lm_head.forward(&hidden)))
            }
            _ => Ok(StageExecutionOutput::HiddenStates(hidden)),
        }
    }
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>> {
    weights
        .get(name)
        .map(|weight| copy(weight))
        .ok_or_else(|| anyhow!("weight not found: {}", name))
}
