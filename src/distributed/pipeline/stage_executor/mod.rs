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
//! The runtime and wire protocol stay backend-agnostic. Family-specific stage
//! loaders plug into [`LoadedStageExecutor`] via [`FamilyStageExecutor`].
//!
//! Used by: pipeline worker loop, CLI pipeline runtime, server pipeline runtime

mod common;
mod gemma3;
mod gemma4;
mod gpt_oss;
mod llama;

use std::path::Path;

use anyhow::{Result, bail, ensure};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::{ModelType, get_model_type};

use super::partial_loading::LayerFilter;
use super::partition::StageAssignment;
use gemma3::Gemma3StageExecutor;
use gemma4::Gemma4StageExecutor;
use gpt_oss::GptOssStageExecutor;
use llama::LlamaStageExecutor;

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
    fn release_caches(&self, caches: &[KVCache]);
    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput>;
}

pub trait FamilyStageExecutor {
    fn make_caches(&self) -> Vec<KVCache>;
    fn release_caches(&self, _caches: &[KVCache]) {}
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
    backend: Box<dyn FamilyStageExecutor>,
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
        let backend = load_family_backend(model_dir, &filter, stage.stage_index, model_type)?;

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
        self.backend.make_caches()
    }

    fn release_caches(&self, caches: &[KVCache]) {
        self.backend.release_caches(caches);
    }

    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        self.backend.execute(input, caches, mask)
    }
}

fn load_family_backend(
    model_dir: &Path,
    filter: &LayerFilter,
    stage_index: usize,
    model_type: ModelType,
) -> Result<Box<dyn FamilyStageExecutor>> {
    match model_type {
        ModelType::Llama => Ok(Box::new(LlamaStageExecutor::load(
            model_dir,
            filter,
            stage_index,
        )?)),
        ModelType::GptOss => Ok(Box::new(GptOssStageExecutor::load(
            model_dir,
            filter,
            stage_index,
        )?)),
        ModelType::Gemma3 => Ok(Box::new(Gemma3StageExecutor::load(
            model_dir,
            filter,
            stage_index,
        )?)),
        ModelType::Gemma4 | ModelType::Gemma4VLM => Ok(Box::new(Gemma4StageExecutor::load(
            model_dir,
            filter,
            stage_index,
        )?)),
        other => bail!(
            "pipeline stage executor is not implemented for model type {:?} yet",
            other
        ),
    }
}
