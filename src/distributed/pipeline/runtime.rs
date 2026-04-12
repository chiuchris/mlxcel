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

//! Runtime backends for server-side pipeline execution.
//!
//! Used by: `server_runtime`, future remote PP runtime

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::concatenate;
use mlxcel_core::{MlxArray, UniquePtr, copy, slice};

use crate::distributed::RequestId;

use super::{
    InProcessStageWorkerLoop, PipelineWorkerInput, load_in_process_stage_worker,
    resolve_in_process_pipeline_num_layers, resolve_in_process_stage_assignments,
};

/// Server-side pipeline runtime abstraction.
///
/// Implementations may execute stages locally, dispatch them to remote peers,
/// or mix both strategies while keeping the `LanguageModel`-facing control
/// plane unchanged.
pub trait PipelineModelRuntime {
    fn prepare_sequence_state(&self, seq_id: SequenceId);
    fn release_sequence_state_by_id(&self, seq_id: SequenceId);
    fn forward_sequence(
        &self,
        seq_id: SequenceId,
        input_ids: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>>;
    fn forward_batched(
        &self,
        seq_ids: &[SequenceId],
        input_ids: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>>;
}

/// Current local runtime backed by the shared in-process stage worker loop.
pub struct InProcessPipelineRuntime {
    worker_loop: Mutex<InProcessStageWorkerLoop>,
    request_ids: Mutex<HashMap<SequenceId, RequestId>>,
}

impl InProcessPipelineRuntime {
    pub fn load(
        model_dir: &Path,
        pp_layers: Option<&str>,
        pp_micro_batch_size: usize,
    ) -> Result<(usize, Vec<i32>, Self)> {
        let num_layers = resolve_in_process_pipeline_num_layers(model_dir)?;
        let assignments = resolve_in_process_stage_assignments(num_layers, None, pp_layers)?;
        let worker_loop =
            load_in_process_stage_worker(model_dir, &assignments, pp_micro_batch_size)?;
        Ok((
            num_layers,
            crate::read_eos_token_ids(model_dir),
            Self {
                worker_loop: Mutex::new(worker_loop),
                request_ids: Mutex::new(HashMap::new()),
            },
        ))
    }

    fn request_id_for(&self, seq_id: SequenceId) -> RequestId {
        self.request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .get(&seq_id)
            .cloned()
            .unwrap_or_else(|| panic!("pipeline request id missing for sequence {seq_id}"))
    }
}

impl PipelineModelRuntime for InProcessPipelineRuntime {
    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        let request_id = RequestId::from_string(format!("pp-seq-{}", seq_id.as_u64()))
            .expect("sequence-derived request id must be valid");
        self.request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .insert(seq_id, request_id);
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        let request_id = self
            .request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .remove(&seq_id);
        if let Some(request_id) = request_id {
            self.worker_loop
                .lock()
                .expect("pipeline worker loop poisoned")
                .release_request(&request_id);
        }
    }

    fn forward_sequence(
        &self,
        seq_id: SequenceId,
        input_ids: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let request_id = self.request_id_for(seq_id);
        let mut input = PipelineWorkerInput::new(request_id, copy(input_ids));
        if let Some(mask) = mask {
            input = input.with_attention_mask(copy(mask));
        }
        let mut worker_loop = self
            .worker_loop
            .lock()
            .expect("pipeline worker loop poisoned");
        let mut outputs = worker_loop.run_to_completion(vec![input])?;
        let output = outputs
            .pop()
            .ok_or_else(|| anyhow!("pipeline worker loop returned no logits"))?;
        Ok(output.logits)
    }

    fn forward_batched(
        &self,
        seq_ids: &[SequenceId],
        input_ids: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape.get(1).copied().unwrap_or(1);
        let request_ids: Vec<RequestId> =
            seq_ids.iter().map(|&id| self.request_id_for(id)).collect();
        let inputs: Vec<PipelineWorkerInput> = request_ids
            .iter()
            .enumerate()
            .map(|(i, request_id)| {
                PipelineWorkerInput::new(
                    request_id.clone(),
                    slice(input_ids, &[i as i32, 0], &[i as i32 + 1, seq_len]),
                )
            })
            .collect();
        let mut worker_loop = self
            .worker_loop
            .lock()
            .expect("pipeline worker loop poisoned");
        let outputs = worker_loop.run_to_completion(inputs)?;
        let mut logits_by_request: HashMap<String, UniquePtr<MlxArray>> = HashMap::new();
        for output in outputs {
            logits_by_request.insert(output.request_id.as_str().to_string(), output.logits);
        }

        let mut ordered = request_ids.into_iter();
        let first_request = ordered
            .next()
            .ok_or_else(|| anyhow!("pipeline batched decode received an empty request set"))?;
        let mut merged = logits_by_request
            .remove(first_request.as_str())
            .ok_or_else(|| anyhow!("missing logits for {}", first_request))?;
        for request_id in ordered {
            let logits = logits_by_request
                .remove(request_id.as_str())
                .ok_or_else(|| anyhow!("missing logits for {}", request_id))?;
            merged = concatenate(&merged, &logits, 0);
        }
        Ok(merged)
    }
}

/// Placeholder config for a future remote transport-backed runtime.
#[derive(Debug, Clone)]
pub struct RemotePipelineRuntimeConfig {
    pub stage_peers: Vec<String>,
    pub transport_backend: crate::distributed::TransportBackend,
}

/// Remote runtime placeholder used to prove the server seam is now transport-capable.
pub struct RemotePipelineRuntime {
    config: RemotePipelineRuntimeConfig,
}

impl RemotePipelineRuntime {
    pub fn new(config: RemotePipelineRuntimeConfig) -> Self {
        Self { config }
    }

    fn unavailable(&self) -> anyhow::Error {
        anyhow!(
            "remote pipeline runtime is not implemented yet (backend={}, peers={})",
            self.config.transport_backend,
            self.config.stage_peers.join(",")
        )
    }
}

impl PipelineModelRuntime for RemotePipelineRuntime {
    fn prepare_sequence_state(&self, _seq_id: SequenceId) {}

    fn release_sequence_state_by_id(&self, _seq_id: SequenceId) {}

    fn forward_sequence(
        &self,
        _seq_id: SequenceId,
        _input_ids: &MlxArray,
        _mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        Err(self.unavailable())
    }

    fn forward_batched(
        &self,
        _seq_ids: &[SequenceId],
        _input_ids: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>> {
        Err(self.unavailable())
    }
}
