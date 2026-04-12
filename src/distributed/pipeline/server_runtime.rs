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

//! Server-side in-process pipeline runtime backed by the shared stage worker loop.
//!
//! Used by: server model worker, batch scheduler

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::concatenate;
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr, copy, slice};

use crate::distributed::RequestId;
use crate::distributed::pipeline::{
    InProcessStageWorkerLoop, PipelineWorkerInput, load_in_process_stage_worker,
    resolve_in_process_pipeline_num_layers, resolve_in_process_stage_assignments,
};

pub struct InProcessPipelineModel {
    num_layers: usize,
    eos_token_ids: Vec<i32>,
    worker_loop: Mutex<InProcessStageWorkerLoop>,
    request_ids: Mutex<HashMap<SequenceId, RequestId>>,
}

impl InProcessPipelineModel {
    pub fn load(
        model_dir: &Path,
        pp_layers: Option<&str>,
        pp_micro_batch_size: usize,
    ) -> Result<Self> {
        let num_layers = resolve_in_process_pipeline_num_layers(model_dir)?;
        let assignments = resolve_in_process_stage_assignments(num_layers, None, pp_layers)?;
        let worker_loop =
            load_in_process_stage_worker(model_dir, &assignments, pp_micro_batch_size)?;
        Ok(Self {
            num_layers,
            eos_token_ids: crate::read_eos_token_ids(model_dir),
            worker_loop: Mutex::new(worker_loop),
            request_ids: Mutex::new(HashMap::new()),
        })
    }

    fn request_id_for(&self, seq_id: SequenceId) -> RequestId {
        self.request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .get(&seq_id)
            .cloned()
            .unwrap_or_else(|| panic!("pipeline request id missing for sequence {seq_id}"))
    }

    fn run_single_input(
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

    fn run_batched_inputs(
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

impl LanguageModel for InProcessPipelineModel {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        panic!("in-process pipeline server model requires scheduler sequence ids")
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
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

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        let request_id = RequestId::from_string(format!("pp-seq-{}", seq_id.as_u64()))
            .expect("sequence-derived request id must be valid");
        self.request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .insert(seq_id, request_id);
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.num_layers)
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let seq_id = seq_id.expect("pipeline server model requires a sequence id");
        self.run_single_input(seq_id, input_ids, mask)
            .unwrap_or_else(|err| panic!("pipeline forward failed for sequence {seq_id}: {err}"))
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        assert!(
            input_embeddings.is_none(),
            "in-process pipeline server model currently supports text-only requests"
        );
        self.forward_with_sequence_id(input_ids, seq_id, caches, mask)
    }

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        _batch_caches: &mut [&mut [KVCache]],
        _mask: Option<&MlxArray>,
        _context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let seq_ids = seq_ids.expect("pipeline batched decode requires sequence ids");
        self.run_batched_inputs(seq_ids, input_ids)
            .unwrap_or_else(|err| panic!("pipeline batched decode failed: {err}"))
    }
}
