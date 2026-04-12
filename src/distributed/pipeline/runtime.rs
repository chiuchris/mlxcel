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
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::concatenate;
use mlxcel_core::{MlxArray, UniquePtr, copy, slice};

use crate::distributed::RequestId;
use crate::distributed::{
    TcpTransport, TcpTransportConfig, Transport, TransportBackend, TransportMessage,
};

use super::remote_service::{
    PIPELINE_STAGE_COMMAND_OPERATION, PIPELINE_STAGE_RESPONSE_OPERATION, RemoteStageCommand,
    RemoteStageRequest, RemoteStageResponse,
};
use super::wire_tensor::{deserialize_wire_tensor, serialize_mlx_array};
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
    pub transport_backend: TransportBackend,
    pub bind_address: String,
}

pub struct RemotePipelineRuntime {
    stage_peers: Vec<String>,
    transport: Arc<TcpTransport>,
    io_runtime: Mutex<tokio::runtime::Runtime>,
    request_ids: Mutex<HashMap<SequenceId, RequestId>>,
}

impl RemotePipelineRuntime {
    pub fn new(config: RemotePipelineRuntimeConfig) -> Result<Self> {
        if config.transport_backend != TransportBackend::Tcp {
            return Err(anyhow!(
                "remote pipeline runtime currently supports only tcp backend, got {}",
                config.transport_backend
            ));
        }
        if config.stage_peers.is_empty() {
            return Err(anyhow!(
                "remote pipeline runtime requires at least one stage peer"
            ));
        }
        let io_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| anyhow!("failed to build remote pipeline runtime: {err}"))?;
        let transport = io_runtime.block_on(async {
            let transport = Arc::new(
                TcpTransport::bind(TcpTransportConfig {
                    bind_address: config.bind_address.clone(),
                    ..Default::default()
                })
                .await?,
            );
            transport.connect(&config.stage_peers).await?;
            Ok::<_, anyhow::Error>(transport)
        })?;
        Ok(Self {
            stage_peers: config.stage_peers,
            transport,
            io_runtime: Mutex::new(io_runtime),
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

    fn exchange(&self, peer: &str, request: RemoteStageCommand) -> Result<RemoteStageResponse> {
        let transport = Arc::clone(&self.transport);
        let peer = peer.to_string();
        let reply_addr = self.transport.local_addr()?;
        self.io_runtime
            .lock()
            .expect("remote pipeline runtime poisoned")
            .block_on(async move {
                let payload = serde_json::to_vec(&RemoteStageRequest {
                    reply_addr,
                    command: request,
                })?;
                transport
                    .send(
                        &peer,
                        TransportMessage::Control {
                            operation: PIPELINE_STAGE_COMMAND_OPERATION.to_string(),
                            payload: payload.into(),
                        },
                    )
                    .await?;

                loop {
                    let (_from, msg) = transport.recv().await?;
                    let TransportMessage::Control { operation, payload } = msg else {
                        continue;
                    };
                    if operation != PIPELINE_STAGE_RESPONSE_OPERATION {
                        continue;
                    }
                    return serde_json::from_slice::<RemoteStageResponse>(&payload)
                        .map_err(Into::into);
                }
            })
    }

    fn broadcast_ack(&self, request: RemoteStageCommand) -> Result<()> {
        for peer in &self.stage_peers {
            match self.exchange(peer, request.clone())? {
                RemoteStageResponse::Ack { .. } => {}
                RemoteStageResponse::Error { message, .. } => {
                    bail!("remote stage {} failed request: {}", peer, message);
                }
                other => {
                    bail!(
                        "unexpected remote stage response from {}: {:?}",
                        peer,
                        other
                    );
                }
            }
        }
        Ok(())
    }

    pub fn probe_stages(&self) -> Result<Vec<RemoteStageResponse>> {
        let mut states = Vec::with_capacity(self.stage_peers.len());
        for peer in &self.stage_peers {
            match self.exchange(peer, RemoteStageCommand::ProbeState)? {
                state @ RemoteStageResponse::State { .. } => states.push(state),
                RemoteStageResponse::Error { message, .. } => {
                    bail!("remote stage {} failed probe: {}", peer, message);
                }
                other => bail!("unexpected remote stage probe response: {:?}", other),
            }
        }
        Ok(states)
    }
}

impl PipelineModelRuntime for RemotePipelineRuntime {
    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        let request_id = RequestId::from_string(format!("pp-seq-{}", seq_id.as_u64()))
            .expect("sequence-derived request id must be valid");
        self.request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .insert(seq_id, request_id.clone());
        self.broadcast_ack(RemoteStageCommand::PrepareSequence {
            request_id: request_id.as_str().to_string(),
        })
        .expect("failed to prepare remote pipeline sequence state");
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        let Some(request_id) = self
            .request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .remove(&seq_id)
        else {
            return;
        };
        let _ = self.broadcast_ack(RemoteStageCommand::ReleaseSequence {
            request_id: request_id.as_str().to_string(),
        });
    }

    fn forward_sequence(
        &self,
        seq_id: SequenceId,
        input_ids: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let request_id = self.request_id_for(seq_id);
        let response = self.exchange(
            &self.stage_peers[0],
            RemoteStageCommand::RunEntry {
                request_id: request_id.as_str().to_string(),
                micro_batch_id: 0,
                input_ids: serialize_mlx_array(input_ids)?,
                attention_mask: mask.map(serialize_mlx_array).transpose()?,
            },
        )?;
        match response {
            RemoteStageResponse::RunResult { logits, .. } => deserialize_wire_tensor(&logits),
            RemoteStageResponse::Error { message, .. } => Err(anyhow!(message)),
            other => Err(anyhow!("unexpected remote forward response: {:?}", other)),
        }
    }

    fn forward_batched(
        &self,
        seq_ids: &[SequenceId],
        input_ids: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape.get(1).copied().unwrap_or(1);
        let mut ordered = seq_ids.iter().enumerate();
        let Some((first_idx, first_seq_id)) = ordered.next() else {
            return Err(anyhow!(
                "pipeline batched decode received an empty request set"
            ));
        };
        let mut merged = self.forward_sequence(
            *first_seq_id,
            slice(
                input_ids,
                &[first_idx as i32, 0],
                &[first_idx as i32 + 1, seq_len],
            )
            .as_ref()
            .unwrap(),
            None,
        )?;
        for (idx, seq_id) in ordered {
            let logits = self.forward_sequence(
                *seq_id,
                slice(input_ids, &[idx as i32, 0], &[idx as i32 + 1, seq_len])
                    .as_ref()
                    .unwrap(),
                None,
            )?;
            merged = concatenate(&merged, &logits, 0);
        }
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_runtime_rejects_empty_peer_set() {
        let err = RemotePipelineRuntime::new(RemotePipelineRuntimeConfig {
            stage_peers: Vec::new(),
            transport_backend: TransportBackend::Tcp,
            bind_address: "127.0.0.1:0".to_string(),
        })
        .err()
        .expect("empty stage peer config must fail");
        assert!(err.to_string().contains("at least one stage peer"));
    }

    #[test]
    fn remote_runtime_rejects_unsupported_backend() {
        let err = RemotePipelineRuntime::new(RemotePipelineRuntimeConfig {
            stage_peers: vec!["127.0.0.1:20000".to_string()],
            transport_backend: TransportBackend::Thunderbolt,
            bind_address: "127.0.0.1:0".to_string(),
        })
        .err()
        .expect("unsupported transport backend must fail");
        assert!(err.to_string().contains("supports only tcp backend"));
    }
}
