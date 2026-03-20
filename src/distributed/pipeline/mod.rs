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

//! Pipeline parallelism layer partitioning, configuration, activation transfer,
//! micro-batching, scheduling, and metrics.
//!
//! Provides the algorithm and types for distributing model layers across
//! multiple devices in a pipeline-parallel topology:
//!
//! - [`ModelProfile`] — describes a model's layer count, parameter sizes,
//!   and embedding/lm_head costs
//! - [`DeviceSpec`] — per-device available memory and compute capability
//! - [`StageAssignment`] — the output of partitioning: which layers go where
//! - [`PartitionConfig`] — auto vs. manual partition specification
//! - [`auto_partition`] — memory-proportional layer assignment algorithm
//! - [`parse_manual_partition`] — parse `--pp-layers 0-15,16-31` syntax
//! - [`validate_partition`] — reject gaps, overlaps, and memory violations
//!
//! Partial-loading support for pipeline stages:
//!
//! - [`LayerFilter`] — describes which model subset a stage needs
//! - [`classify_weight_key`] — categorise a weight key (layer/embedding/lm_head/norm/other)
//! - [`SafeTensorsIndex`] — parse `model.safetensors.index.json` to map keys → shard files
//! - [`filter_weight_map`] — drop unneeded tensors from an already-loaded weight map
//! - [`estimate_partial_memory`] / [`validate_partial_memory`] — memory budget helpers
//!
//! Activation transfer between pipeline stages:
//!
//! - [`ActivationMessage`] — structured payload with tensor, mask, position IDs
//! - [`ActivationSender`] / [`ActivationReceiver`] — async channels with back-pressure
//! - [`PipelineChannel`] — bidirectional channel between adjacent stages
//! - [`StageLink`] / [`build_pipeline_links`] — connect N stages into a pipeline
//!
//! Micro-batching and pipeline schedule:
//!
//! - [`MicroBatchSpec`] / [`MicroBatch`] — micro-batch splitting and tracking
//! - [`PipelineSchedule`] — trait for pipeline schedule implementations
//! - [`GPipeSchedule`] — GPipe-style forward-all-then-collect schedule
//! - [`PipelineConfig`] — schedule configuration (stages, micro-batch size)
//! - [`ScheduleAction`] — actions emitted by the schedule to drive execution
//!
//! Pipeline metrics:
//!
//! - [`StageMetrics`] — per-stage timing breakdown
//! - [`PipelineMetrics`] — bubble ratio, utilization, latency breakdown
//! - [`MetricsCollector`] — accumulates metrics across pipeline steps

pub mod activation_transfer;
pub mod metrics;
pub mod micro_batch;
pub mod partial_loading;
pub mod partition;
pub mod schedule;

pub use activation_transfer::{
    ActivationMessage, ActivationReceiver, ActivationSender, ChannelConfig, PipelineChannel,
    StageEndpoint, StageLink, activation_channel, activation_latency, build_pipeline_links,
    validate_activation,
};
pub use metrics::{
    MetricsCollector as PipelineMetricsCollector, MetricsSummary, PipelineMetrics, StageMetrics,
};
pub use micro_batch::{
    MicroBatch, MicroBatchSpec, split_into_micro_batches, suggested_micro_batch_size,
};
pub use partial_loading::{
    LayerFilter, SafeTensorsIndex, WeightClass, classify_weight_key, estimate_partial_memory,
    filter_weight_keys, filter_weight_map, identify_required_shards, should_load_key,
    validate_partial_memory,
};
pub use partition::{
    DeviceSpec, ModelProfile, PartitionConfig, StageAssignment, auto_partition,
    build_manual_assignments, parse_manual_partition, validate_memory_fit, validate_partition,
};
pub use schedule::{
    GPipeSchedule, PipelineConfig, PipelineSchedule, ScheduleAction, create_gpipe_schedule,
};
