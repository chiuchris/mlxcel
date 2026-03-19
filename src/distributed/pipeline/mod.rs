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

//! Pipeline parallelism layer partitioning and configuration.
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

pub mod partition;

pub use partition::{
    DeviceSpec, ModelProfile, PartitionConfig, StageAssignment, auto_partition,
    build_manual_assignments, parse_manual_partition, validate_memory_fit, validate_partition,
};
