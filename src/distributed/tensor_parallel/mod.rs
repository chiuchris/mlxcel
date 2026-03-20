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

//! Tensor parallelism: weight sharding strategy and configuration.
//!
//! This module provides the types and algorithms for distributing model weights
//! across multiple tensor-parallel ranks:
//!
//! - [`ShardStrategy`] — how a weight tensor is split (column/row/expert/vocab/replicated)
//! - [`CommPattern`] — communication required after each sharded operation
//! - [`LayerShardPlan`] — per-weight sharding metadata
//! - [`ModelShardPlan`] — complete model shard plan with layer expansion
//! - [`ShardConfig`] — user-configurable TP options (tp_size, MoE mode, embedding mode)
//! - [`MoeShardMode`] — expert-parallel vs within-expert sharding
//! - [`EmbeddingMode`] — vocab-parallel vs replicated embedding/LM head
//! - [`generate_shard_plan`] — architecture-aware shard plan generator

pub mod config;
pub mod plan_generator;
pub mod shard_strategy;

pub use config::{EmbeddingMode, MoeShardMode, ShardConfig};
pub use plan_generator::generate_shard_plan;
pub use shard_strategy::{CommPattern, LayerShardPlan, ModelShardPlan, ShardStrategy};
