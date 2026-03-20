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

//! Distributed inference: node discovery, cluster configuration, transport,
//! health monitoring, and request scheduling.
//!
//! This module provides the building blocks for multi-node inference:
//!
//! - [`config`] — `NodeRole`, `ClusterConfig`, and TOML parsing
//! - [`registry`] — thread-safe runtime node registry
//! - [`discovery`] — static peer discovery and health probing
//! - [`transport`] — abstract transport trait with streaming and RPC modes
//! - [`tcp_transport`] — TCP backend with connection pooling
//! - [`thunderbolt_transport`] — Thunderbolt backend (stubbed)
//! - [`connection_pool`] — connection pooling with reconnection logic
//! - [`heartbeat`] — heartbeat protocol with configurable interval and failure threshold
//! - [`failure_detector`] — threshold-based failure detection integrated with node registry
//! - [`metrics`] — per-node metrics collection (throughput, latency, memory, network)
//! - [`correlation`] — cluster-wide correlation IDs for cross-node request tracing
//! - [`bench`] — throughput and latency benchmarking harness
//! - [`scheduler`] — distributed scheduler coordinator with pluggable routing
//! - [`routing`] — trait-based routing strategies (role-based, load-balanced, round-robin, PP)
//! - [`request_tracker`] — request lifecycle tracking with unique IDs and state transitions
//! - [`backpressure`] — per-node load tracking with configurable thresholds and overflow policies
//! - [`handoff_queue`] — bounded cross-node request handoff queues
//! - [`pipeline`] — layer partitioning strategy and configuration for pipeline parallelism
//! - [`tensor_parallel`] — weight sharding strategy and configuration for tensor parallelism

pub mod backpressure;
pub mod bench;
pub mod config;
pub mod connection_pool;
pub mod correlation;
pub mod discovery;
pub mod failure_detector;
pub mod handoff_queue;
pub mod heartbeat;
pub mod metrics;
#[cfg(any(test, feature = "test-utils"))]
pub mod mock_transport;
pub mod pipeline;
pub mod registry;
pub mod request_tracker;
pub mod routing;
pub mod scheduler;
pub mod tcp_transport;
pub mod tensor_chunked;
pub mod tensor_compress;
pub mod tensor_parallel;
pub mod tensor_protocol;
pub mod tensor_quantize;
pub mod tensor_serialize;
#[cfg(any(test, feature = "test-utils"))]
pub mod test_harness;
pub mod thunderbolt_transport;
pub mod transport;

pub use backpressure::{
    BackpressureConfig, BackpressureMonitor, BackpressurePolicy, BackpressureSignal, LoadLevel,
};
pub use config::{ClusterConfig, ClusterMeta, NodeConfig, NodeResources, NodeRole};
pub use connection_pool::{ConnectionPool, PoolConfig, PoolStats};
pub use correlation::{CorrelationId, RequestContext};
pub use discovery::{initialize_distributed, log_cluster_topology, probe_peers};
pub use failure_detector::{FailureDetector, FailureDetectorConfig, FailureEvent};
pub use handoff_queue::{
    HandoffItem, HandoffQueue, HandoffQueueConfig, HandoffQueueManager, OverflowPolicy, QueueStats,
};
pub use heartbeat::{HEARTBEAT_OPERATION, HeartbeatConfig, HeartbeatPayload, HeartbeatService};
pub use metrics::{
    ClusterMetrics, LatencyPercentiles, MetricsCollector, MetricsConfig, NodeMetrics,
};
pub use pipeline::{
    DeviceSpec, ModelProfile, PartitionConfig, StageAssignment, auto_partition,
    build_manual_assignments, parse_manual_partition, validate_memory_fit, validate_partition,
};
pub use registry::{NodeRegistry, NodeStatus, RegisteredNode};
pub use request_tracker::{
    RequestId, RequestLifecycle, RequestState, RequestTracker, RequestTrackerConfig,
};
pub use routing::{
    LoadBalancedRouter, NodeCandidate, PipelineStageRouter, RoleBasedRouter, RoundRobinRouter,
    RoutingDecision, RoutingRequest, RoutingStrategy,
};
pub use scheduler::{CoordinationMode, Scheduler, SchedulerConfig};
pub use tcp_transport::{TcpTransport, TcpTransportConfig};
pub use tensor_chunked::{ChunkAssembler, ChunkedTensor, ChunkedTransferConfig};
pub use tensor_parallel::{
    CommPattern, EmbeddingMode, LayerShardPlan, ModelShardPlan, MoeShardMode, ShardConfig,
    ShardStrategy, generate_shard_plan,
};
pub use tensor_protocol::{
    PROTOCOL_VERSION, QuantizationMode, TensorDtype, TensorFlags, TensorHeader, TensorKind,
};
pub use tensor_serialize::{
    DeserializedTensor, SerializeOptions, deserialize_tensor, serialize_tensor,
    serialize_tensor_to_bytes,
};
pub use thunderbolt_transport::{ThunderboltTransport, ThunderboltTransportConfig};
pub use transport::{Transport, TransportBackend, TransportMessage};
