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

//! Distributed inference: node discovery, cluster configuration, and transport.
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
//! - [`bench`] — throughput and latency benchmarking harness

pub mod bench;
pub mod config;
pub mod connection_pool;
pub mod discovery;
pub mod registry;
pub mod tcp_transport;
pub mod thunderbolt_transport;
pub mod transport;

pub use config::{ClusterConfig, ClusterMeta, NodeConfig, NodeResources, NodeRole};
pub use connection_pool::{ConnectionPool, PoolConfig, PoolStats};
pub use discovery::{initialize_distributed, log_cluster_topology, probe_peers};
pub use registry::{NodeRegistry, NodeStatus, RegisteredNode};
pub use tcp_transport::{TcpTransport, TcpTransportConfig};
pub use thunderbolt_transport::{ThunderboltTransport, ThunderboltTransportConfig};
pub use transport::{Transport, TransportBackend, TransportMessage};
