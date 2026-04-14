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

//! Distributed cluster configuration types.
//!
//! Defines the node roles, per-node configuration, and the overall cluster
//! configuration that can be loaded from a TOML file or assembled from CLI
//! arguments.

use std::net::SocketAddr;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::TransportBackend;

/// Maximum allowed length for node IDs and cluster names to prevent abuse
/// from untrusted TOML configuration files.
const MAX_ID_LENGTH: usize = 128;

/// Role a node plays in the distributed inference cluster.
///
/// A single node may serve one of several purposes depending on the
/// parallelism strategy (pipeline parallel, tensor parallel, disaggregated
/// inference, or a hybrid combination).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum NodeRole {
    /// Handles prompt prefill only (disaggregated inference).
    Prefill,
    /// Handles autoregressive decode only (disaggregated inference).
    Decode,
    /// Owns one stage of a pipeline-parallel topology.
    PipelineStage,
    /// Participates as a rank in tensor-parallel execution.
    TensorParallelRank,
    /// General-purpose node that can perform any function.
    Hybrid,
}

impl std::fmt::Display for NodeRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Prefill => write!(f, "prefill"),
            Self::Decode => write!(f, "decode"),
            Self::PipelineStage => write!(f, "pipeline_stage"),
            Self::TensorParallelRank => write!(f, "tensor_parallel_rank"),
            Self::Hybrid => write!(f, "hybrid"),
        }
    }
}

impl std::str::FromStr for NodeRole {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "prefill" => Ok(Self::Prefill),
            "decode" => Ok(Self::Decode),
            "pipeline_stage" | "pipeline" => Ok(Self::PipelineStage),
            "tensor_parallel_rank" | "tensor_parallel" | "tp" => Ok(Self::TensorParallelRank),
            "hybrid" => Ok(Self::Hybrid),
            other => anyhow::bail!(
                "unknown node role '{other}'; expected one of: prefill, decode, \
                 pipeline_stage, tensor_parallel_rank, hybrid"
            ),
        }
    }
}

/// Resource constraints and capabilities for a single node.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeResources {
    /// Available memory in bytes (0 = auto-detect).
    #[serde(default)]
    pub memory_bytes: u64,
    /// Number of compute units (GPU cores / neural engine cores; 0 = auto-detect).
    #[serde(default)]
    pub compute_units: u32,
}

/// Configuration for a single node in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Unique identifier for this node.
    pub id: String,
    /// Network address (host:port) this node listens on or can be reached at.
    pub address: SocketAddr,
    /// Role this node plays in the cluster.
    pub role: NodeRole,
    /// Pipeline stage index (only meaningful when `role == PipelineStage`).
    #[serde(default)]
    pub stage: Option<u32>,
    /// Tensor-parallel rank (only meaningful when `role == TensorParallelRank`).
    #[serde(default)]
    pub rank: Option<u32>,
    /// Resource constraints / capabilities.
    #[serde(default)]
    pub resources: NodeResources,
}

/// Top-level cluster configuration.
///
/// Can be loaded from a TOML file or constructed programmatically from CLI
/// arguments.
///
/// ## TOML Example
///
/// ```toml
/// [cluster]
/// name = "my-cluster"
/// tensor_parallel_size = 1
/// pipeline_parallel_size = 2
///
/// [[nodes]]
/// id = "node-0"
/// address = "192.168.1.10:8080"
/// role = "pipeline_stage"
/// stage = 0
///
/// [[nodes]]
/// id = "node-1"
/// address = "192.168.1.11:8080"
/// role = "pipeline_stage"
/// stage = 1
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Cluster-level metadata.
    pub cluster: ClusterMeta,
    /// All nodes participating in this cluster.
    pub nodes: Vec<NodeConfig>,
}

/// Cluster-level metadata embedded in [`ClusterConfig`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMeta {
    /// Human-readable cluster name.
    #[serde(default = "default_cluster_name")]
    pub name: String,
    /// Tensor-parallel world size (number of TP ranks).
    #[serde(default = "default_one")]
    pub tensor_parallel_size: u32,
    /// Pipeline-parallel depth (number of PP stages).
    #[serde(default = "default_one")]
    pub pipeline_parallel_size: u32,
    /// Inter-node transport backend used for remote pipeline traffic.
    #[serde(default)]
    pub transport_backend: TransportBackend,
}

fn default_cluster_name() -> String {
    "mlxcel-cluster".to_string()
}

fn default_one() -> u32 {
    1
}

impl Default for ClusterMeta {
    fn default() -> Self {
        Self {
            name: default_cluster_name(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            transport_backend: TransportBackend::Tcp,
        }
    }
}

impl ClusterConfig {
    /// Load a cluster configuration from a TOML file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read cluster config: {path:?}"))?;
        Self::from_toml(&content)
    }

    /// Parse a cluster configuration from a TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self> {
        let config: Self =
            toml::from_str(toml_str).context("failed to parse cluster configuration TOML")?;
        config.validate()?;
        Ok(config)
    }

    /// Build a minimal single-node cluster config from CLI arguments.
    ///
    /// Used when the user provides `--node-role`, `--node-id`, and optionally
    /// `--peers` instead of a full TOML config file.
    ///
    /// Filters out any peer address that duplicates the local node's address
    /// to prevent misconfiguration when the user accidentally includes their
    /// own address in `--peers`.
    pub fn from_cli(
        node_id: String,
        address: SocketAddr,
        role: NodeRole,
        peers: Vec<SocketAddr>,
    ) -> Self {
        let mut nodes = vec![NodeConfig {
            id: node_id,
            address,
            role,
            stage: None,
            rank: None,
            resources: NodeResources::default(),
        }];

        for (i, peer_addr) in peers.into_iter().enumerate() {
            // Skip peers whose address matches the local node to avoid duplicates.
            if peer_addr == address {
                continue;
            }
            nodes.push(NodeConfig {
                id: format!("peer-{i}"),
                address: peer_addr,
                role: NodeRole::Hybrid,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            });
        }

        Self {
            cluster: ClusterMeta::default(),
            nodes,
        }
    }

    /// Validate internal consistency of the configuration.
    fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            anyhow::bail!("cluster config must contain at least one node");
        }

        // Validate cluster name length and content.
        if self.cluster.name.len() > MAX_ID_LENGTH {
            anyhow::bail!("cluster name exceeds maximum length of {MAX_ID_LENGTH} characters");
        }
        if self.cluster.name.chars().any(|c| c.is_control()) {
            anyhow::bail!("cluster name contains invalid control characters");
        }

        // Check for duplicate node IDs and validate ID format.
        let mut seen_ids = std::collections::HashSet::new();
        for node in &self.nodes {
            if node.id.is_empty() {
                anyhow::bail!("node id must not be empty");
            }
            if node.id.len() > MAX_ID_LENGTH {
                anyhow::bail!("node id exceeds maximum length of {MAX_ID_LENGTH} characters");
            }
            if node.id.chars().any(|c| c.is_control()) {
                anyhow::bail!("node id '{}' contains invalid control characters", node.id);
            }
            if !seen_ids.insert(&node.id) {
                anyhow::bail!("duplicate node id '{}' in cluster config", node.id);
            }
        }

        // Check for duplicate addresses.
        let mut seen_addrs = std::collections::HashSet::new();
        for node in &self.nodes {
            if !seen_addrs.insert(node.address) {
                anyhow::bail!("duplicate address '{}' in cluster config", node.address);
            }
        }

        let pipeline_stage_nodes: Vec<&NodeConfig> = self
            .nodes
            .iter()
            .filter(|node| node.role == NodeRole::PipelineStage)
            .collect();
        if !pipeline_stage_nodes.is_empty() || self.cluster.pipeline_parallel_size > 1 {
            anyhow::ensure!(
                self.cluster.pipeline_parallel_size > 0,
                "pipeline_parallel_size must be greater than 0 when pipeline stages are configured"
            );
            anyhow::ensure!(
                pipeline_stage_nodes.len() == self.cluster.pipeline_parallel_size as usize,
                "cluster config declares pipeline_parallel_size={} but defines {} pipeline_stage nodes",
                self.cluster.pipeline_parallel_size,
                pipeline_stage_nodes.len()
            );

            let mut seen_stages = std::collections::HashSet::new();
            for node in pipeline_stage_nodes {
                let stage = node.stage.ok_or_else(|| {
                    anyhow::anyhow!(
                        "pipeline stage node '{}' is missing required 'stage' index",
                        node.id
                    )
                })?;
                anyhow::ensure!(
                    stage < self.cluster.pipeline_parallel_size,
                    "pipeline stage node '{}' has out-of-range stage index {} for pipeline_parallel_size={}",
                    node.id,
                    stage,
                    self.cluster.pipeline_parallel_size
                );
                anyhow::ensure!(
                    seen_stages.insert(stage),
                    "duplicate pipeline stage index {} in cluster config",
                    stage
                );
            }
            for expected_stage in 0..self.cluster.pipeline_parallel_size {
                anyhow::ensure!(
                    seen_stages.contains(&expected_stage),
                    "missing pipeline stage index {} in cluster config",
                    expected_stage
                );
            }
        }

        for node in &self.nodes {
            if node.role != NodeRole::PipelineStage
                && let Some(stage) = node.stage
            {
                anyhow::bail!(
                    "node '{}' sets stage={} but role is {}; only pipeline_stage nodes may set stage",
                    node.id,
                    stage,
                    node.role
                );
            }
        }

        Ok(())
    }

    /// Return the node config for the given node ID, if present.
    pub fn find_node(&self, id: &str) -> Option<&NodeConfig> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Return pipeline-stage nodes sorted by stage index.
    pub fn pipeline_stage_nodes(&self) -> Vec<&NodeConfig> {
        let mut nodes: Vec<&NodeConfig> = self
            .nodes
            .iter()
            .filter(|node| node.role == NodeRole::PipelineStage)
            .collect();
        nodes.sort_by_key(|node| node.stage.unwrap_or(u32::MAX));
        nodes
    }

    /// Return the pipeline-stage node for the given stage index, if present.
    pub fn pipeline_stage_node(&self, stage: u32) -> Option<&NodeConfig> {
        self.nodes
            .iter()
            .find(|node| node.role == NodeRole::PipelineStage && node.stage == Some(stage))
    }

    /// Pretty-print a summary of the cluster topology.
    pub fn topology_summary(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(out, "Cluster: {}", self.cluster.name);
        let _ = writeln!(
            out,
            "  TP size: {}, PP size: {}",
            self.cluster.tensor_parallel_size, self.cluster.pipeline_parallel_size
        );
        let _ = writeln!(out, "  Nodes ({}):", self.nodes.len());
        for node in &self.nodes {
            let _ = writeln!(out, "    - {} @ {} [{}]", node.id, node.address, node.role);
        }
        out
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
