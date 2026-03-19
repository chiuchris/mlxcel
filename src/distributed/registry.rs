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

//! Thread-safe node registry for tracking cluster membership.
//!
//! The registry is the runtime source of truth for which nodes are currently
//! participating in the cluster, their roles, and their capabilities. It
//! supports dynamic updates so nodes can join and leave without a restart.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};

use super::config::{ClusterConfig, ClusterMeta, NodeConfig, NodeResources, NodeRole};

/// Health status of a registered node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is reachable and ready to serve.
    Online,
    /// Node has not responded to recent health checks.
    Unreachable,
    /// Node is in the process of joining (not yet ready).
    Joining,
}

impl std::fmt::Display for NodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Online => write!(f, "online"),
            Self::Unreachable => write!(f, "unreachable"),
            Self::Joining => write!(f, "joining"),
        }
    }
}

/// Runtime entry for a registered node, combining static config with dynamic
/// state such as health status.
#[derive(Debug, Clone)]
pub struct RegisteredNode {
    /// Static configuration for this node.
    pub config: NodeConfig,
    /// Current health status.
    pub status: NodeStatus,
}

/// Thread-safe registry of nodes in the cluster.
///
/// Designed for concurrent reads from request handlers and infrequent writes
/// from the discovery / health-check subsystem.
#[derive(Clone)]
pub struct NodeRegistry {
    inner: Arc<RwLock<RegistryInner>>,
}

struct RegistryInner {
    /// Cluster-level metadata.
    meta: ClusterMeta,
    /// Node ID -> registered node.
    nodes: HashMap<String, RegisteredNode>,
    /// ID of the local node (the node running this process).
    local_node_id: String,
}

impl NodeRegistry {
    /// Create a new registry seeded from a [`ClusterConfig`].
    ///
    /// All nodes start in [`NodeStatus::Joining`] except the local node,
    /// which is set to [`NodeStatus::Online`].
    pub fn from_config(config: &ClusterConfig, local_node_id: &str) -> Self {
        let mut nodes = HashMap::with_capacity(config.nodes.len());
        for node_cfg in &config.nodes {
            let status = if node_cfg.id == local_node_id {
                NodeStatus::Online
            } else {
                NodeStatus::Joining
            };
            nodes.insert(
                node_cfg.id.clone(),
                RegisteredNode {
                    config: node_cfg.clone(),
                    status,
                },
            );
        }
        Self {
            inner: Arc::new(RwLock::new(RegistryInner {
                meta: config.cluster.clone(),
                nodes,
                local_node_id: local_node_id.to_string(),
            })),
        }
    }

    /// Return the local node's ID.
    pub fn local_node_id(&self) -> String {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .local_node_id
            .clone()
    }

    /// Return the number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .nodes
            .len()
    }

    /// Look up a node by ID.
    pub fn get_node(&self, id: &str) -> Option<RegisteredNode> {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .nodes
            .get(id)
            .cloned()
    }

    /// Return a snapshot of all registered nodes.
    pub fn all_nodes(&self) -> Vec<RegisteredNode> {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .nodes
            .values()
            .cloned()
            .collect()
    }

    /// Return nodes filtered by role.
    pub fn nodes_with_role(&self, role: NodeRole) -> Vec<RegisteredNode> {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .nodes
            .values()
            .filter(|n| n.config.role == role)
            .cloned()
            .collect()
    }

    /// Return the cluster metadata.
    pub fn cluster_meta(&self) -> ClusterMeta {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .meta
            .clone()
    }

    /// Update the status of a node. Returns `true` if the node was found.
    pub fn set_node_status(&self, id: &str, status: NodeStatus) -> bool {
        let mut inner = self.inner.write().expect("registry lock poisoned");
        if let Some(node) = inner.nodes.get_mut(id) {
            node.status = status;
            true
        } else {
            false
        }
    }

    /// Register a new node or update an existing one.
    pub fn upsert_node(&self, config: NodeConfig, status: NodeStatus) {
        let mut inner = self.inner.write().expect("registry lock poisoned");
        inner
            .nodes
            .insert(config.id.clone(), RegisteredNode { config, status });
    }

    /// Remove a node from the registry. Returns the removed node if present.
    pub fn remove_node(&self, id: &str) -> Option<RegisteredNode> {
        let mut inner = self.inner.write().expect("registry lock poisoned");
        inner.nodes.remove(id)
    }

    /// Return the address of each peer (all nodes except the local one).
    pub fn peer_addresses(&self) -> Vec<SocketAddr> {
        let inner = self.inner.read().expect("registry lock poisoned");
        inner
            .nodes
            .values()
            .filter(|n| n.config.id != inner.local_node_id)
            .map(|n| n.config.address)
            .collect()
    }

    /// Return a human-readable cluster topology string.
    pub fn topology_summary(&self) -> String {
        use std::fmt::Write;
        let inner = self.inner.read().expect("registry lock poisoned");
        let mut out = String::new();
        let _ = writeln!(out, "Cluster: {}", inner.meta.name);
        let _ = writeln!(
            out,
            "  TP size: {}, PP size: {}",
            inner.meta.tensor_parallel_size, inner.meta.pipeline_parallel_size
        );
        let _ = writeln!(out, "  Local node: {}", inner.local_node_id);
        let _ = writeln!(out, "  Nodes ({}):", inner.nodes.len());
        for node in inner.nodes.values() {
            let local_tag = if node.config.id == inner.local_node_id {
                " (local)"
            } else {
                ""
            };
            let _ = writeln!(
                out,
                "    - {} @ {} [{}] status={}{local_tag}",
                node.config.id, node.config.address, node.config.role, node.status
            );
        }
        out
    }

    /// Update the resource capabilities for a node. Returns `true` if found.
    pub fn update_resources(&self, id: &str, resources: NodeResources) -> bool {
        let mut inner = self.inner.write().expect("registry lock poisoned");
        if let Some(node) = inner.nodes.get_mut(id) {
            node.config.resources = resources;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
