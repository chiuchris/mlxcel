use super::*;
use crate::distributed::config::{ClusterConfig, NodeRole};

fn sample_config() -> ClusterConfig {
    let toml_str = r#"
[cluster]
name = "test"

[[nodes]]
id = "local"
address = "127.0.0.1:8080"
role = "hybrid"

[[nodes]]
id = "peer-1"
address = "192.168.1.2:8080"
role = "decode"
"#;
    ClusterConfig::from_toml(toml_str).unwrap()
}

#[test]
fn registry_from_config() {
    let config = sample_config();
    let registry = NodeRegistry::from_config(&config, "local");

    assert_eq!(registry.node_count(), 2);
    assert_eq!(registry.local_node_id(), "local");

    let local = registry.get_node("local").unwrap();
    assert_eq!(local.status, NodeStatus::Online);

    let peer = registry.get_node("peer-1").unwrap();
    assert_eq!(peer.status, NodeStatus::Joining);
}

#[test]
fn set_node_status() {
    let config = sample_config();
    let registry = NodeRegistry::from_config(&config, "local");

    assert!(registry.set_node_status("peer-1", NodeStatus::Online));
    assert_eq!(
        registry.get_node("peer-1").unwrap().status,
        NodeStatus::Online
    );

    assert!(!registry.set_node_status("nonexistent", NodeStatus::Online));
}

#[test]
fn upsert_and_remove() {
    let config = sample_config();
    let registry = NodeRegistry::from_config(&config, "local");

    let new_node = NodeConfig {
        id: "new-node".to_string(),
        address: "10.0.0.5:8080".parse().unwrap(),
        role: NodeRole::Prefill,
        stage: None,
        rank: None,
        resources: NodeResources::default(),
    };

    registry.upsert_node(new_node, NodeStatus::Online);
    assert_eq!(registry.node_count(), 3);
    assert!(registry.get_node("new-node").is_some());

    let removed = registry.remove_node("new-node");
    assert!(removed.is_some());
    assert_eq!(registry.node_count(), 2);
}

#[test]
fn nodes_with_role_filter() {
    let config = sample_config();
    let registry = NodeRegistry::from_config(&config, "local");

    let decode_nodes = registry.nodes_with_role(NodeRole::Decode);
    assert_eq!(decode_nodes.len(), 1);
    assert_eq!(decode_nodes[0].config.id, "peer-1");

    let prefill_nodes = registry.nodes_with_role(NodeRole::Prefill);
    assert!(prefill_nodes.is_empty());
}

#[test]
fn peer_addresses_excludes_local() {
    let config = sample_config();
    let registry = NodeRegistry::from_config(&config, "local");

    let peers = registry.peer_addresses();
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0], "192.168.1.2:8080".parse().unwrap());
}

#[test]
fn topology_summary_includes_all() {
    let config = sample_config();
    let registry = NodeRegistry::from_config(&config, "local");

    let summary = registry.topology_summary();
    assert!(summary.contains("local"));
    assert!(summary.contains("peer-1"));
    assert!(summary.contains("(local)"));
}

#[test]
fn update_resources() {
    let config = sample_config();
    let registry = NodeRegistry::from_config(&config, "local");

    let res = NodeResources {
        memory_bytes: 16_000_000_000,
        compute_units: 38,
    };
    assert!(registry.update_resources("local", res.clone()));

    let node = registry.get_node("local").unwrap();
    assert_eq!(node.config.resources.memory_bytes, 16_000_000_000);
    assert_eq!(node.config.resources.compute_units, 38);
}

#[test]
fn concurrent_access() {
    let config = sample_config();
    let registry = NodeRegistry::from_config(&config, "local");

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let reg = registry.clone();
            std::thread::spawn(move || {
                // Readers
                let _ = reg.all_nodes();
                let _ = reg.node_count();
                let _ = reg.peer_addresses();

                // Writer
                reg.set_node_status(
                    "peer-1",
                    if i % 2 == 0 {
                        NodeStatus::Online
                    } else {
                        NodeStatus::Unreachable
                    },
                );
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Registry should still be consistent
    assert_eq!(registry.node_count(), 2);
}

// -- 2D (PP x TP) parallelism ------------------------------------------------

fn pp_tp_2x2_config() -> ClusterConfig {
    let toml_str = r#"
[cluster]
name = "pptp"
pipeline_parallel_size = 2
tensor_parallel_size = 2

[[nodes]]
id = "s0-r0"
address = "10.0.0.10:8080"
role = "pipeline_tensor_parallel"
stage = 0
rank = 0

[[nodes]]
id = "s0-r1"
address = "10.0.0.11:8080"
role = "pipeline_tensor_parallel"
stage = 0
rank = 1

[[nodes]]
id = "s1-r0"
address = "10.0.0.20:8080"
role = "pipeline_tensor_parallel"
stage = 1
rank = 0

[[nodes]]
id = "s1-r1"
address = "10.0.0.21:8080"
role = "pipeline_tensor_parallel"
stage = 1
rank = 1
"#;
    ClusterConfig::from_toml(toml_str).unwrap()
}

#[test]
fn find_pp_tp_node_returns_exact_intersection() {
    let config = pp_tp_2x2_config();
    let registry = NodeRegistry::from_config(&config, "s0-r0");
    assert_eq!(registry.find_pp_tp_node(0, 0).unwrap().config.id, "s0-r0");
    assert_eq!(registry.find_pp_tp_node(1, 1).unwrap().config.id, "s1-r1");
    assert!(registry.find_pp_tp_node(2, 0).is_none());
    assert!(registry.find_pp_tp_node(0, 5).is_none());
}

#[test]
fn nodes_at_stage_returns_all_ranks() {
    let config = pp_tp_2x2_config();
    let registry = NodeRegistry::from_config(&config, "s0-r0");
    let stage0 = registry.nodes_at_stage(0);
    assert_eq!(stage0.len(), 2);
    assert_eq!(stage0[0].config.id, "s0-r0");
    assert_eq!(stage0[1].config.id, "s0-r1");
    let stage1 = registry.nodes_at_stage(1);
    assert_eq!(stage1.len(), 2);
    assert_eq!(stage1[0].config.id, "s1-r0");
    assert_eq!(stage1[1].config.id, "s1-r1");
}

#[test]
fn nodes_at_rank_returns_all_stages() {
    let config = pp_tp_2x2_config();
    let registry = NodeRegistry::from_config(&config, "s0-r0");
    let rank0 = registry.nodes_at_rank(0);
    assert_eq!(rank0.len(), 2);
    assert_eq!(rank0[0].config.id, "s0-r0");
    assert_eq!(rank0[1].config.id, "s1-r0");
}

#[test]
fn local_pp_tp_coords_reflect_local_node() {
    let config = pp_tp_2x2_config();
    let registry = NodeRegistry::from_config(&config, "s1-r0");
    assert_eq!(registry.local_pp_tp_coords(), Some((1, 0)));
}

#[test]
fn local_pp_tp_coords_none_for_legacy_topology() {
    let config = sample_config();
    let registry = NodeRegistry::from_config(&config, "local");
    assert!(registry.local_pp_tp_coords().is_none());
}
