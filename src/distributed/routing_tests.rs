use super::*;

fn make_candidate(id: &str, role: NodeRole, status: NodeStatus) -> NodeCandidate {
    NodeCandidate {
        node_id: id.to_string(),
        role,
        status,
        metrics: None,
        stage: None,
        rank: None,
    }
}

fn make_candidate_with_load(id: &str, role: NodeRole, active_requests: u32) -> NodeCandidate {
    NodeCandidate {
        node_id: id.to_string(),
        role,
        status: NodeStatus::Online,
        metrics: Some(NodeMetrics {
            active_requests,
            ..NodeMetrics::default()
        }),
        stage: None,
        rank: None,
    }
}

fn make_request(preferred_role: Option<NodeRole>) -> RoutingRequest {
    RoutingRequest {
        request_id: "req-1".to_string(),
        preferred_role,
        preferred_stage: None,
        affinity_node: None,
    }
}

// -- RoleBasedRouter --

#[test]
fn role_based_selects_matching_role() {
    let router = RoleBasedRouter;
    let candidates = vec![
        make_candidate("node-0", NodeRole::Decode, NodeStatus::Online),
        make_candidate("node-1", NodeRole::Prefill, NodeStatus::Online),
    ];
    let request = make_request(Some(NodeRole::Prefill));
    let decision = router.select_node(&request, &candidates).unwrap();
    assert_eq!(decision.target_node, "node-1");
    assert_eq!(decision.strategy, "role-based");
}

#[test]
fn role_based_matches_hybrid_as_wildcard() {
    let router = RoleBasedRouter;
    let candidates = vec![make_candidate(
        "node-0",
        NodeRole::Hybrid,
        NodeStatus::Online,
    )];
    let request = make_request(Some(NodeRole::Prefill));
    let decision = router.select_node(&request, &candidates).unwrap();
    assert_eq!(decision.target_node, "node-0");
}

#[test]
fn role_based_falls_back_when_no_role_preference() {
    let router = RoleBasedRouter;
    let candidates = vec![make_candidate(
        "node-0",
        NodeRole::Decode,
        NodeStatus::Online,
    )];
    let request = make_request(None);
    let decision = router.select_node(&request, &candidates).unwrap();
    assert_eq!(decision.target_node, "node-0");
}

#[test]
fn role_based_skips_unreachable_nodes() {
    let router = RoleBasedRouter;
    let candidates = vec![
        make_candidate("node-0", NodeRole::Prefill, NodeStatus::Unreachable),
        make_candidate("node-1", NodeRole::Prefill, NodeStatus::Online),
    ];
    let request = make_request(Some(NodeRole::Prefill));
    let decision = router.select_node(&request, &candidates).unwrap();
    assert_eq!(decision.target_node, "node-1");
}

#[test]
fn role_based_errors_when_no_online_nodes() {
    let router = RoleBasedRouter;
    let candidates = vec![make_candidate(
        "node-0",
        NodeRole::Prefill,
        NodeStatus::Unreachable,
    )];
    let request = make_request(Some(NodeRole::Prefill));
    assert!(router.select_node(&request, &candidates).is_err());
}

#[test]
fn role_based_respects_affinity() {
    let router = RoleBasedRouter;
    let candidates = vec![
        make_candidate("node-0", NodeRole::Prefill, NodeStatus::Online),
        make_candidate("node-1", NodeRole::Prefill, NodeStatus::Online),
    ];
    let request = RoutingRequest {
        request_id: "req-1".to_string(),
        preferred_role: Some(NodeRole::Prefill),
        preferred_stage: None,
        affinity_node: Some("node-1".to_string()),
    };
    let decision = router.select_node(&request, &candidates).unwrap();
    assert_eq!(decision.target_node, "node-1");
}

// -- LoadBalancedRouter --

#[test]
fn load_balanced_selects_least_loaded() {
    let router = LoadBalancedRouter;
    let candidates = vec![
        make_candidate_with_load("node-0", NodeRole::Hybrid, 5),
        make_candidate_with_load("node-1", NodeRole::Hybrid, 2),
        make_candidate_with_load("node-2", NodeRole::Hybrid, 8),
    ];
    let request = make_request(None);
    let decision = router.select_node(&request, &candidates).unwrap();
    assert_eq!(decision.target_node, "node-1");
}

#[test]
fn load_balanced_filters_by_role() {
    let router = LoadBalancedRouter;
    let candidates = vec![
        make_candidate_with_load("node-0", NodeRole::Decode, 1),
        make_candidate_with_load("node-1", NodeRole::Prefill, 3),
        make_candidate_with_load("node-2", NodeRole::Prefill, 2),
    ];
    let request = make_request(Some(NodeRole::Prefill));
    let decision = router.select_node(&request, &candidates).unwrap();
    assert_eq!(decision.target_node, "node-2");
}

// -- RoundRobinRouter --

#[test]
fn round_robin_distributes_evenly() {
    let router = RoundRobinRouter::new();
    let candidates = vec![
        make_candidate("node-a", NodeRole::Hybrid, NodeStatus::Online),
        make_candidate("node-b", NodeRole::Hybrid, NodeStatus::Online),
    ];
    let request = make_request(None);

    let d1 = router.select_node(&request, &candidates).unwrap();
    let d2 = router.select_node(&request, &candidates).unwrap();
    let d3 = router.select_node(&request, &candidates).unwrap();

    // Should alternate between the two nodes.
    assert_ne!(d1.target_node, d2.target_node);
    assert_eq!(d1.target_node, d3.target_node);
}

#[test]
fn round_robin_errors_when_no_online() {
    let router = RoundRobinRouter::new();
    let candidates = vec![make_candidate(
        "node-0",
        NodeRole::Hybrid,
        NodeStatus::Unreachable,
    )];
    let request = make_request(None);
    assert!(router.select_node(&request, &candidates).is_err());
}

// -- PipelineStageRouter --

#[test]
fn pipeline_stage_selects_correct_stage() {
    let router = PipelineStageRouter;
    let mut c0 = make_candidate("node-0", NodeRole::PipelineStage, NodeStatus::Online);
    c0.stage = Some(0);
    let mut c1 = make_candidate("node-1", NodeRole::PipelineStage, NodeStatus::Online);
    c1.stage = Some(1);

    let request = RoutingRequest {
        request_id: "req-1".to_string(),
        preferred_role: None,
        preferred_stage: Some(1),
        affinity_node: None,
    };
    let decision = router.select_node(&request, &[c0, c1]).unwrap();
    assert_eq!(decision.target_node, "node-1");
}

#[test]
fn pipeline_stage_errors_without_stage() {
    let router = PipelineStageRouter;
    let candidates = vec![make_candidate(
        "node-0",
        NodeRole::PipelineStage,
        NodeStatus::Online,
    )];
    let request = make_request(None);
    assert!(router.select_node(&request, &candidates).is_err());
}

#[test]
fn pipeline_stage_errors_when_no_matching_stage() {
    let router = PipelineStageRouter;
    let mut c0 = make_candidate("node-0", NodeRole::PipelineStage, NodeStatus::Online);
    c0.stage = Some(0);

    let request = RoutingRequest {
        request_id: "req-1".to_string(),
        preferred_role: None,
        preferred_stage: Some(5),
        affinity_node: None,
    };
    assert!(router.select_node(&request, &[c0]).is_err());
}

#[test]
fn routing_decision_display_fields() {
    let decision = RoutingDecision {
        target_node: "node-0".to_string(),
        strategy: "test".to_string(),
        reason: Some("test reason".to_string()),
    };
    assert_eq!(decision.target_node, "node-0");
    assert_eq!(decision.strategy, "test");
    assert_eq!(decision.reason.as_deref(), Some("test reason"));
}
