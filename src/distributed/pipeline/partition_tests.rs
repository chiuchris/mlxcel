use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_model(
    num_layers: usize,
    layer_bytes: u64,
    embed_bytes: u64,
    head_bytes: u64,
) -> ModelProfile {
    ModelProfile {
        num_layers,
        layer_param_bytes: layer_bytes,
        embedding_param_bytes: embed_bytes,
        lm_head_param_bytes: head_bytes,
    }
}

fn make_device(id: &str, mem: u64) -> DeviceSpec {
    DeviceSpec {
        device_id: id.to_string(),
        available_memory_bytes: mem,
        compute_units: 0,
    }
}

fn make_device_with_compute(id: &str, mem: u64, cu: u32) -> DeviceSpec {
    DeviceSpec {
        device_id: id.to_string(),
        available_memory_bytes: mem,
        compute_units: cu,
    }
}

/// Verify that every layer 0..num_layers is covered exactly once.
fn assert_full_coverage(assignments: &[StageAssignment], num_layers: usize) {
    let mut covered = vec![false; num_layers];
    for a in assignments {
        for l in a.layer_range.clone() {
            assert!(!covered[l], "layer {l} covered by multiple stages");
            covered[l] = true;
        }
    }
    for (l, &c) in covered.iter().enumerate() {
        assert!(c, "layer {l} not covered by any stage");
    }
}

// ---------------------------------------------------------------------------
// ModelProfile
// ---------------------------------------------------------------------------

#[test]
fn model_profile_total_bytes() {
    let m = make_model(32, 100, 50, 50);
    assert_eq!(m.total_param_bytes(), 50 + 32 * 100 + 50);
}

// ---------------------------------------------------------------------------
// parse_manual_partition
// ---------------------------------------------------------------------------

#[test]
fn parse_manual_partition_basic() {
    let ranges = parse_manual_partition("0-15,16-31", 32).unwrap();
    assert_eq!(ranges, vec![0..16, 16..32]);
}

#[test]
fn parse_manual_partition_three_stages() {
    let ranges = parse_manual_partition("0-9,10-19,20-31", 32).unwrap();
    assert_eq!(ranges, vec![0..10, 10..20, 20..32]);
}

#[test]
fn parse_manual_partition_whitespace() {
    let ranges = parse_manual_partition(" 0-7 , 8-15 ", 16).unwrap();
    assert_eq!(ranges, vec![0..8, 8..16]);
}

#[test]
fn parse_manual_partition_single_layer_range() {
    let ranges = parse_manual_partition("0-0,1-1", 2).unwrap();
    assert_eq!(ranges, vec![0..1, 1..2]);
}

#[test]
fn parse_manual_partition_empty_fails() {
    assert!(parse_manual_partition("", 32).is_err());
}

#[test]
fn parse_manual_partition_inverted_range_fails() {
    assert!(parse_manual_partition("15-0,16-31", 32).is_err());
}

#[test]
fn parse_manual_partition_exceeds_layers_fails() {
    assert!(parse_manual_partition("0-31,32-63", 32).is_err());
}

#[test]
fn parse_manual_partition_missing_dash_fails() {
    assert!(parse_manual_partition("0_15,16-31", 32).is_err());
}

#[test]
fn parse_manual_partition_non_numeric_fails() {
    assert!(parse_manual_partition("a-b,c-d", 32).is_err());
}

// ---------------------------------------------------------------------------
// validate_partition
// ---------------------------------------------------------------------------

#[test]
fn validate_partition_valid() {
    let assignments = vec![
        StageAssignment {
            stage_index: 0,
            device_id: "d0".into(),
            layer_range: 0..16,
            has_embedding: true,
            has_lm_head: false,
            estimated_memory_bytes: 1000,
        },
        StageAssignment {
            stage_index: 1,
            device_id: "d1".into(),
            layer_range: 16..32,
            has_embedding: false,
            has_lm_head: true,
            estimated_memory_bytes: 1000,
        },
    ];
    validate_partition(&assignments, 32).unwrap();
}

#[test]
fn validate_partition_gap_fails() {
    let assignments = vec![
        StageAssignment {
            stage_index: 0,
            device_id: "d0".into(),
            layer_range: 0..10,
            has_embedding: true,
            has_lm_head: false,
            estimated_memory_bytes: 0,
        },
        StageAssignment {
            stage_index: 1,
            device_id: "d1".into(),
            layer_range: 12..32,
            has_embedding: false,
            has_lm_head: true,
            estimated_memory_bytes: 0,
        },
    ];
    let err = validate_partition(&assignments, 32).unwrap_err();
    assert!(err.to_string().contains("gap or overlap"));
}

#[test]
fn validate_partition_overlap_fails() {
    let assignments = vec![
        StageAssignment {
            stage_index: 0,
            device_id: "d0".into(),
            layer_range: 0..18,
            has_embedding: true,
            has_lm_head: false,
            estimated_memory_bytes: 0,
        },
        StageAssignment {
            stage_index: 1,
            device_id: "d1".into(),
            layer_range: 16..32,
            has_embedding: false,
            has_lm_head: true,
            estimated_memory_bytes: 0,
        },
    ];
    let err = validate_partition(&assignments, 32).unwrap_err();
    assert!(err.to_string().contains("gap or overlap"));
}

#[test]
fn validate_partition_missing_embedding_fails() {
    let assignments = vec![StageAssignment {
        stage_index: 0,
        device_id: "d0".into(),
        layer_range: 0..32,
        has_embedding: false,
        has_lm_head: true,
        estimated_memory_bytes: 0,
    }];
    let err = validate_partition(&assignments, 32).unwrap_err();
    assert!(err.to_string().contains("embedding"));
}

#[test]
fn validate_partition_missing_lm_head_fails() {
    let assignments = vec![StageAssignment {
        stage_index: 0,
        device_id: "d0".into(),
        layer_range: 0..32,
        has_embedding: true,
        has_lm_head: false,
        estimated_memory_bytes: 0,
    }];
    let err = validate_partition(&assignments, 32).unwrap_err();
    assert!(err.to_string().contains("lm_head"));
}

#[test]
fn validate_partition_empty_fails() {
    let err = validate_partition(&[], 32).unwrap_err();
    assert!(err.to_string().contains("at least one stage"));
}

// ---------------------------------------------------------------------------
// validate_memory_fit
// ---------------------------------------------------------------------------

#[test]
fn validate_memory_fit_ok() {
    let assignments = vec![StageAssignment {
        stage_index: 0,
        device_id: "d0".into(),
        layer_range: 0..4,
        has_embedding: true,
        has_lm_head: true,
        estimated_memory_bytes: 500,
    }];
    let devices = vec![make_device("d0", 1000)];
    validate_memory_fit(&assignments, &devices).unwrap();
}

#[test]
fn validate_memory_fit_exceeds() {
    let assignments = vec![StageAssignment {
        stage_index: 0,
        device_id: "d0".into(),
        layer_range: 0..4,
        has_embedding: true,
        has_lm_head: true,
        estimated_memory_bytes: 2000,
    }];
    let devices = vec![make_device("d0", 1000)];
    let err = validate_memory_fit(&assignments, &devices).unwrap_err();
    assert!(err.to_string().contains("insufficient") || err.to_string().contains("requires"));
}

// ---------------------------------------------------------------------------
// auto_partition — uniform devices
// ---------------------------------------------------------------------------

#[test]
fn auto_partition_single_device() {
    let model = make_model(32, 100, 50, 50);
    let devices = vec![make_device("d0", 100_000)];
    let result = auto_partition(&model, &devices).unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].layer_range, 0..32);
    assert!(result[0].has_embedding);
    assert!(result[0].has_lm_head);
    assert_eq!(result[0].estimated_memory_bytes, 50 + 32 * 100 + 50);
}

#[test]
fn auto_partition_two_equal_devices() {
    let model = make_model(32, 100, 50, 50);
    let devices = vec![make_device("d0", 50_000), make_device("d1", 50_000)];
    let result = auto_partition(&model, &devices).unwrap();

    assert_eq!(result.len(), 2);
    assert_full_coverage(&result, 32);
    assert!(result[0].has_embedding);
    assert!(!result[0].has_lm_head);
    assert!(!result[1].has_embedding);
    assert!(result[1].has_lm_head);
    // With equal memory and equal reservations, expect roughly 16/16 split.
    let total_layers: usize = result.iter().map(|a| a.layer_range.len()).sum();
    assert_eq!(total_layers, 32);
}

#[test]
fn auto_partition_four_equal_devices() {
    let model = make_model(32, 100, 50, 50);
    let devices: Vec<DeviceSpec> = (0..4)
        .map(|i| make_device(&format!("d{i}"), 50_000))
        .collect();
    let result = auto_partition(&model, &devices).unwrap();

    assert_eq!(result.len(), 4);
    assert_full_coverage(&result, 32);
    assert!(result[0].has_embedding);
    assert!(result[3].has_lm_head);
    // Roughly 8 layers each.
    for a in &result {
        assert!(a.layer_range.len() >= 1);
    }
}

// ---------------------------------------------------------------------------
// auto_partition — non-uniform devices
// ---------------------------------------------------------------------------

#[test]
fn auto_partition_non_uniform_two_devices() {
    // 128GB device + 48GB device. The larger should get more layers.
    let model = make_model(32, 1_000_000_000, 500_000_000, 500_000_000);
    let devices = vec![
        make_device("big", 128_000_000_000),
        make_device("small", 48_000_000_000),
    ];
    let result = auto_partition(&model, &devices).unwrap();

    assert_eq!(result.len(), 2);
    assert_full_coverage(&result, 32);

    let big_layers = result[0].layer_range.len();
    let small_layers = result[1].layer_range.len();
    // The big device (128GB) should get more layers than the small one (48GB).
    assert!(
        big_layers > small_layers,
        "expected big device to get more layers ({big_layers} vs {small_layers})"
    );
}

#[test]
fn auto_partition_three_non_uniform_devices() {
    let model = make_model(48, 100, 50, 50);
    let devices = vec![
        make_device("d0", 50_000), // ~50k
        make_device("d1", 25_000), // ~25k
        make_device("d2", 25_000), // ~25k
    ];
    let result = auto_partition(&model, &devices).unwrap();

    assert_eq!(result.len(), 3);
    assert_full_coverage(&result, 48);

    // d0 should get roughly double the layers of d1/d2.
    let l0 = result[0].layer_range.len();
    let l1 = result[1].layer_range.len();
    let l2 = result[2].layer_range.len();
    assert!(
        l0 >= l1 && l0 >= l2,
        "expected d0 to get the most layers ({l0}, {l1}, {l2})"
    );
}

// ---------------------------------------------------------------------------
// auto_partition — edge cases
// ---------------------------------------------------------------------------

#[test]
fn auto_partition_one_layer_per_device() {
    let model = make_model(3, 100, 50, 50);
    let devices = vec![
        make_device("d0", 10_000),
        make_device("d1", 10_000),
        make_device("d2", 10_000),
    ];
    let result = auto_partition(&model, &devices).unwrap();

    assert_eq!(result.len(), 3);
    assert_full_coverage(&result, 3);
    for a in &result {
        assert_eq!(a.layer_range.len(), 1);
    }
}

#[test]
fn auto_partition_more_devices_than_layers_fails() {
    let model = make_model(2, 100, 50, 50);
    let devices = vec![
        make_device("d0", 10_000),
        make_device("d1", 10_000),
        make_device("d2", 10_000),
    ];
    let err = auto_partition(&model, &devices).unwrap_err();
    assert!(err.to_string().contains("more devices"));
}

#[test]
fn auto_partition_no_devices_fails() {
    let model = make_model(32, 100, 50, 50);
    let err = auto_partition(&model, &[]).unwrap_err();
    assert!(err.to_string().contains("at least one device"));
}

#[test]
fn auto_partition_zero_layers_fails() {
    let model = make_model(0, 100, 50, 50);
    let devices = vec![make_device("d0", 10_000)];
    let err = auto_partition(&model, &devices).unwrap_err();
    assert!(err.to_string().contains("at least one layer"));
}

#[test]
fn auto_partition_insufficient_memory_fails() {
    let model = make_model(32, 1_000_000, 500_000, 500_000);
    let devices = vec![
        make_device("d0", 100), // way too small
        make_device("d1", 100),
    ];
    let err = auto_partition(&model, &devices).unwrap_err();
    assert!(
        err.to_string().contains("insufficient")
            || err.to_string().contains("requires")
            || err.to_string().contains("bytes")
    );
}

#[test]
fn auto_partition_embedding_too_large_fails() {
    let model = make_model(4, 100, 999_999, 50);
    let devices = vec![make_device("d0", 1000), make_device("d1", 1000)];
    let err = auto_partition(&model, &devices).unwrap_err();
    assert!(err.to_string().contains("embedding"));
}

#[test]
fn auto_partition_lm_head_too_large_fails() {
    let model = make_model(4, 100, 50, 999_999);
    let devices = vec![make_device("d0", 1000), make_device("d1", 1000)];
    let err = auto_partition(&model, &devices).unwrap_err();
    assert!(err.to_string().contains("lm_head"));
}

// ---------------------------------------------------------------------------
// auto_partition — memory estimates
// ---------------------------------------------------------------------------

#[test]
fn auto_partition_memory_estimates_include_embedding_and_head() {
    let model = make_model(8, 100, 200, 300);
    let devices = vec![make_device("d0", 100_000), make_device("d1", 100_000)];
    let result = auto_partition(&model, &devices).unwrap();

    // First stage should include embedding cost.
    let first = &result[0];
    assert!(first.estimated_memory_bytes >= model.embedding_param_bytes);

    // Last stage should include lm_head cost.
    let last = result.last().unwrap();
    assert!(last.estimated_memory_bytes >= model.lm_head_param_bytes);
}

// ---------------------------------------------------------------------------
// build_manual_assignments
// ---------------------------------------------------------------------------

#[test]
fn build_manual_assignments_valid() {
    let model = make_model(32, 100, 50, 50);
    let devices = vec![make_device("d0", 100_000), make_device("d1", 100_000)];
    let ranges = parse_manual_partition("0-15,16-31", 32).unwrap();
    let result = build_manual_assignments(&ranges, &model, &devices).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(result[0].layer_range, 0..16);
    assert_eq!(result[1].layer_range, 16..32);
    assert!(result[0].has_embedding);
    assert!(result[1].has_lm_head);
}

#[test]
fn build_manual_assignments_gap_fails() {
    let model = make_model(32, 100, 50, 50);
    let devices = vec![make_device("d0", 100_000), make_device("d1", 100_000)];
    // Gap: layers 10-15 are missing.
    let ranges = vec![0..10, 16..32];
    let err = build_manual_assignments(&ranges, &model, &devices).unwrap_err();
    assert!(err.to_string().contains("gap or overlap"));
}

#[test]
fn build_manual_assignments_mismatched_device_count_fails() {
    let model = make_model(32, 100, 50, 50);
    let devices = vec![make_device("d0", 100_000)];
    let ranges = vec![0..16, 16..32];
    let err = build_manual_assignments(&ranges, &model, &devices).unwrap_err();
    assert!(err.to_string().contains("ranges") || err.to_string().contains("devices"));
}

#[test]
fn build_manual_assignments_memory_exceeded_fails() {
    let model = make_model(32, 1_000_000, 500_000, 500_000);
    let devices = vec![
        make_device("d0", 100), // too small
        make_device("d1", 100_000_000),
    ];
    let ranges = vec![0..16, 16..32];
    let err = build_manual_assignments(&ranges, &model, &devices).unwrap_err();
    assert!(err.to_string().contains("requires") || err.to_string().contains("available"));
}

// ---------------------------------------------------------------------------
// PartitionConfig
// ---------------------------------------------------------------------------

#[test]
fn partition_config_default_is_auto() {
    assert_eq!(PartitionConfig::default(), PartitionConfig::Auto);
}

#[test]
fn partition_config_manual_stores_ranges() {
    let cfg = PartitionConfig::Manual(vec![0..16, 16..32]);
    match cfg {
        PartitionConfig::Manual(ranges) => {
            assert_eq!(ranges.len(), 2);
        }
        _ => panic!("expected Manual variant"),
    }
}

// ---------------------------------------------------------------------------
// DeviceSpec with compute_units
// ---------------------------------------------------------------------------

#[test]
fn device_spec_with_compute_units() {
    let d = make_device_with_compute("gpu0", 1_000_000, 128);
    assert_eq!(d.compute_units, 128);
    assert_eq!(d.available_memory_bytes, 1_000_000);
}
