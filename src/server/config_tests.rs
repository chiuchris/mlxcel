use super::ServerConfig;

#[test]
fn server_config_default_matches_llama_server_compatibility_defaults() {
    let config = ServerConfig::default();

    assert_eq!(config.timeout_seconds, 600);
    assert_eq!(config.context_size, 0);
    assert_eq!(config.n_parallel, 1);
    assert!(config.enable_slots_endpoint);
    assert!(!config.enable_props_endpoint);
    assert!(!config.enable_metrics_endpoint);
    assert_eq!(config.default_temperature, 0.8);
    assert_eq!(config.default_top_p, 0.9);
    assert_eq!(config.default_top_k, 40);
    assert_eq!(config.default_min_p, 0.1);
    assert_eq!(config.default_repetition_penalty, 1.0);
    assert_eq!(config.default_repetition_context_size, 64);
    assert_eq!(config.default_max_tokens, 512);
    assert_eq!(config.default_dry_multiplier, 0.0);
    assert_eq!(config.default_dry_base, 1.75);
    assert_eq!(config.default_dry_allowed_length, 2);
    assert_eq!(config.default_dry_penalty_last_n, 0);
    assert_eq!(config.num_draft_tokens, 3);
}
