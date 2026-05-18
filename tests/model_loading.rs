use mlxcel::utils::config::load_config;
use std::path::PathBuf;

#[test]
fn test_load_llama_3_1_config() {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("reference/Meta-Llama-3.1-8B-Instruct-4bit");

    // Check if the directory exists (it might not on CI/other environments, but user said it's there)
    if !d.exists() {
        eprintln!("Skipping test: Model directory not found at {:?}", d);
        return;
    }

    let config = load_config(&d).expect("Failed to load config");

    println!("Loaded config: {:?}", config);

    assert_eq!(config.model_type, "llama");
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.vocab_size, 128256); // Llama 3 specific
    assert_eq!(config.rope_theta, 500000.0); // Llama 3.1 uses high theta
}
