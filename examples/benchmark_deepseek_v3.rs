//! Benchmark for DeepSeek-V3 mlxcel-core implementation

use mlxcel::models::deepseek_v3::{DeepSeekV3Config, DeepSeekV3Model};
use mlxcel_core::generate::{CxxGenerator, LanguageModel, SamplingConfig};
use std::time::Instant;

const MODEL_PATH: &str = "models/DeepSeek-V3-0324-4bit";

fn main() {
    println!("=======================================================");
    println!("  DeepSeek-V3 Performance Benchmark");
    println!("=======================================================");
    println!();
    println!("Model path: {}", MODEL_PATH);
    println!();

    // Set wired memory limit for optimal performance
    let max_memory = mlxcel_core::gpu_max_memory_size();
    mlxcel_core::set_wired_limit(max_memory);
    println!(
        "GPU max memory size: {:.2} GB",
        max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!();

    // Load config for model info display
    let config_path = std::path::Path::new(MODEL_PATH).join("config.json");
    let config_str = std::fs::read_to_string(&config_path).expect("Failed to read config");
    let config: DeepSeekV3Config =
        serde_json::from_str(&config_str).expect("Failed to parse config");

    println!("Model Configuration:");
    println!("  Model type: {}", config.model_type);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Num layers: {}", config.num_hidden_layers);
    println!("  Num attention heads: {}", config.num_attention_heads);
    println!("  Num KV heads: {}", config.num_key_value_heads);
    println!("  KV LoRA rank: {}", config.kv_lora_rank);
    println!("  Q LoRA rank: {}", config.q_lora_rank);
    println!("  Vocab size: {}", config.vocab_size);
    println!(
        "  N routed experts: {:?}",
        config.n_routed_experts.unwrap_or(0)
    );
    println!(
        "  N shared experts: {:?}",
        config.n_shared_experts.unwrap_or(0)
    );
    println!("  Experts per token: {}", config.num_experts_per_tok);
    println!("  MoE layer freq: {}", config.moe_layer_freq);
    println!("  First k dense replace: {}", config.first_k_dense_replace);
    println!();

    // Test prompt tokens
    let prompt_tokens = vec![
        15339_i32, // "Hello"
        11,        // ","
        1268,      // " how"
        527,       // " are"
        499,       // " you"
        3432,      // " today"
        30,        // "?"
        358,       // " I"
        1097,      // " am"
    ];

    println!(
        "Prompt tokens: {:?} ({} tokens)",
        prompt_tokens,
        prompt_tokens.len()
    );
    println!();

    let max_tokens = 50;

    // =========================================================================
    // Benchmark mlxcel-core implementation
    // =========================================================================
    println!("=======================================================");
    println!("  mlxcel-core Implementation");
    println!("=======================================================");
    println!();

    println!("Loading mlxcel-core model...");
    let cxx_load_start = Instant::now();
    let cxx_model = match DeepSeekV3Model::load(MODEL_PATH) {
        Ok((model, _)) => model,
        Err(e) => {
            eprintln!("Failed to load mlxcel-core model: {}", e);
            eprintln!("This may be due to weight loading or configuration issues.");
            return;
        }
    };
    let cxx_load_time = cxx_load_start.elapsed();
    println!(
        "mlxcel-core model loaded in {:.2}s",
        cxx_load_time.as_secs_f64()
    );
    println!();

    // Create generator
    let num_layers = cxx_model.num_layers();
    let mut cxx_generator = CxxGenerator::new(num_layers);

    // Warmup
    println!("Warming up mlxcel-core...");
    let _ = cxx_generator.generate(&cxx_model, &prompt_tokens, 3, &SamplingConfig::greedy());
    cxx_generator.reset();
    mlxcel_core::clear_memory_cache();

    // Prefill benchmark
    let cxx_prefill_start = Instant::now();
    let input = mlxcel_core::from_slice_i32(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let mut cxx_caches = cxx_model.make_caches();
    let logits = LanguageModel::forward(&cxx_model, &input, &mut cxx_caches, None);
    mlxcel_core::eval(&logits);
    let cxx_prefill_time = cxx_prefill_start.elapsed();

    // Generation benchmark
    cxx_generator.reset();
    let cxx_gen_start = Instant::now();
    let cxx_generated = cxx_generator.generate(
        &cxx_model,
        &prompt_tokens,
        max_tokens,
        &SamplingConfig::greedy(),
    );
    let cxx_total_time = cxx_gen_start.elapsed();
    let cxx_gen_time = cxx_total_time.saturating_sub(cxx_prefill_time);

    let cxx_throughput = if cxx_gen_time.as_secs_f64() > 0.0 {
        cxx_generated.len() as f64 / cxx_gen_time.as_secs_f64()
    } else {
        cxx_generated.len() as f64 / cxx_total_time.as_secs_f64()
    };

    println!("Results (mlxcel-core):");
    println!(
        "  Prefill time: {:.2} ms",
        cxx_prefill_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Generation time: {:.2} ms",
        cxx_gen_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Total time: {:.2} ms",
        cxx_total_time.as_secs_f64() * 1000.0
    );
    println!("  Generated tokens: {}", cxx_generated.len());
    println!("  Throughput: {:.2} tok/s", cxx_throughput);
    println!();

    // Cleanup
    mlxcel_core::clear_memory_cache();
    mlxcel_core::set_wired_limit(0);
}
