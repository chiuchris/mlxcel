//! Benchmark for Nemotron-H model using mlxcel-core
//!
//! This binary loads the Nemotron-H hybrid Mamba+Transformer+MoE model and benchmarks
//! token generation throughput.

use mlxcel::models::nemotron_h::NemotronHModel;
use mlxcel_core::generate::{CxxGenerator, SamplingConfig};
use std::time::Instant;

const MODEL_PATH: &str = "models/nemotron-30b-4bit";

fn main() {
    println!("=======================================================");
    println!("  Nemotron-H 30B Benchmark - mlxcel-core");
    println!("=======================================================");
    println!();

    // Load model
    println!("Loading model from: {}", MODEL_PATH);
    let start = Instant::now();
    let (model, args) = match NemotronHModel::load(MODEL_PATH) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };
    let load_time = start.elapsed();
    println!("Model loaded in {:.2}s", load_time.as_secs_f64());
    println!();

    // Print model info
    println!("Model Configuration:");
    println!("  Hidden size: {}", args.hidden_size);
    println!("  Num layers: {}", args.num_hidden_layers);
    println!("  Num attention heads: {}", args.num_attention_heads);
    println!("  Hybrid pattern: {:?}", args.hybrid_override_pattern);
    println!("  Num routed experts: {:?}", args.n_routed_experts);
    println!();

    // Set wired memory limit for optimal performance
    let max_memory = mlxcel_core::gpu_max_memory_size();
    mlxcel_core::set_wired_limit(max_memory);
    println!(
        "GPU max memory size: {:.2} GB",
        max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!();

    // Test prompt tokens
    let prompt_tokens = vec![1_i32, 9906, 11, 1268, 527, 499, 3432, 30, 358, 1097];
    println!("Prompt tokens: {:?}", prompt_tokens);
    println!();

    // Create generator
    let num_layers = model.num_layers();
    let mut generator = CxxGenerator::new(num_layers);

    // Warmup
    println!("Warming up...");
    let _ = generator.generate(&model, &prompt_tokens, 3, &SamplingConfig::greedy());
    generator.reset();
    mlxcel_core::clear_memory_cache();
    println!("Warmup complete");
    println!();

    // Benchmark
    let max_tokens = 50;
    println!("Benchmarking generation of {} tokens...", max_tokens);
    println!();

    // Generation benchmark (includes prefill)
    generator.reset();
    let gen_start = Instant::now();
    let generated = generator.generate(
        &model,
        &prompt_tokens,
        max_tokens,
        &SamplingConfig::greedy(),
    );
    let total_time = gen_start.elapsed();
    let throughput = generated.len() as f64 / total_time.as_secs_f64();

    // Results
    println!("=======================================================");
    println!("  Results");
    println!("=======================================================");
    println!();
    println!("Prompt tokens: {}", prompt_tokens.len());
    println!("Generated tokens: {}", generated.len());
    println!();
    println!("Total time: {:.2} ms", total_time.as_secs_f64() * 1000.0);
    println!("Throughput: {:.2} tok/s", throughput);
    println!();
    println!("RESULT:total_ms={:.2}", total_time.as_secs_f64() * 1000.0);
    println!("RESULT:gen_tokens={}", generated.len());
    println!("RESULT:throughput={:.2}", throughput);
    println!();

    // Cleanup
    mlxcel_core::clear_memory_cache();
    mlxcel_core::set_wired_limit(0);
}
