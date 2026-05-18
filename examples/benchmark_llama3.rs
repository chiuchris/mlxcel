//! Benchmark for Llama 3.1 8B Instruct model using mlxcel-core
//!
//! This binary loads the actual Llama 3.1 8B model and benchmarks
//! token generation throughput.

use mlxcel::models::llama3::Llama3Model;
use mlxcel_core::generate::{CxxGenerator, SamplingConfig};
use std::time::Instant;

const MODEL_PATH: &str = "models/Meta-Llama-3.1-8B-Instruct-4bit";

fn main() {
    println!("=======================================================");
    println!("  Llama 3.1 8B Instruct Benchmark - mlxcel-core");
    println!("=======================================================");
    println!();

    // Load model
    println!("Loading model from: {}", MODEL_PATH);
    let start = Instant::now();
    let (model, args) = match Llama3Model::load(MODEL_PATH) {
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
    println!("  Model type: {}", args.model_type);
    println!("  Hidden size: {}", args.hidden_size);
    println!("  Num layers: {}", args.num_hidden_layers);
    println!("  Num attention heads: {}", args.num_attention_heads);
    println!("  Num KV heads: {}", args.num_kv_heads());
    println!("  Head dim: {}", args.head_dim());
    println!("  Vocab size: {}", args.vocab_size);
    println!("  RMS norm eps: {}", args.rms_norm_eps);
    println!("  RoPE theta: {}", args.rope_theta);
    println!("  Group size: {}", args.group_size());
    println!("  Bits: {}", args.bits());
    println!();

    // Create generator
    let num_layers = model.layers.len();
    let mut generator = CxxGenerator::new(num_layers);

    // Test prompt tokens for "Hello, how are you today? I am"
    // Llama 3.1 tokenizer outputs
    let prompt_tokens = vec![
        128000_i32, // <|begin_of_text|>
        9906,       // "Hello"
        11,         // ","
        1268,       // " how"
        527,        // " are"
        499,        // " you"
        3432,       // " today"
        30,         // "?"
        358,        // " I"
        1097,       // " am"
    ];

    println!("Prompt tokens: {:?}", prompt_tokens);
    println!();

    // Set wired memory limit for optimal performance
    let max_memory = mlxcel_core::gpu_max_memory_size();
    mlxcel_core::set_wired_limit(max_memory);
    println!(
        "GPU max memory size: {:.2} GB",
        max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!();

    // Warmup
    println!("Warming up...");
    let warmup_tokens = generator.generate(&model, &prompt_tokens, 5, &SamplingConfig::greedy());
    generator.reset();
    println!("Warmup generated {} tokens", warmup_tokens.len());
    mlxcel_core::clear_memory_cache();
    println!();

    // Benchmark
    let max_tokens = 50;
    println!("Benchmarking generation of {} tokens...", max_tokens);
    println!();

    // Measure prefill and generation separately
    // 1. Prefill phase (separate from generation)
    let prefill_start = Instant::now();
    let input = mlxcel_core::from_slice_i32(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let mut caches = model.make_caches();
    let logits = model.forward(&input, &mut caches, None);
    mlxcel_core::eval(&logits);
    let prefill_time = prefill_start.elapsed();

    // 2. Full generation (includes its own prefill for fair comparison)
    generator.reset();
    let gen_start = Instant::now();
    let generated = generator.generate(
        &model,
        &prompt_tokens,
        max_tokens,
        &SamplingConfig::greedy(),
    );
    let total_time = gen_start.elapsed();

    // Estimate generation-only time by subtracting prefill time
    // This assumes generator's internal prefill takes similar time
    let gen_time = total_time.saturating_sub(prefill_time);

    // Calculate throughput based on generation-only time
    let gen_tokens = generated.len();
    let throughput = if gen_time.as_secs_f64() > 0.0 {
        gen_tokens as f64 / gen_time.as_secs_f64()
    } else {
        gen_tokens as f64 / total_time.as_secs_f64()
    };

    // Also calculate total throughput for comparison
    let total_throughput = gen_tokens as f64 / total_time.as_secs_f64();

    // Results
    println!("=======================================================");
    println!("  Results");
    println!("=======================================================");
    println!();
    println!("Prompt tokens: {}", prompt_tokens.len());
    println!("Generated tokens: {}", gen_tokens);
    println!();
    println!(
        "Prefill time: {:.2} ms",
        prefill_time.as_secs_f64() * 1000.0
    );
    println!("Generation time: {:.2} ms", gen_time.as_secs_f64() * 1000.0);
    println!("Total time: {:.2} ms", total_time.as_secs_f64() * 1000.0);
    println!();
    println!("Throughput (gen only): {:.2} tok/s", throughput);
    println!("Throughput (total): {:.2} tok/s", total_throughput);
    println!();

    // Print generated tokens
    println!("Generated token IDs: {:?}", generated);
    println!();

    // Cleanup
    mlxcel_core::clear_memory_cache();
    mlxcel_core::set_wired_limit(0);
}
