//! Benchmark for Llama 4 Scout (MoE) model using mlxcel-core
//!
//! This binary loads the actual Llama 4 Scout model and benchmarks
//! token generation throughput.

use mlxcel::models::llama4::Llama4CxxModel;
use mlxcel_core::generate::{CxxGenerator, SamplingConfig};
use std::time::Instant;

const MODEL_PATH: &str = "models/Llama-4-Scout-17B-16E-Instruct-4bit";

fn main() {
    println!("=======================================================");
    println!("  Llama 4 Scout (MoE) Benchmark - mlxcel-core");
    println!("=======================================================");
    println!();

    // Load model
    println!("Loading model from: {}", MODEL_PATH);
    let start = Instant::now();
    let (model, args) = match Llama4CxxModel::load(MODEL_PATH) {
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
    println!("  Num KV heads: {}", args.num_key_value_heads);
    println!("  Num experts: {}", args.num_local_experts);
    println!("  Top-k experts: {}", args.num_experts_per_tok);
    println!("  Vocab size: {}", args.vocab_size);
    println!("  Group size: {}", args.group_size());
    println!("  Bits: {}", args.bits());
    println!();

    // Create generator
    let num_layers = model.layers.len();
    let mut generator = CxxGenerator::new(num_layers);

    // Test prompt tokens for "Hello, how are you?"
    let prompt_tokens = vec![
        200000_i32, // <|begin_of_text|>
        19873,      // "Hello"
        24,         // ","
        1659,       // " how"
        583,        // " are"
        650,        // " you"
        43,         // "?"
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
    let warmup_tokens = generator.generate(&model, &prompt_tokens, 3, &SamplingConfig::greedy());
    generator.reset();
    println!("Warmup generated {} tokens", warmup_tokens.len());
    mlxcel_core::clear_memory_cache();
    println!();

    // Benchmark
    let max_tokens = 30;
    println!("Benchmarking generation of {} tokens...", max_tokens);
    println!();

    let start = Instant::now();

    // Prefill phase
    let input = mlxcel_core::from_slice_i32(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let mut caches = model.make_caches();
    let logits = model.forward(&input, &mut caches, None);
    mlxcel_core::eval(&logits);
    let prefill_time = start.elapsed();

    // Generation phase
    generator.reset();
    let gen_start = Instant::now();
    let generated = generator.generate(
        &model,
        &prompt_tokens,
        max_tokens,
        &SamplingConfig::greedy(),
    );
    let total_time = gen_start.elapsed();
    let gen_time = total_time.saturating_sub(prefill_time);

    // Calculate throughput
    let gen_tokens = generated.len();
    let throughput = if gen_time.as_secs_f64() > 0.0 {
        gen_tokens as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

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
    println!("Throughput: {:.2} tok/s", throughput);
    println!();

    // Print generated tokens
    println!("Generated token IDs: {:?}", generated);
    println!();

    // Cleanup
    mlxcel_core::clear_memory_cache();
    mlxcel_core::set_wired_limit(0);
}
