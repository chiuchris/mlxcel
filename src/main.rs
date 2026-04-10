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

use clap::{Args, Parser, Subcommand};
use mlxcel::tokenizer::MlxcelTokenizer;
use std::path::PathBuf;

mod commands;

/// mlxcel: High-performance LLM/VLM/VLA inference on Apple Silicon and CUDA GPUs
///
/// A Rust implementation for running Large Language Models, Vision-Language
/// Models, and Vision-Language-Action Models efficiently on Apple Silicon and
/// CUDA GPUs using the MLX framework.
#[derive(Parser, Debug)]
#[command(
    name = "mlxcel",
    author = "Lablup Inc.",
    version,
    about = "High-performance LLM/VLM/VLA inference on Apple Silicon and CUDA GPUs",
    long_about = None,
    after_help = "\
Environment Variables:
  MLXCEL_DEVICE          Runtime device: \"gpu\" (default), \"cpu\"
  MLXCEL_WIRED_LIMIT     GPU wired memory limit (default: none, like Python mlx-lm)
                           \"max\"  — pin all GPU memory (may OOM on large models)
                           \"0\"    — no limit (default)
                           \"96GB\" — explicit limit (supports GB, MB, or bytes)

Tensor Parallel Runtime:
  Current multi-rank support: dense Llama, Qwen2/2.5, Qwen3, Gemma 3 text, ERNIE 4.5, Hunyuan v1 Dense
  Current constraints: --tp-embedding-mode replicated, --tp-lm-head-mode replicated
                       LoRA unsupported, server batching supported for all listed dense runtimes

For more information, visit: https://github.com/lablup/mlxcel"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate text from a prompt
    #[command(visible_alias = "gen")]
    Generate(GenerateArgs),

    /// Start an OpenAI/llama-server compatible HTTP server
    Serve(ServeArgs),

    /// List supported model architectures
    #[command(visible_alias = "ls")]
    List,
}

#[derive(Args, Debug)]
pub(crate) struct GenerateArgs {
    #[command(flatten)]
    pub(crate) model: ModelOptions,

    #[command(flatten)]
    pub(crate) generation: GenerationOptions,

    #[command(flatten)]
    pub(crate) sampling: SamplingOptions,

    #[command(flatten)]
    pub(crate) tensor_parallel: TensorParallelOptions,
}

/// Model loading options
#[derive(Args, Debug)]
#[command(next_help_heading = "Model Options")]
pub(crate) struct ModelOptions {
    /// Path to the model directory
    #[arg(short, long, value_name = "PATH")]
    pub(crate) model: PathBuf,

    /// Path to LoRA adapter directory (optional)
    #[arg(long, value_name = "PATH")]
    pub(crate) adapter: Option<PathBuf>,

    /// Path to draft model for speculative decoding (optional)
    #[arg(long, value_name = "PATH")]
    pub(crate) draft_model: Option<PathBuf>,

    /// Number of draft tokens per speculation step (default: 3)
    #[arg(long, default_value_t = 3, value_name = "N")]
    pub(crate) num_draft_tokens: usize,
}

/// Text generation options
#[derive(Args, Debug)]
#[command(next_help_heading = "Generation Options")]
pub(crate) struct GenerationOptions {
    /// The prompt to generate text from
    #[arg(short, long, value_name = "TEXT")]
    pub(crate) prompt: String,

    /// Image file paths for vision-language models (VLM)
    #[arg(long, value_name = "PATH", num_args = 1..)]
    pub(crate) image: Vec<PathBuf>,

    /// Audio file path for audio-language models (e.g. Gemma4 with audio)
    #[arg(long, value_name = "PATH")]
    pub(crate) audio: Option<PathBuf>,

    /// Maximum number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 100, value_name = "N")]
    pub(crate) max_tokens: usize,

    /// Enable profiling mode with detailed timing breakdown
    #[arg(long, default_value_t = false)]
    pub(crate) profile: bool,

    /// Disable automatic chat template application
    #[arg(long, default_value_t = false)]
    pub(crate) no_chat_template: bool,

    /// Print the recommended quantization mode for this model on the current hardware.
    ///
    /// Detects Apple Silicon generation and available memory, estimates model
    /// parameter count from config.json, then suggests the optimal quantization
    /// (int8, int4, or fp16). On M5 hardware with sufficient memory, INT8 is
    /// recommended because the Neural Accelerator delivers ~2x throughput over
    /// FP16 for 8-bit integer matmuls.
    ///
    /// Also warns when the model uses BFloat16 weights on M5 hardware, since
    /// the Neural Accelerator does not support BFloat16 computation.
    #[arg(long, default_value_t = false)]
    pub(crate) recommend_quant: bool,

    /// KV cache quantization mode.
    ///
    /// Controls how accumulated key/value tensors are stored:
    ///   fp16  — Standard half-precision storage (default, no overhead).
    ///   int8  — Per-token INT8 absmax quantization; reduces KV cache memory
    ///           by ~50% at the cost of small quantization error per token.
    ///
    /// INT8 mode is most beneficial for long context generation where KV cache
    /// becomes the memory bottleneck.
    #[arg(long = "kv-cache-mode", default_value = "fp16", value_name = "MODE")]
    pub(crate) kv_cache_mode: String,
}

/// Sampling strategy options
#[derive(Args, Debug)]
#[command(next_help_heading = "Sampling Options")]
pub(crate) struct SamplingOptions {
    /// Sampling temperature (0.0 = greedy, higher = more random)
    #[arg(short, long, default_value_t = 0.0, value_name = "FLOAT")]
    pub(crate) temp: f32,

    /// Top-P (nucleus) sampling threshold
    #[arg(long, default_value_t = 1.0, value_name = "FLOAT")]
    pub(crate) top_p: f32,

    /// Top-K sampling limit
    #[arg(long, default_value_t = 0, value_name = "K")]
    pub(crate) top_k: i32,

    /// Min-P sampling threshold (0.0 = disabled)
    #[arg(long, default_value_t = 0.0, value_name = "FLOAT")]
    pub(crate) min_p: f32,

    /// Repetition penalty multiplier
    #[arg(long, default_value_t = 1.0, value_name = "FLOAT")]
    pub(crate) repetition_penalty: f32,

    /// DRY (Don't Repeat Yourself) penalty multiplier (0.0 = disabled)
    #[arg(long, default_value_t = 0.0, value_name = "FLOAT")]
    pub(crate) dry_multiplier: f32,

    /// DRY exponential base for penalty scaling
    #[arg(long, default_value_t = 1.75, value_name = "FLOAT")]
    pub(crate) dry_base: f32,

    /// DRY minimum match length before penalty applies
    #[arg(long, default_value_t = 2, value_name = "N")]
    pub(crate) dry_allowed_length: usize,

    /// DRY lookback window size (0 = use full history)
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub(crate) dry_penalty_last_n: usize,
}

/// Tensor-parallel options
#[derive(Args, Debug)]
#[command(next_help_heading = "Tensor Parallel Options")]
pub(crate) struct TensorParallelOptions {
    /// Number of tensor-parallel ranks (must be a power of 2).
    ///
    /// Current multi-rank runtime support is limited to dense Llama, Qwen2/2.5,
    /// Qwen3, Gemma 3 text, ERNIE 4.5, and Hunyuan v1 Dense models.
    #[arg(long = "tp-size", default_value_t = 1, value_name = "N")]
    pub(crate) tp_size: usize,

    /// MoE expert sharding mode: "expert_parallel" or "within_expert"
    #[arg(
        long = "tp-moe-mode",
        default_value = "expert_parallel",
        value_name = "MODE"
    )]
    pub(crate) tp_moe_mode: String,

    /// Embedding sharding mode: "vocab_parallel" or "replicated".
    ///
    /// The current in-process tensor-parallel runtime requires "replicated".
    #[arg(
        long = "tp-embedding-mode",
        default_value = "replicated",
        value_name = "MODE"
    )]
    pub(crate) tp_embedding_mode: String,

    /// LM head sharding mode: "vocab_parallel" or "replicated".
    ///
    /// The current in-process tensor-parallel runtime requires "replicated".
    #[arg(
        long = "tp-lm-head-mode",
        default_value = "replicated",
        value_name = "MODE"
    )]
    pub(crate) tp_lm_head_mode: String,
}

/// Server options
#[derive(Args, Debug)]
pub(crate) struct ServeArgs {
    /// Path to the model directory
    #[arg(short, long, env = "LLAMA_ARG_MODEL", value_name = "PATH")]
    model: PathBuf,

    /// Path to LoRA adapter directory
    #[arg(long, visible_alias = "lora", value_name = "PATH")]
    adapter: Option<PathBuf>,

    /// Model alias (shown in API responses instead of directory name)
    #[arg(short = 'a', long, env = "LLAMA_ARG_ALIAS", value_name = "NAME")]
    alias: Option<String>,

    /// Host address to bind to (or Unix socket path when --port 0)
    #[arg(long, env = "LLAMA_ARG_HOST", default_value = "127.0.0.1")]
    host: String,

    /// Port number to listen on (0 = Unix socket mode using --host as socket path)
    #[arg(long, env = "LLAMA_ARG_PORT", default_value_t = 8080)]
    port: u16,

    /// API key for authentication
    #[arg(long, env = "LLAMA_API_KEY", value_name = "KEY")]
    api_key: Option<String>,

    /// Path to file containing API key
    #[arg(long, value_name = "PATH")]
    api_key_file: Option<PathBuf>,

    /// Number of parallel request slots
    #[arg(long, env = "LLAMA_ARG_N_PARALLEL", default_value_t = 1)]
    n_parallel: usize,

    /// Context size limit (0 = use model default)
    #[arg(long, env = "LLAMA_ARG_CTX_SIZE", default_value_t = 0)]
    ctx_size: usize,

    /// Maximum tokens to predict (-1 = unlimited)
    #[arg(long = "n-predict", env = "LLAMA_ARG_N_PREDICT", default_value_t = -1)]
    n_predict: i32,

    /// Path to draft model for speculative decoding
    #[arg(long, value_name = "PATH")]
    draft_model: Option<PathBuf>,

    /// Maximum number of draft tokens per speculation step
    #[arg(long, env = "LLAMA_ARG_DRAFT_MAX", default_value_t = 16)]
    draft_max: usize,

    /// Maximum number of concurrent decode sequences (default: --n-parallel value)
    #[arg(long, value_name = "N")]
    max_batch_size: Option<usize>,

    /// Disable continuous batching and use the legacy sequential worker.
    ///
    /// When set, requests are processed one at a time in FIFO order with no
    /// batch scheduler overhead. Equivalent to using `--max-batch-size 1` but
    /// with explicit sequential semantics and no prefill chunking.
    #[arg(long)]
    no_batch: bool,

    /// Maximum number of requests waiting in the prefill queue (default: 32)
    #[arg(long, default_value_t = 32)]
    max_queue_depth: usize,

    /// Prefill chunk size in tokens (0 = disabled, default: 512)
    ///
    /// When set, long prompts are broken into chunks of this size and
    /// decode steps are interleaved between chunks to prevent latency
    /// spikes for active sequences.
    #[arg(long, default_value_t = 512)]
    prefill_chunk_size: usize,

    /// Prefill batch size [llama-server alias for --prefill-chunk-size] [default: 512]
    #[arg(
        short = 'b',
        long = "batch-size",
        env = "LLAMA_ARG_BATCH_SIZE",
        value_name = "N"
    )]
    batch_size: Option<usize>,

    /// Physical micro-batch size [not applicable on Apple Silicon unified memory; ignored]
    #[arg(long = "ubatch-size", env = "LLAMA_ARG_UBATCH_SIZE", value_name = "N")]
    ubatch_size: Option<usize>,

    /// Enable preemptive eviction of lower-priority sequences
    ///
    /// When enabled and the batch is full, a high-priority incoming
    /// request may evict a lower-priority active sequence (which will
    /// be re-queued for re-prefill).
    #[arg(long)]
    enable_preemption: bool,

    /// Preemption policy: "longest-first" (default) or "lowest-priority"
    ///
    /// Controls which active sequence is evicted when preemption is
    /// triggered. "longest-first" evicts the sequence with the most
    /// generated tokens; "lowest-priority" evicts the lowest-priority
    /// sequence (ties broken by longest).
    #[arg(long, default_value = "longest-first")]
    preemption_policy: String,

    /// Maximum number of requests to batch together for prefill (default: 1)
    ///
    /// When > 1, the scheduler collects up to this many pending requests and
    /// runs a single batched forward pass [batch_size, max_seq_len] for better
    /// Neural Accelerator utilization. Falls back to sequential prefill when
    /// only one request is pending. Recommended: 4-8 on M5 Pro/Max hardware.
    #[arg(long, default_value_t = 1)]
    max_batch_prefill: usize,

    /// Request timeout in seconds
    #[arg(long, env = "LLAMA_ARG_TIMEOUT", default_value_t = 600)]
    timeout: u64,

    /// Override chat template (Jinja2 template string)
    #[arg(long, value_name = "TEMPLATE")]
    chat_template: Option<String>,

    /// Path to chat template file
    #[arg(long, value_name = "PATH")]
    chat_template_file: Option<PathBuf>,

    /// Enable /slots endpoint
    #[arg(long = "slots", overrides_with = "_no_slots", default_value_t = true)]
    slots: bool,

    /// Disable /slots endpoint
    #[arg(long = "no-slots", overrides_with = "slots", hide = true)]
    _no_slots: bool,

    /// Enable /props endpoint
    #[arg(long)]
    props: bool,

    /// Enable /metrics endpoint
    #[arg(long)]
    metrics: bool,

    /// Enable model warmup on startup
    #[arg(long = "warmup", overrides_with = "_no_warmup", default_value_t = true)]
    warmup: bool,

    /// Disable model warmup on startup
    #[arg(long = "no-warmup", overrides_with = "warmup", hide = true)]
    _no_warmup: bool,

    // Default sampling parameters.
    /// Default sampling temperature
    #[arg(long = "temp", default_value_t = 0.8)]
    temp: f32,

    /// Default top-K sampling
    #[arg(long, env = "LLAMA_ARG_TOP_K", default_value_t = 40)]
    top_k: i32,

    /// Default top-P (nucleus) sampling
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Default min-P sampling
    #[arg(long, default_value_t = 0.1)]
    min_p: f32,

    /// Random seed (-1 = random)
    #[arg(short = 's', long, default_value_t = -1)]
    seed: i64,

    /// Default repetition penalty lookback window
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Default repetition penalty multiplier
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// Default presence penalty
    #[arg(long, default_value_t = 0.0)]
    presence_penalty: f32,

    /// Default frequency penalty
    #[arg(long, default_value_t = 0.0)]
    frequency_penalty: f32,

    // DRY sampling parameters.
    /// DRY penalty multiplier (0.0 = disabled)
    #[arg(long, default_value_t = 0.0)]
    dry_multiplier: f32,

    /// DRY exponential base
    #[arg(long, default_value_t = 1.75)]
    dry_base: f32,

    /// DRY minimum match length before penalty
    #[arg(long, default_value_t = 2)]
    dry_allowed_length: usize,

    /// DRY lookback window (-1 = full context)
    #[arg(long, default_value_t = -1)]
    dry_penalty_last_n: i32,

    /// DRY sequence breaker token strings (e.g. "\n", "\t")
    #[arg(long, value_delimiter = ',')]
    dry_sequence_breakers: Vec<String>,

    // Logging.
    /// Enable verbose (debug) logging
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Disable all logging
    #[arg(long)]
    log_disable: bool,

    /// Log output file
    #[arg(long, env = "LLAMA_LOG_FILE", value_name = "PATH")]
    log_file: Option<PathBuf>,

    // Distributed inference.
    /// Path to TOML cluster configuration file for distributed inference
    #[arg(long, value_name = "PATH")]
    distributed_config: Option<std::path::PathBuf>,

    /// Role this node plays in the cluster (prefill, decode, pipeline_stage, tensor_parallel_rank, hybrid)
    #[arg(long, value_name = "ROLE")]
    node_role: Option<String>,

    /// Unique identifier for this node in the cluster
    #[arg(long, value_name = "ID")]
    node_id: Option<String>,

    /// Comma-separated list of peer addresses (host:port) for static discovery
    #[arg(long, value_delimiter = ',', value_name = "ADDR")]
    peers: Vec<std::net::SocketAddr>,

    /// Manual pipeline-parallel layer partition (e.g. "0-15,16-31")
    ///
    /// Specifies explicit layer ranges per pipeline stage. Each range is
    /// inclusive on both ends. When omitted, layers are auto-partitioned
    /// proportionally to device memory.
    #[arg(long = "pp-layers", value_name = "RANGES")]
    pp_layers: Option<String>,

    /// Micro-batch size for pipeline parallelism
    ///
    /// Splits incoming batches into micro-batches of this size to fill the
    /// pipeline and reduce bubble time. Smaller values improve pipeline
    /// utilization but add scheduling overhead. Default: 1.
    #[arg(long = "pp-micro-batch-size", default_value_t = 1, value_name = "N")]
    pp_micro_batch_size: usize,

    /// Number of tensor-parallel ranks (must be a power of 2).
    ///
    /// When set to N > 1, model weights are sharded across N in-process ranks.
    /// Current multi-rank runtime support is limited to dense Llama, Qwen2/2.5,
    /// Qwen3, Gemma 3 text, ERNIE 4.5, and Hunyuan v1 Dense models.
    #[arg(long = "tp-size", default_value_t = 1, value_name = "N")]
    tp_size: usize,

    /// MoE expert sharding mode: "expert_parallel" or "within_expert"
    #[arg(
        long = "tp-moe-mode",
        default_value = "expert_parallel",
        value_name = "MODE"
    )]
    tp_moe_mode: String,

    /// Embedding sharding mode: "vocab_parallel" or "replicated".
    ///
    /// The current in-process tensor-parallel runtime requires "replicated".
    #[arg(
        long = "tp-embedding-mode",
        default_value = "replicated",
        value_name = "MODE"
    )]
    tp_embedding_mode: String,

    /// LM head sharding mode: "vocab_parallel" or "replicated".
    ///
    /// The current in-process tensor-parallel runtime requires "replicated".
    #[arg(
        long = "tp-lm-head-mode",
        default_value = "replicated",
        value_name = "MODE"
    )]
    tp_lm_head_mode: String,

    // llama-server compatibility arguments (accepted but ignored).
    /// Accepted for llama-server CLI compatibility (ignored — mlxcel has no web UI)
    #[arg(long, hide = true)]
    _no_webui: bool,

    /// Accepted for llama-server CLI compatibility (ignored — mlxcel always processes templates)
    #[arg(long, hide = true)]
    _jinja: bool,

    /// Accepted for llama-server CLI compatibility (ignored — mlxcel always uses Metal)
    #[arg(long = "n-gpu-layers", hide = true)]
    _n_gpu_layers: Option<i32>,

    /// Accepted for llama-server CLI compatibility (ignored — vision projector loaded automatically)
    #[arg(long, hide = true)]
    _mmproj: Option<String>,

    /// Accepted for llama-server CLI compatibility (ignored)
    #[arg(long, hide = true)]
    _flash_attn: bool,

    /// Accepted for llama-server CLI compatibility (ignored — not applicable to MLX)
    #[arg(long, hide = true)]
    _mlock: bool,

    /// Accepted for llama-server CLI compatibility (ignored — not applicable to MLX)
    #[arg(long = "no-mmap", hide = true)]
    _no_mmap: bool,

    /// Accepted for llama-server CLI compatibility (ignored — mlxcel handles batching internally)
    #[arg(long, hide = true)]
    _cont_batching: bool,

    /// KV cache quantization mode.
    ///
    /// Controls how accumulated key/value tensors are stored:
    ///   fp16  — Standard half-precision storage (default, no overhead).
    ///   int8  — Per-token INT8 absmax quantization; reduces KV cache memory
    ///           by ~50% at the cost of small quantization error per token.
    #[arg(long = "kv-cache-mode", default_value = "fp16", value_name = "MODE")]
    kv_cache_mode: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate(args) => commands::run_generate(args),
        Commands::Serve(args) => commands::run_serve(args),
        Commands::List => {
            print_supported_models();
            Ok(())
        }
    }
}

fn print_supported_models() {
    println!("Supported Model Architectures (57+):\n");

    println!("TRANSFORMER MODELS:");
    println!("  Llama family:     Llama 1/2/3/4, Yi, TinyLlama, Vicuna");
    println!("  Qwen family:      Qwen 2/2.5/3, Qwen MoE variants");
    println!("  Gemma family:     Gemma 1/2/3/3n, RecurrentGemma");
    println!("  Phi family:       Phi 1/2/3/3-small, PhiMoE");
    println!("  Mistral family:   Mistral, Mixtral, Ministral3, Mistral3");
    println!("  DeepSeek:         DeepSeek v1/v2/v3/v3.2, DeepSeek R1");
    println!("  Cohere:           Command R/R+ (Cohere, Cohere2)");
    println!("  InternLM:         InternLM 2/3");
    println!("  GLM:              GLM4, GLM4 MoE");
    println!("  ExaOne:           ExaOne 3/4, ExaOne MoE");
    println!("  OLMo:             OLMo 1/2/3, OLMoE");
    println!("  MiniMax:          MiniMax-M2 (MoE, 256 experts)");
    println!("  Others:           StarCoder2, StableLM, Baichuan, MiniCPM 1/3");
    println!();

    println!("STATE SPACE / RNN MODELS:");
    println!("  Mamba:            Mamba 1/2, Falcon Mamba");
    println!("  RWKV:             RWKV v7");
    println!("  RecurrentGemma:   Griffin hybrid (RGLRU + attention)");
    println!();

    println!("HYBRID MODELS (Attention + SSM/Linear):");
    println!("  Jamba:            Mamba + Transformer + MoE");
    println!("  Qwen3 Next:       Full Attention + GatedDeltaNet + MoE");
    println!("  Nemotron-H:       Mamba2 + Attention + MLP/MoE hybrid");
    println!();

    println!("SPECIALIZED MODELS:");
    println!("  Nemotron:         Nemotron-4, Nemotron-H, Nemotron-NAS");
    println!("  ERNIE:            ERNIE 4.5, ERNIE 4.5 MoE");
    println!("  SmolLM3:          Efficient small model");
    println!("  Hunyuan:          Hunyuan v1 Dense");
    println!("  MiMo:             Multi-token prediction");
    println!();

    println!("For the full list, see: docs/model_implementations.md");
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
