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

use clap::Parser;
use std::path::PathBuf;

use mlxcel::server::{ServerStartupInput, start_server};

/// mlxcel-server: llama-server compatible HTTP server for MLX inference
///
/// Drop-in replacement for llama-server (llama.cpp) using Apple Silicon MLX or
/// CUDA backends. Supports OpenAI-compatible API endpoints and llama-server
/// native endpoints.
#[derive(Parser, Debug)]
#[command(
    name = "mlxcel-server",
    author = "Lablup Inc.",
    version,
    about = "llama-server compatible HTTP server for MLX inference on Apple Silicon and CUDA GPUs",
    after_help = "\
Tensor Parallel Runtime:
  Current multi-rank support: dense Llama, Qwen2/2.5, Qwen3, Gemma 3 text, ERNIE 4.5, Hunyuan v1 Dense
  Current constraints: --tp-embedding-mode replicated, --tp-lm-head-mode replicated
                       LoRA unsupported, sequential worker forced for tp_size > 1"
)]
struct Args {
    /// Path to the model directory
    #[arg(
        short = 'm',
        long = "model",
        env = "LLAMA_ARG_MODEL",
        value_name = "PATH"
    )]
    model: PathBuf,

    /// Model alias (shown in API responses instead of directory name)
    #[arg(
        short = 'a',
        long = "alias",
        env = "LLAMA_ARG_ALIAS",
        value_name = "NAME"
    )]
    alias: Option<String>,

    /// Path to LoRA adapter directory
    #[arg(long = "lora", value_name = "PATH")]
    lora: Option<PathBuf>,

    /// Host address to bind to (or Unix socket path when --port 0)
    #[arg(long, env = "LLAMA_ARG_HOST", default_value = "127.0.0.1")]
    host: String,

    /// Port number to listen on (0 = Unix socket mode using --host as socket path)
    #[arg(long, env = "LLAMA_ARG_PORT", default_value_t = 8080)]
    port: u16,

    /// Context size limit (0 = use model default)
    #[arg(
        short = 'c',
        long = "ctx-size",
        env = "LLAMA_ARG_CTX_SIZE",
        default_value_t = 0
    )]
    ctx_size: usize,

    /// Maximum tokens to predict (-1 = unlimited)
    #[arg(
        short = 'n',
        long = "predict",
        visible_alias = "n-predict",
        env = "LLAMA_ARG_N_PREDICT",
        default_value_t = -1
    )]
    predict: i32,

    /// Number of parallel request slots
    #[arg(long = "parallel", env = "LLAMA_ARG_N_PARALLEL", default_value_t = 1)]
    parallel: usize,

    /// API key for authentication
    #[arg(long = "api-key", env = "LLAMA_API_KEY", value_name = "KEY")]
    api_key: Option<String>,

    /// Path to file containing API key
    #[arg(long = "api-key-file", value_name = "PATH")]
    api_key_file: Option<PathBuf>,

    /// Request timeout in seconds
    #[arg(long, env = "LLAMA_ARG_TIMEOUT", default_value_t = 600)]
    timeout: u64,

    /// Path to draft model for speculative decoding
    #[arg(
        long = "model-draft",
        env = "LLAMA_ARG_MODEL_DRAFT",
        value_name = "PATH"
    )]
    model_draft: Option<PathBuf>,

    /// Maximum number of draft tokens per speculation step
    #[arg(
        long = "draft",
        visible_alias = "draft-max",
        env = "LLAMA_ARG_DRAFT_MAX",
        default_value_t = 16
    )]
    draft: usize,

    /// Maximum number of concurrent decode sequences (default: --parallel value)
    #[arg(long = "max-batch-size", value_name = "N")]
    max_batch_size: Option<usize>,

    /// Disable continuous batching and use the legacy sequential worker.
    ///
    /// When set, requests are processed one at a time in FIFO order with no
    /// batch scheduler overhead. Equivalent to using `--max-batch-size 1` but
    /// with explicit sequential semantics and no prefill chunking. This is
    /// forced automatically when `--tp-size > 1`.
    #[arg(long = "no-batch")]
    no_batch: bool,

    /// Maximum number of requests waiting in the prefill queue (default: 32)
    #[arg(long = "max-queue-depth", default_value_t = 32)]
    max_queue_depth: usize,

    /// Prefill chunk size in tokens (0 = disabled, default: 512)
    #[arg(long = "prefill-chunk-size", default_value_t = 512)]
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
    #[arg(long = "enable-preemption")]
    enable_preemption: bool,

    /// Preemption policy: "longest-first" (default) or "lowest-priority"
    #[arg(long = "preemption-policy", default_value = "longest-first")]
    preemption_policy: String,

    /// Maximum number of requests to batch together for prefill (default: 1)
    ///
    /// When > 1, the scheduler collects up to this many pending requests and
    /// runs a single batched forward pass [batch_size, max_seq_len] for better
    /// Neural Accelerator utilization. Recommended: 4-8 on M5 Pro/Max hardware.
    #[arg(long = "max-batch-prefill", default_value_t = 1)]
    max_batch_prefill: usize,

    /// Override chat template (Jinja2 template string)
    #[arg(long = "chat-template", value_name = "TEMPLATE")]
    chat_template: Option<String>,

    /// Path to chat template file
    #[arg(long = "chat-template-file", value_name = "PATH")]
    chat_template_file: Option<PathBuf>,

    /// Enable /slots endpoint
    #[arg(long = "slots", overrides_with = "_no_slots", default_value_t = true)]
    slots: bool,

    /// Disable /slots endpoint
    #[arg(long = "no-slots", overrides_with = "slots", hide = true)]
    _no_slots: bool,

    /// Enable /props endpoint
    #[arg(long = "props")]
    props: bool,

    /// Enable /metrics endpoint
    #[arg(long = "metrics")]
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
    #[arg(long = "top-k", env = "LLAMA_ARG_TOP_K", default_value_t = 40)]
    top_k: i32,

    /// Default top-P (nucleus) sampling
    #[arg(long = "top-p", default_value_t = 0.9)]
    top_p: f32,

    /// Default min-P sampling
    #[arg(long = "min-p", default_value_t = 0.1)]
    min_p: f32,

    /// Random seed (-1 = random)
    #[arg(short = 's', long = "seed", default_value_t = -1)]
    seed: i64,

    /// Default repetition penalty lookback window
    #[arg(long = "repeat-last-n", default_value_t = 64)]
    repeat_last_n: usize,

    /// Default repetition penalty multiplier
    #[arg(long = "repeat-penalty", default_value_t = 1.0)]
    repeat_penalty: f32,

    /// Default presence penalty
    #[arg(long = "presence-penalty", default_value_t = 0.0)]
    presence_penalty: f32,

    /// Default frequency penalty
    #[arg(long = "frequency-penalty", default_value_t = 0.0)]
    frequency_penalty: f32,

    // DRY sampling parameters.
    /// DRY penalty multiplier (0.0 = disabled)
    #[arg(long = "dry-multiplier", default_value_t = 0.0)]
    dry_multiplier: f32,

    /// DRY exponential base
    #[arg(long = "dry-base", default_value_t = 1.75)]
    dry_base: f32,

    /// DRY minimum match length before penalty
    #[arg(long = "dry-allowed-length", default_value_t = 2)]
    dry_allowed_length: usize,

    /// DRY lookback window (-1 = full context)
    #[arg(long = "dry-penalty-last-n", default_value_t = -1)]
    dry_penalty_last_n: i32,

    /// DRY sequence breaker token strings (e.g. "\n", "\t")
    #[arg(long = "dry-sequence-breaker", value_delimiter = ',')]
    dry_sequence_breakers: Vec<String>,

    // Logging.
    /// Enable verbose (debug) logging
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,

    /// Disable all logging
    #[arg(long = "log-disable")]
    log_disable: bool,

    /// Log output file
    #[arg(long = "log-file", env = "LLAMA_LOG_FILE", value_name = "PATH")]
    log_file: Option<PathBuf>,

    // Distributed inference.
    /// Path to TOML cluster configuration file for distributed inference
    #[arg(long, value_name = "PATH")]
    distributed_config: Option<PathBuf>,

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

    /// Number of tensor-parallel ranks (must be a power of 2).
    ///
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    start_server(build_startup_input(args).into_startup_config()).await
}

fn build_startup_input(args: Args) -> ServerStartupInput {
    ServerStartupInput {
        model_path: args.model,
        adapter_path: args.lora,
        model_alias: args.alias,
        host: args.host,
        port: args.port,
        api_key: args.api_key,
        api_key_file: args.api_key_file,
        n_parallel: args.parallel,
        ctx_size: args.ctx_size,
        n_predict: args.predict,
        timeout: args.timeout,
        draft_model_path: args.model_draft,
        draft_max: args.draft,
        max_batch_size: args.max_batch_size,
        no_batch: args.no_batch,
        max_queue_depth: args.max_queue_depth,
        prefill_chunk_size: args.prefill_chunk_size,
        batch_size: args.batch_size,
        ubatch_size: args.ubatch_size,
        enable_preemption: args.enable_preemption,
        preemption_policy: args.preemption_policy,
        max_batch_prefill: args.max_batch_prefill,
        chat_template: args.chat_template,
        chat_template_file: args.chat_template_file,
        slots: args.slots,
        no_slots: args._no_slots,
        props: args.props,
        metrics: args.metrics,
        warmup: args.warmup,
        no_warmup: args._no_warmup,
        temperature: args.temp,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
        repeat_last_n: args.repeat_last_n,
        repeat_penalty: args.repeat_penalty,
        presence_penalty: args.presence_penalty,
        frequency_penalty: args.frequency_penalty,
        dry_multiplier: args.dry_multiplier,
        dry_base: args.dry_base,
        dry_allowed_length: args.dry_allowed_length,
        dry_penalty_last_n: args.dry_penalty_last_n,
        dry_sequence_breakers: args.dry_sequence_breakers,
        verbose: args.verbose,
        log_disable: args.log_disable,
        log_file: args.log_file,
        distributed_config: args.distributed_config,
        node_role: args.node_role,
        node_id: args.node_id,
        peers: args.peers,
        pp_layers: args.pp_layers,
        tp_size: args.tp_size,
        tp_moe_mode: args.tp_moe_mode,
        tp_embedding_mode: args.tp_embedding_mode,
        tp_lm_head_mode: args.tp_lm_head_mode,
    }
}
