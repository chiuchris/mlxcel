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

use mlxcel::lang_bias::LangBiasCliArgs;
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
  Current multi-rank support: dense Llama, Qwen2/2.5, Qwen3, Qwen3.5 text, Gemma 3 text, Gemma 4 text, ERNIE 4.5, Hunyuan v1 Dense
  Current constraints: --tp-embedding-mode replicated, --tp-lm-head-mode replicated
                       LoRA unsupported, server batching supported for listed dense runtimes
                       except Gemma 4 E2B-style conservative fallback checkpoints

Remote Pipeline Parallel Example (TCP):
  1. Generate a shared cluster config:
       CLUSTER_NAME=studio-pp \\
       TRANSPORT_BACKEND=tcp \\
       COORDINATOR_CONTROL_ADDR=192.168.1.22:19000 \\
       STAGE0_ADDR=192.168.1.22:19001 \\
       STAGE1_ADDR=192.168.1.24:19001 \\
       scripts/benchmark_pipeline_remote_rollout.sh write-config \\
         examples/distributed/generated_pipeline_remote_2node_tcp.toml

  2. Start stage-1 on machine B:
       mlxcel-server -m models/llama-3.2-1b-4bit \\
         --distributed-config examples/distributed/generated_pipeline_remote_2node_tcp.toml \\
         --node-id stage-1 --host 0.0.0.0 --port 18081 --no-warmup

  3. Start stage-0 on machine A:
       mlxcel-server -m models/llama-3.2-1b-4bit \\
         --distributed-config examples/distributed/generated_pipeline_remote_2node_tcp.toml \\
         --node-id stage-0 --host 0.0.0.0 --port 18081 --no-warmup

  4. Start the coordinator on machine A:
       mlxcel-server -m models/llama-3.2-1b-4bit --alias llama-remote-pp \\
         --distributed-config examples/distributed/generated_pipeline_remote_2node_tcp.toml \\
         --node-id coordinator --host 0.0.0.0 --port 18080 \\
         --parallel 2 --max-batch-size 2 --pp-micro-batch-size 2 \\
         --metrics --no-warmup

Thunderbolt mode:
  Use the same workflow with TRANSPORT_BACKEND=thunderbolt and each node's
  Thunderbolt Bridge IP (for example 169.254.x.x). The current Thunderbolt
  path uses the shared TCP transport core over the Bridge network.

See also: docs/PIPELINE_PARALLELISM.md"
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
    /// with explicit sequential semantics and no prefill chunking.
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

    /// Micro-batch size for single-machine pipeline execution.
    #[arg(long = "pp-micro-batch-size", default_value_t = 1, value_name = "N")]
    pp_micro_batch_size: usize,

    /// Zero-config pipeline-parallel bring-up: declare the desired number of stages.
    ///
    /// When set (N >= 2), `mlxcel-server` acts as the coordinator and resolves
    /// peers either from `--cluster-peers` or via `--cluster-discovery=mdns`,
    /// allocates ports for the coordinator control plane and stage data ports
    /// if they are not explicitly provided, and emits a deterministic cluster
    /// TOML to `--cluster-config-out` before starting the server. The flag is
    /// mutually exclusive with `--distributed-config`.
    #[arg(long = "pp-auto", value_name = "N")]
    pp_auto: Option<u32>,

    /// Peer role for zero-config pipeline bring-up: register with the coordinator
    /// instead of starting a server of our own.
    ///
    /// When set, `mlxcel-server` announces its availability (either statically
    /// by registering against a known coordinator address, or via broadcast
    /// when `--cluster-discovery=mdns`) and then starts a pipeline stage
    /// service using the stage assignment the coordinator returns.
    #[arg(long = "pp-peer")]
    pp_peer: bool,

    /// Cluster discovery mechanism: "static" (default) or "mdns" for UDP broadcast.
    ///
    /// "static" consumes `--cluster-peers` verbatim. "mdns" enables opt-in
    /// LAN peer discovery via UDP broadcast. The name is retained for future
    /// zeroconf compatibility; today the implementation uses plain UDP so no
    /// extra dependency is required.
    #[arg(
        long = "cluster-discovery",
        default_value = "static",
        value_name = "MODE"
    )]
    cluster_discovery: String,

    /// Human-readable cluster name used to scope discovery and as the TOML header.
    ///
    /// Defaults to the value embedded in the generated TOML when `--pp-auto`
    /// runs (currently `mlxcel-cluster`). Peers with a mismatching name are
    /// ignored by the coordinator during mDNS discovery.
    #[arg(long = "cluster-name", value_name = "NAME")]
    cluster_name: Option<String>,

    /// Static peer addresses for zero-config bring-up (host:port, comma-separated).
    ///
    /// Each peer address should point at the control+data socket that the
    /// corresponding `mlxcel-server --pp-peer` exposes. Ignored when
    /// `--cluster-discovery=mdns` fully resolves the expected peer count.
    #[arg(long = "cluster-peers", value_delimiter = ',', value_name = "ADDR")]
    cluster_peers: Vec<std::net::SocketAddr>,

    /// UDP port for the discovery beacon when `--cluster-discovery=mdns` is used.
    #[arg(long = "cluster-discovery-port", value_name = "PORT")]
    cluster_discovery_port: Option<u16>,

    /// Coordinator control-plane bind address for zero-config bring-up (host:port).
    ///
    /// Kept deliberately distinct from the HTTP listen address so operators do
    /// not have to co-schedule two services on a single port.
    #[arg(long = "cluster-control-addr", value_name = "ADDR")]
    cluster_control_addr: Option<std::net::SocketAddr>,

    /// Output path for the emitted cluster TOML.
    ///
    /// Defaults to `<current directory>/.mlxcel/cluster.toml` when
    /// `--pp-auto` is used and this flag is omitted.
    #[arg(long = "cluster-config-out", value_name = "PATH")]
    cluster_config_out: Option<PathBuf>,

    /// Plan the cluster topology and emit the TOML without starting workers.
    ///
    /// Exits with non-zero status when port, version, or peer-count conflicts
    /// cannot be resolved. Only meaningful in combination with `--pp-auto`.
    #[arg(long = "dry-run", default_value_t = false)]
    dry_run: bool,

    /// Number of tensor-parallel ranks (must be a power of 2).
    ///
    /// Current multi-rank runtime support is limited to dense Llama, Qwen2/2.5,
    /// Qwen3, Qwen3.5 text, Gemma 3 text, Gemma 4 text, ERNIE 4.5, and
    /// Hunyuan v1 Dense models.
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

    /// Maximum number of cached post-projection image features per loaded VLM.
    ///
    /// Multi-turn conversations that revisit the same image reuse cached
    /// vision features and skip the vision tower + multimodal embedder on
    /// subsequent turns. `0` disables caching. Default: 20.
    #[arg(long = "vision-cache-size", default_value_t = 20, value_name = "N")]
    vision_cache_size: usize,

    /// Enable experimental elastic pipeline-parallel repartitioning.
    ///
    /// When set, `mlxcel-server` constructs a repartition coordinator (see
    /// `docs_internal/architecture/elastic-pipeline-repartition-20260418.md`)
    /// that can drain in-flight requests, recompute the partition plan, and
    /// reload layer weights without a full cluster restart. Off by default —
    /// v1 is explicitly opt-in.
    #[arg(long = "enable-elastic-pp", default_value_t = false)]
    enable_elastic_pp: bool,

    /// Maximum wait (seconds) for in-flight requests to drain during an
    /// elastic repartition. Only meaningful with `--enable-elastic-pp`.
    #[arg(
        long = "elastic-pp-drain-timeout",
        default_value_t = 120,
        value_name = "SECONDS"
    )]
    elastic_pp_drain_timeout: u64,

    /// Memory usage fraction above which a memory-pressure trigger fires.
    /// Values outside (0.0, 1.0] are clamped. Default: 0.92. Only meaningful
    /// with `--enable-elastic-pp`.
    #[arg(
        long = "elastic-pp-pressure-fraction",
        default_value_t = 0.92,
        value_name = "FRACTION"
    )]
    elastic_pp_pressure_fraction: f64,

    /// Cool-down (seconds) between successive memory-pressure repartition
    /// triggers on the same stage. Explicit operator triggers bypass this
    /// debounce. Default: 30. Only meaningful with `--enable-elastic-pp`.
    #[arg(
        long = "elastic-pp-cool-down",
        default_value_t = 30,
        value_name = "SECONDS"
    )]
    elastic_pp_cool_down: u64,

    /// Enable `/metrics` and advertise the port operators should scrape.
    ///
    /// Currently the Prometheus endpoint is multiplexed onto the same HTTP
    /// port as the OpenAI API. Passing this flag enables the endpoint.
    /// A warning is logged when the requested port differs from `--port`
    /// because a separate socket is deferred to a follow-up rollout.
    #[arg(long = "metrics-port", value_name = "PORT")]
    metrics_port: Option<u16>,

    /// Write a chrome-tracing-compatible JSON trace of pipeline scheduler
    /// actions (batch arrival, stage enter/exit, activation send/receive,
    /// admission reject) to this file for offline analysis in
    /// `chrome://tracing` or Perfetto.
    #[arg(long = "debug-pp-trace", value_name = "PATH")]
    debug_pp_trace: Option<PathBuf>,

    /// Axis B Epic #362 (B8): language-bias options for server-wide output
    /// steering. See `--lang-bias`, `--lang-bias-config`, `--lang-bias-policy`,
    /// and the `--lang-bias-include-*` family of flags.
    #[command(flatten)]
    lang_bias: LangBiasCliArgs,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    start_server(build_startup_input(args)?.into_startup_config()).await
}

fn build_startup_input(args: Args) -> anyhow::Result<ServerStartupInput> {
    // Axis B (B8): resolve once up-front so CLI errors surface before the
    // server starts listening. Baseline path returns `None` (bit-exact).
    let lang_bias_config = args
        .lang_bias
        .resolve()
        .map_err(|e| anyhow::anyhow!("--lang-bias: {e}"))?;

    Ok(ServerStartupInput {
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
        pp_micro_batch_size: args.pp_micro_batch_size,
        pp_auto: args.pp_auto,
        pp_peer: args.pp_peer,
        cluster_discovery: args.cluster_discovery,
        cluster_name: args.cluster_name,
        cluster_peers: args.cluster_peers,
        cluster_discovery_port: args.cluster_discovery_port,
        cluster_control_addr: args.cluster_control_addr,
        cluster_config_out: args.cluster_config_out,
        dry_run: args.dry_run,
        tp_size: args.tp_size,
        tp_moe_mode: args.tp_moe_mode,
        tp_embedding_mode: args.tp_embedding_mode,
        tp_lm_head_mode: args.tp_lm_head_mode,
        vision_cache_size: args.vision_cache_size,
        enable_elastic_pp: args.enable_elastic_pp,
        elastic_pp_drain_timeout: args.elastic_pp_drain_timeout,
        elastic_pp_pressure_fraction: args.elastic_pp_pressure_fraction,
        elastic_pp_cool_down: args.elastic_pp_cool_down,
        metrics_port: args.metrics_port,
        debug_pp_trace: args.debug_pp_trace,
        lang_bias_config,
    })
}
