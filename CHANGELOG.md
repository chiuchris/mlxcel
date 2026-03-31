# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v0.0.13] - 2026-03-31

### Added
- Mistral4 MLA (Multi-head Latent Attention) language model support (#144)
- Molmo-Point VLM model support (#148)
- NemotronSuper model support (upstream mlx-lm sync) (#131)
- `sync-upstream` Claude Code command for tracking mlx-lm/mlx-vlm changes

### Changed
- Fuse GatedDeltaNet decode step with `mlx::core::compile` for improved throughput
- Apply MRoPE and position ID optimizations to Qwen3-VL-MoE
- Fast-path single-token decode position IDs in Qwen3-VL
- Vectorize Qwen3-VL interleaved MRoPE with `take_along_axis`
- Optimize VLM vision encoding and sampling pipeline (#149)
- Use SDPA for NemotronH attention, boosting decode throughput 59%

### Fixed
- Improve SSM/Mamba2 numerical precision with float32 dt computation (#133)
- Improve GatedDelta numerical precision with float32 state (#132)
- Resolve Mamba/NemotronNAS output corruption with softplus overflow and fused norm grouping
- Guard Qwen3.5 GatedDeltaNet state batch dimension mismatches (#145)
- Use `h.shape` instead of `inputs.shape` for Ministral3 attn_scale (#146)
- Document scalar offset invariant for Llama4 BatchKVCache compatibility (#147)
- Correct model_tests.md table placement and dedup nemotron entries

## [v0.0.12] - 2026-03-26

### Added
- Compiled C++ operations using `mlx::core::compile(shapeless=true)` for small model throughput:
  - `compiled_gelu` / `compiled_gelu_approx`: fused GELU activation kernels
  - `compiled_geglu_activation`: fused GELU-gated activation (`gelu(gate) * x`)
  - `compiled_softcap`: fused softcap (`tanh(x/cap)*cap`) for Gemma2
  - `compiled_softcap_sdpa`: entire attention path with softcap fused into single compiled graph
  - `compiled_softcap_sdpa_gqa`: fused GQA + softcap SDPA variant
  - `compiled_clip_residual`: fused float16-safe residual addition for Gemma3
  - `compiled_gelu_mlp_forward`: full GELU MLP as single compiled graph
- `UnifiedLinear::quantized_weight()` accessor for compiled MLP kernel dispatch
- Distributed inference framework: node discovery, cluster configuration, tensor/pipeline parallelism, disaggregated serving
- Comprehensive mkdocs documentation site (EN/KO) with PDF export
- Project-specific Claude Code commands and skills

### Changed
- Gemma3: fused SDPA, pre-computed GemmaRMSNorm, skip decode masks — Gemma3 1B reaches 94% of Python mlx-lm
- Gemma2: uses `compiled_softcap_sdpa_gqa` with internal GQA head expansion
- StarCoder2: uses `compiled_gelu` activation
- Phi3: pre-compute SuScaledRoPE scale array at load time
- Hoist env var checks out of generation hot loop
- Incremental token history and cached EOS in BatchScheduler
- Use MLX native `load_safetensors()` for faster weight loading
- Optimize model loading with batched synchronization

### Fixed
- OpenAI API streaming response format compatibility
- Guard compiled MLP/MoE paths against non-standard quantization params (`group_size != 64` or `bits != 4`)

## [v0.0.11] - 2026-03-18

### Added
- Compiled kernel fusion for `relu_squared` and `silu` activation functions
- Compiled kernel fusion for MoE gate and `compute_dt` operations
- Fused SSM Metal kernel for Mamba2 single-token decode
- Compiled MoE gate function for NemotronH
- Fused MoE forward function for NemotronH
- Fused Mamba2 mixer forward for NemotronH
- NemotronH full-forward C++ decode path (experimental, disabled)
- `MLXCEL_FORCE_SYNC` debug flag for pipelining analysis
- `MLXCEL_PROFILE_PIPELINE` for precise build/wait timing
- Per-block and build/eval profiling for NemotronH

### Fixed
- Auto-cast SDPA mask to Q dtype, preventing mask type errors across models
- Load float16 weights natively on Metal (was converting to float32)
- Eliminate float32 type promotion across all models
- Prevent float32 type promotion in NemotronH hidden states
- Add affine fast-path for quantized_matmul (omit mode parameter)
- Correct mlx-lm benchmark baselines and update nemotron/mamba results

### Changed
- Optimize Mamba single-token decode path and remove unnecessary copies

## [v0.0.10] - 2026-03-17

### Fixed
- ExaOne4: Cast causal mask to bfloat16 to match model weights dtype (MLX SDPA requires mask type to promote to output type)
- StableLM: Read `eos_token_id` from config.json instead of hardcoding 0, fixing premature 1-token generation

### Changed
- Add static mode string pool for quantized ops to avoid per-call heap allocation in C++ bridge hot path

## [v0.0.9] - 2026-03-17

### Added
- GptOss MoE model with sinks SDPA support
- MXFP4/NVFP4/MXFP8 quantization mode support across FFI bridge and model layers
- GPT-OSS benchmark results to model test documentation

### Fixed
- Set wired memory limit to `gpu_max_memory_size` by default

### Changed
- Re-benchmark all models after wired limit fix

## [v0.0.8] - 2026-03-17

### Fixed
- Support explicit `head_dim` config field in Qwen3-VL, Qwen2-VL, and Qwen2-MoE models — fixes Qwen3-VL-32B crash where `head_dim(128) != hidden_size/num_heads(80)`
- Switch macOS CI runner to macos-15 for Xcode 16+ C++20 ranges support

### Changed
- Add CUDA release pipeline and refresh benchmark report with MoE results

## [v0.0.7] - 2026-03-16

### Added
- GatherMM/GatherQMM for MoE model support on CUDA (#34)
- CUDA bf16 support: type promotion table patching, mixed-precision binary kernels, normalization ops, reduce accumulation with fp32 precision, native bf16 array creation in bridge layer (#42-#46)
- CUDA bf16 validation scripts and documentation (#47)
- CUDA GB10 benchmark results for 57 models
- GB10 vs M1 Ultra benchmark comparison report
- `--batch-size` and `--ubatch-size` as llama-server compatible aliases (#32)
- Debian packaging, man pages, and optimized release profile
- CUDA build guide and build troubleshooting documentation (#33)

### Fixed
- CUDA qmv shared-memory optimization with block.sync() fix
- CUDA dtype and fp16 bridge fixes
- C++ bridge build: removed `-flto`, upgraded to C++20
- C++ bridge LTO enabled only on macOS

### Changed
- Bumped MLX to v0.31.1, GPU backend now shown in runtime display
- CUDA qmv kernel optimized with shared memory x-broadcast and `__restrict__`
- Phase 19 CUDA optimization report and final benchmarks

## [v0.0.6] - 2026-03-14

### Added
- Continuous batching with iteration-level BatchScheduler for concurrent request handling
- Request lifecycle types and sequence state machine for batch management
- Per-sequence KV cache isolation and CachePool for independent request processing
- Tensor-batched decode forward pass for efficient multi-sequence generation
- Preemptive scheduling and chunked prefill for better latency and throughput
- HTTP server integration with batch scheduler and concurrency support
- Explicit `forward_batched()` for Qwen3 with split-attention support
- Continuous batching benchmarks and observability instrumentation
- Feature gate for batching to preserve CLI single-request path

### Fixed
- Scheduling policy now admits queued requests to grow batch beyond initial size

### Changed
- Added continuous batching development guide and benchmark comparison documentation
- Benchmark results for 84 models with scheduler fix improvements

## [v0.0.5] - 2026-03-11

### Added
- Phi4-SigLIP vision-language model support with NaFlex-style patch processor and SigLIP2 vision tower
- Phi4MM vision-language model support with SigLIP + HD transform + AvgPool2d pipeline
- MiniCPM-o vision-language model support with SigLIP + Perceiver-style resampler
- Moondream3 vision-language model support with packed int4 dequantization and BOS-prefix prompting
- Runtime LoRA support on Linear layers with `Cell<bool>` active toggle for on-the-fly application
- `after_prefill()` dispatch through LoadedModel enum and LanguageModel trait
- Server support for data URIs, file URLs, bare local paths, and http(s) image fetches

### Fixed
- Phi4MM VLM: add SuScaledRoPE (longrope) to Phi3 attention for correct positional encoding
- Phi4MM VLM: fix image token placement in prompt (insert after `<|user|>` tag, not before entire prompt)
- Phi4MM VLM: use runtime LoRA instead of weight fusion, matching Python PEFT behavior
- MiniCPM-o VLM: switch text backbone from Qwen3-VL (MRoPE) to standard Qwen3 (standard RoPE)
- MiniCPM-o VLM: add automatic Qwen3-style chat template wrapping for models without chat_template
- Moondream3 VLM: fix RoPE layout (NeoX-style halves), attention mask dtype, and vision tiling
- Moondream3 VLM: use exact GELU for tau scaling and MoE GeGLU matching Python F.gelu

### Changed
- Synced mlx-vlm upstream Qwen-VL: fused-SDPA head-dim padding in shared Qwen3-VL vision encoder
- Refactored server image extraction into async edge helpers with multi-format support

## [v0.0.4] - 2026-03-10

### Added
- Tiktoken BPE tokenizer support for models using `.tiktoken` vocabulary files (HunYuan MoE 13B)
- Quality gate entry point script (`scripts/run_quality_gate.sh`) with `--include-serial-helpers` and `--full` modes
- Comprehensive model validation: 71/74 local models pass (95.9%)

### Fixed
- Solar Open 100B-4bit config parsing: add serde defaults for `n_group`/`topk_group` in GLM4 MoE config
- GatedDeltaNet `RMSNormGated`: promote SwiGLU gate path to float32 before restoring hidden-state dtype (upstream mlx-lm parity for Qwen3Next/Qwen3.5)
- Step3p5 sliding-window layers now use `RotatingKVCache` instead of plain `KVCache`
- Suppress deprecated-copy warning in mlxcel-core build for MLX v0.31.0

### Changed
- Converged model registration: centralized config-backed text model registration in `src/model_metadata.rs`
- Split mlxcel-core internals into focused modules: `cache.rs`, `ops.rs`, `dtype.rs`, `sampling.rs`, `generation_policy.rs`, `streams.rs`
- Extracted large-model helper hotspots: `gemma3n_helpers.rs`, `llama4_helpers.rs`, `qwen3_next_helpers.rs`
- Split `LoadedModel` capabilities into `loaded_model_capabilities.rs` with `VlmRuntimeRef`
- Separated model detection (`detection.rs`) and sanitization (`sanitize.rs`) helpers
- Unified model loading descriptors with `StaticModelDescriptor` and `model_load_policy()`
- Normalized server startup edge inputs into `cli_input.rs`
- Removed unsafe `Send`/`Sync` auto traits from `ModelProvider`
- Strengthened vision merge contracts with dedicated tests
- Refreshed architecture, control-plane guide, and model addition documentation

## [v0.0.3] - 2026-03-10

### Fixed
- Streaming UTF-8 corruption for multi-byte characters (e.g., Korean, CJK) caused by byte-level BPE token boundaries
- Default `max_tokens` increased from 512 to 4096 so thinking models produce complete responses
- Release archive now includes `mlx.metallib` for Metal GPU acceleration

## [v0.0.2] - 2026-03-10

### Added
- Solar Open 100B INT4 model support with GPTQ conversion
- MiniMax-M2 MoE model support

### Fixed
- GPU wired memory limit now opt-in via `MLXCEL_WIRED_LIMIT` environment variable
- Llama4 vision encoder now uses UnifiedLinear to support quantized weights
- Molmo2 VLM inherits quantization config correctly; stale examples updated
- PaliGemma2 VLM no longer produces pad/EOS tokens instead of correct output
- Qwen3.5 VLM loader variants corrected
- Resolved all clippy warnings in vision and loading modules

### Changed
- Major codebase refactoring: modularized server, CLI, loader, and multimodal paths
- Extracted loader modules into `src/loading/` directory (SigLIP, Pixtral, Gemma, LLaVA, Qwen VLM loaders)
- Moved CLI command handlers under `src/commands/`
- Grouped execution policy helpers under `src/execution/`
- Grouped multimodal helpers under `src/multimodal/`
- Split server into config, state, streaming, and media helper modules
- Centralized LoadedModel embedding dispatch and reduced accessor boilerplate
- Shared sampling config assembly across CLI and server
- Refined model detection helpers with added guide
- Refreshed architecture and vision documentation

## [v0.0.1] - 2026-03-07

Initial public release of mlxcel.

### Added
- 59+ model architectures: Transformers, MoE, SSM/RNN, and Hybrid models
- Vision-Language Model support: Gemma 3, LLaVA, Llama 4, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Pixtral, Phi-3.5 Vision, and more
- OpenAI-compatible HTTP server with SSE streaming
- `mlxcel-server` standalone binary as llama-server drop-in replacement
- LoRA adapter loading and fusion at runtime
- Speculative decoding with draft models
- Advanced sampling: Top-P, Top-K, Min-P, XTC, DRY penalty, repetition/frequency/presence penalties
- Chat template support via Jinja2 (minijinja)
- Unix domain socket support for server mode
- EOS token detection from generation_config.json
- SentencePiece tokenizer support
- Linux + CUDA backend support (CUDA 12.0+, cuDNN 9+)
- Direct MLX C++ bindings via cxx FFI (zero Python dependencies)
- Pre-allocated KV cache with slice_update for O(1) per-token performance
- Sliding window and rotating KV cache support
- UnifiedLinear layer supporting both quantized and non-quantized models
- GitHub Actions release workflow for macOS ARM64
- Profile mode for prefill/decode timing analysis

[v0.0.13]: https://github.com/lablup/mlxcel/compare/v0.0.12...v0.0.13
[v0.0.12]: https://github.com/lablup/mlxcel/compare/v0.0.11...v0.0.12
[v0.0.11]: https://github.com/lablup/mlxcel/compare/v0.0.10...v0.0.11
[v0.0.10]: https://github.com/lablup/mlxcel/compare/v0.0.9...v0.0.10
[v0.0.9]: https://github.com/lablup/mlxcel/compare/v0.0.8...v0.0.9
[v0.0.8]: https://github.com/lablup/mlxcel/compare/v0.0.7...v0.0.8
[v0.0.7]: https://github.com/lablup/mlxcel/compare/v0.0.6...v0.0.7
[v0.0.6]: https://github.com/lablup/mlxcel/compare/v0.0.5...v0.0.6
[v0.0.5]: https://github.com/lablup/mlxcel/compare/v0.0.4...v0.0.5
[v0.0.4]: https://github.com/lablup/mlxcel/compare/v0.0.3...v0.0.4
[v0.0.3]: https://github.com/lablup/mlxcel/compare/v0.0.2...v0.0.3
[v0.0.2]: https://github.com/lablup/mlxcel/compare/v0.0.1...v0.0.2
[v0.0.1]: https://github.com/lablup/mlxcel/releases/tag/v0.0.1
