# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

[v0.0.4]: https://github.com/lablup/mlxcel/compare/v0.0.3...v0.0.4
[v0.0.3]: https://github.com/lablup/mlxcel/compare/v0.0.2...v0.0.3
[v0.0.2]: https://github.com/lablup/mlxcel/compare/v0.0.1...v0.0.2
[v0.0.1]: https://github.com/lablup/mlxcel/releases/tag/v0.0.1
