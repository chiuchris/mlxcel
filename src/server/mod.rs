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

//! OpenAI/llama-server compatible HTTP server for mlxcel

pub mod app;
pub mod batch;
mod chat_request;
pub mod chat_template;
pub mod chat_template_kwargs;
mod cli_input;
mod config;
mod media;
pub mod model_provider;
mod request_options;
pub mod routes;
mod startup;
mod state;
mod streaming;
pub mod thinking_budget;
pub mod tool_calls;
pub mod types;

pub use app::create_app;
pub use chat_template::ChatTemplateProcessor;
pub use chat_template_kwargs::{
    ChatTemplateKwargs, ChatTemplateKwargsError, LLAMA_ARG_CHAT_TEMPLATE_KWARGS,
    env_fallback_chat_template_kwargs,
};
pub use cli_input::{
    ServerStartupInput, env_fallback_lang_bias, env_fallback_lang_bias_include_byte_fragments,
    env_fallback_reasoning_budget,
};
pub use config::{
    DecodeStorageBackend, PipelineParallelRuntimeConfig, PreemptionPolicy,
    RemotePipelineStageConfig, ServerConfig, ServerGenerateOptions,
};
pub use model_provider::{GenerationResult, ModelProvider};
pub use startup::{ServerStartupConfig, start_server};
pub use state::{AppState, BatchMetrics, Metrics};
