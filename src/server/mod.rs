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
mod cli_input;
mod config;
mod media;
pub mod model_provider;
mod request_options;
pub mod routes;
mod startup;
mod state;
mod streaming;
pub mod tool_calls;
pub mod types;

pub use app::create_app;
pub use chat_template::ChatTemplateProcessor;
pub use cli_input::ServerStartupInput;
pub use config::{PreemptionPolicy, ServerConfig, ServerGenerateOptions};
pub use model_provider::{GenerationResult, ModelProvider};
pub use startup::{ServerStartupConfig, start_server};
pub use state::{AppState, BatchMetrics, Metrics};
