//! OpenAI/llama-server compatible HTTP server for mlxcel

pub mod app;
mod chat_request;
pub mod chat_template;
mod config;
mod media;
pub mod model_provider;
mod request_options;
pub mod routes;
mod startup;
mod state;
mod streaming;
pub mod types;

pub use app::create_app;
pub use chat_template::ChatTemplateProcessor;
pub use config::{ServerConfig, ServerGenerateOptions};
pub use model_provider::{GenerationResult, ModelProvider};
pub use startup::{ServerStartupConfig, start_server};
pub use state::{AppState, Metrics};
