//! HTTP route handlers

pub mod chat;
pub mod completions;
pub mod detokenize;
pub mod health;
pub mod metrics;
pub mod models;
pub mod native_completion;
pub mod props;
pub mod slots;
pub mod tokenize;

pub use chat::chat_completions;
pub use completions::completions;
pub use detokenize::detokenize;
pub use health::health_check;
pub use metrics::metrics;
pub use models::list_models;
pub use native_completion::native_completion;
pub use props::props;
pub use slots::slots;
pub use tokenize::tokenize;
