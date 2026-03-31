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

//! Chat template processing for OpenAI-compatible messages
//!
//! Applies Jinja2 chat templates from tokenizer_config.json to format
//! conversation messages into model-specific prompts.

use std::path::Path;

use anyhow::{Context, Result};
use minijinja::{Environment, Value};
use serde::{Deserialize, Serialize};

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Chat template processor
#[derive(Clone)]
pub struct ChatTemplateProcessor {
    template: String,
    bos_token: String,
    eos_token: String,
    add_generation_prompt: bool,
}

impl ChatTemplateProcessor {
    /// Create a new processor by loading template from tokenizer_config.json or chat_template.jinja
    pub fn from_model_path(model_path: &Path) -> Result<Option<Self>> {
        let config_path = model_path.join("tokenizer_config.json");
        let config: Option<serde_json::Value> = if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read {:?}", config_path))?;
            Some(
                serde_json::from_str(&content)
                    .with_context(|| "Failed to parse tokenizer_config.json")?,
            )
        } else {
            None
        };

        // Try tokenizer_config.json "chat_template" field first, then chat_template.jinja file
        let template = config
            .as_ref()
            .and_then(|c| c.get("chat_template"))
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| {
                let jinja_path = model_path.join("chat_template.jinja");
                std::fs::read_to_string(jinja_path).ok()
            });

        let Some(template) = template else {
            return Ok(None);
        };

        // Extract special tokens
        let bos_token = config
            .as_ref()
            .and_then(|c| extract_token(c, "bos_token"))
            .unwrap_or_default();
        let eos_token = config
            .as_ref()
            .and_then(|c| extract_token(c, "eos_token"))
            .unwrap_or_default();

        Ok(Some(Self {
            template,
            bos_token,
            eos_token,
            add_generation_prompt: true,
        }))
    }

    /// Create a processor with a custom template string
    pub fn with_template(template: String) -> Self {
        Self {
            template,
            bos_token: String::new(),
            eos_token: String::new(),
            add_generation_prompt: true,
        }
    }

    /// Set whether to add a generation prompt at the end
    pub fn set_add_generation_prompt(&mut self, add: bool) {
        self.add_generation_prompt = add;
    }

    /// Apply the chat template with raw JSON messages (for multimodal content).
    ///
    /// This allows passing messages with list-type content entries (e.g.,
    /// `[{"type": "image"}, {"type": "text", "text": "..."}]`) that Jinja2
    /// templates like Gemma3 VLM can iterate over.
    pub fn apply_raw(&self, messages: &serde_json::Value) -> Result<String> {
        let mut env = Environment::new();
        env.set_keep_trailing_newline(true);
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);

        env.add_template("chat", &self.template)
            .with_context(|| "Failed to parse chat template")?;

        env.add_function(
            "raise_exception",
            |msg: String| -> Result<Value, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );

        env.add_function("strftime_now", |_format: String| -> String {
            chrono::Utc::now().format("%d %b %Y").to_string()
        });

        let tmpl = env.get_template("chat")?;

        // Convert serde_json::Value to minijinja::Value
        let messages_val = minijinja::Value::from_serialize(messages);

        let tools: Option<Vec<String>> = None;
        let context = minijinja::context! {
            messages => messages_val,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            add_generation_prompt => self.add_generation_prompt,
            tools => tools,
            enable_thinking => false,
        };

        let result = tmpl
            .render(context)
            .with_context(|| "Failed to render chat template")?;

        Ok(result)
    }

    /// Apply the chat template to messages
    pub fn apply(&self, messages: &[ChatMessage]) -> Result<String> {
        let mut env = Environment::new();
        env.set_keep_trailing_newline(true);
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);

        // Add the template
        env.add_template("chat", &self.template)
            .with_context(|| "Failed to parse chat template")?;

        // Add raise_exception function (used by some templates)
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<Value, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );

        // Add strftime_now function if needed
        env.add_function("strftime_now", |_format: String| -> String {
            chrono::Utc::now().format("%d %b %Y").to_string()
        });

        let tmpl = env.get_template("chat")?;

        // Build context
        // Many templates conditionally check variables like `tools`, `enable_thinking`, etc.
        // We must provide them as None/false to avoid undefined variable errors.
        let tools: Option<Vec<String>> = None;
        let context = minijinja::context! {
            messages => messages,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            add_generation_prompt => self.add_generation_prompt,
            tools => tools,
            enable_thinking => false,
        };

        let result = tmpl
            .render(context)
            .with_context(|| "Failed to render chat template")?;

        Ok(result)
    }
}

/// Extract a token from config, handling both string and object formats
fn extract_token(config: &serde_json::Value, key: &str) -> Option<String> {
    config.get(key).and_then(|v| {
        if v.is_string() {
            v.as_str().map(String::from)
        } else if v.is_object() {
            // Some configs use {"content": "<token>"}
            v.get("content").and_then(|c| c.as_str()).map(String::from)
        } else {
            None
        }
    })
}

/// Default fallback template for models without chat_template
pub fn default_chat_template() -> &'static str {
    r#"{% for message in messages %}{% if message.role == 'system' %}System: {{ message.content }}

{% elif message.role == 'user' %}User: {{ message.content }}

{% elif message.role == 'assistant' %}Assistant: {{ message.content }}

{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"#
}

impl Default for ChatTemplateProcessor {
    fn default() -> Self {
        Self::with_template(default_chat_template().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_template() {
        let processor = ChatTemplateProcessor::default();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let result = processor.apply(&messages).unwrap();
        assert!(result.contains("User: Hello"));
        assert!(result.ends_with("Assistant: "));
    }

    #[test]
    fn test_chatml_template() {
        let template = r#"{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;

        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let result = processor.apply(&messages).unwrap();
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant"));
    }
}
