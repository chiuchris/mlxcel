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
use minijinja::{Environment, ErrorKind, Value};
use serde::{Deserialize, Serialize};

use super::types::request::Tool;

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
    /// Cached result of `supports_tools()` introspection.
    /// `None` means not yet computed.
    supports_tools_cached: Option<bool>,
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
            supports_tools_cached: None,
        }))
    }

    /// Create a processor with a custom template string
    pub fn with_template(template: String) -> Self {
        Self {
            template,
            bos_token: String::new(),
            eos_token: String::new(),
            add_generation_prompt: true,
            supports_tools_cached: None,
        }
    }

    /// Set whether to add a generation prompt at the end
    pub fn set_add_generation_prompt(&mut self, add: bool) {
        self.add_generation_prompt = add;
    }

    /// Check if the template uses the `tools` variable.
    ///
    /// Returns true when the template produces different output when `tools`
    /// is set vs. `None`.  The result is computed once by rendering the
    /// template with a sentinel tool and comparing output, then cached.
    ///
    /// Falls back to the string-heuristic if template rendering fails.
    pub fn supports_tools(&mut self) -> bool {
        if let Some(cached) = self.supports_tools_cached {
            return cached;
        }
        let result = self.compute_supports_tools();
        self.supports_tools_cached = Some(result);
        result
    }

    /// Inner computation for `supports_tools` — tries template rendering
    /// introspection and falls back to string heuristics on failure.
    fn compute_supports_tools(&self) -> bool {
        // Sentinel tool used for the probe render
        let sentinel_tool = Tool {
            tool_type: "function".to_string(),
            function: super::types::request::FunctionDefinition {
                name: "__test__".to_string(),
                description: Some("test".to_string()),
                parameters: None,
            },
        };

        let probe_messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];

        let with_tools = self.apply(&probe_messages, Some(&[sentinel_tool]));
        let without_tools = self.apply(&probe_messages, None);

        match (with_tools, without_tools) {
            (Ok(with), Ok(without)) => with != without,
            _ => {
                // Rendering failed — fall back to string heuristic
                self.template.contains("for tool in tools")
                    || self.template.contains("tools | tojson")
                    || self.template.contains("tools|tojson")
                    || (self.template.contains("tools") && self.template.contains("function"))
            }
        }
    }

    /// Check if the template handles multimodal content with image items.
    ///
    /// Returns true when the Jinja2 template iterates over content items and
    /// checks for `type == 'image'`, as Gemma3 VLM templates do.  Templates
    /// without this pattern expect `content` to be a plain string.
    pub fn supports_image_content(&self) -> bool {
        self.template.contains("'image'") || self.template.contains("\"image\"")
    }

    /// Apply the chat template with raw JSON messages (for multimodal content).
    ///
    /// This allows passing messages with list-type content entries (e.g.,
    /// `[{"type": "image"}, {"type": "text", "text": "..."}]`) that Jinja2
    /// templates like Gemma3 VLM can iterate over.
    ///
    /// When `tools` is `Some`, the tool definitions are passed to the Jinja2
    /// template context, enabling tool-calling prompt formatting.
    // Used by: chat_request, routes/chat
    pub fn apply_raw(
        &self,
        messages: &serde_json::Value,
        tools: Option<&[Tool]>,
    ) -> Result<String> {
        let mut env = Environment::new();
        configure_environment(&mut env);

        env.add_template("chat", &self.template)
            .with_context(|| "Failed to parse chat template")?;

        let tmpl = env.get_template("chat")?;

        // Convert serde_json::Value to minijinja::Value
        let messages_val = minijinja::Value::from_serialize(messages);

        let tools_val = tools.map(minijinja::Value::from_serialize);
        let context = minijinja::context! {
            messages => messages_val,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            add_generation_prompt => self.add_generation_prompt,
            tools => tools_val,
            enable_thinking => false,
        };

        let result = tmpl
            .render(context)
            .with_context(|| "Failed to render chat template")?;

        Ok(result)
    }

    /// Apply the chat template to messages.
    ///
    /// When `tools` is `Some`, the tool definitions are passed to the Jinja2
    /// template context, enabling tool-calling prompt formatting.
    // Used by: chat_request, routes/chat
    pub fn apply(&self, messages: &[ChatMessage], tools: Option<&[Tool]>) -> Result<String> {
        let mut env = Environment::new();
        configure_environment(&mut env);

        env.add_template("chat", &self.template)
            .with_context(|| "Failed to parse chat template")?;

        let tmpl = env.get_template("chat")?;

        // Many templates conditionally check variables like `tools`, `enable_thinking`, etc.
        // We must provide them as None/false to avoid undefined variable errors.
        let tools_val = tools.map(minijinja::Value::from_serialize);
        let context = minijinja::context! {
            messages => messages,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            add_generation_prompt => self.add_generation_prompt,
            tools => tools_val,
            enable_thinking => false,
        };

        let result = tmpl
            .render(context)
            .with_context(|| "Failed to render chat template")?;

        Ok(result)
    }
}

/// Configure a minijinja environment with common settings and Python-compat methods.
fn configure_environment(env: &mut Environment<'_>) {
    env.set_keep_trailing_newline(true);
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );

    env.add_function("strftime_now", |_format: String| -> String {
        chrono::Utc::now().format("%d %b %Y").to_string()
    });

    // Handle Python string methods not natively supported by minijinja.
    // Many HuggingFace chat templates (e.g. Gemma 4) use `.split()` which
    // is standard Python/Jinja2 but not a built-in minijinja string method.
    env.set_unknown_method_callback(|_state, value, method, args| match (value.kind(), method) {
        (minijinja::value::ValueKind::String, "split") => {
            let s = value.as_str().unwrap_or_default();
            let sep = args.first().and_then(|a| a.as_str()).unwrap_or_default();
            let parts: Vec<Value> = s.split(sep).map(|p| Value::from(p.to_string())).collect();
            Ok(Value::from(parts))
        }
        _ => Err(minijinja::Error::from(ErrorKind::UnknownMethod)),
    });
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

impl std::fmt::Debug for ChatTemplateProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplateProcessor")
            .field("bos_token", &self.bos_token)
            .field("eos_token", &self.eos_token)
            .field("add_generation_prompt", &self.add_generation_prompt)
            .field("supports_tools_cached", &self.supports_tools_cached)
            .finish_non_exhaustive()
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
        let result = processor.apply(&messages, None).unwrap();
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
        let result = processor.apply(&messages, None).unwrap();
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_apply_with_tools_none_still_works() {
        let processor = ChatTemplateProcessor::default();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        assert!(result.contains("User: Hi"));
    }

    #[test]
    fn test_apply_with_tools_passes_to_template() {
        // Template that explicitly uses tools
        let template = r#"{% if tools %}Tools: {{ tools | length }}{% endif %}
{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"#;

        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Call a tool".to_string(),
        }];

        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: crate::server::types::request::FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: None,
            },
        }];

        let result = processor.apply(&messages, Some(&tools)).unwrap();
        assert!(result.contains("Tools: 1"));
    }

    #[test]
    fn test_supports_tools_detection() {
        // Template that uses tools — rendering differs with/without tools
        let with_tools =
            r#"{% if tools %}[TOOLS]{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}"#;
        let mut processor = ChatTemplateProcessor::with_template(with_tools.to_string());
        assert!(processor.supports_tools());
        // Second call must return cached result
        assert!(processor.supports_tools());

        // Template without tools — rendering is identical with/without tools
        let without = r#"{% for m in messages %}{{ m.content }}{% endfor %}"#;
        let mut processor = ChatTemplateProcessor::with_template(without.to_string());
        assert!(!processor.supports_tools());
    }

    #[test]
    fn test_supports_tools_caching() {
        let template =
            r#"{% if tools %}[TOOLS]{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}"#;
        let mut processor = ChatTemplateProcessor::with_template(template.to_string());
        // Not cached yet
        assert!(processor.supports_tools_cached.is_none());
        // First call computes
        let _ = processor.supports_tools();
        // Now cached
        assert!(processor.supports_tools_cached.is_some());
    }

    #[test]
    fn test_apply_raw_with_tools() {
        let template = r#"{% if tools %}[TOOLS]{% endif %}{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}"#;

        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = serde_json::json!([
            {"role": "user", "content": "hi"}
        ]);

        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: crate::server::types::request::FunctionDefinition {
                name: "test_fn".to_string(),
                description: None,
                parameters: None,
            },
        }];

        let result = processor.apply_raw(&messages, Some(&tools)).unwrap();
        assert!(result.contains("[TOOLS]"));

        // Without tools
        let result_no_tools = processor.apply_raw(&messages, None).unwrap();
        assert!(!result_no_tools.contains("[TOOLS]"));
    }
}
