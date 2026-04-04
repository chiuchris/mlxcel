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

//! Core output parsing logic for tool call extraction.
//!
//! Tries each known format in sequence. If tools were requested and the
//! output matches a known pattern, structured tool calls are extracted.
//!
//! Used by: routes/chat, tool_calls::mod

use super::formats;
use super::types::{ParsedToolCall, ToolCallParseResult};
use crate::server::types::request::Tool;

/// Strip `<think>...</think>` blocks from model output before parsing.
///
/// Many reasoning models (DeepSeek R1, Granite 3.3, etc.) wrap their
/// thought process in `<think>` tags.  We need to remove these to get at
/// the actual tool call output.
fn strip_thinking(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;

    while let Some(start) = remaining.find("<think>") {
        result.push_str(&remaining[..start]);
        if let Some(end) = remaining[start..].find("</think>") {
            remaining = &remaining[start + end + "</think>".len()..];
        } else {
            // Unclosed <think> tag -- strip everything after it
            remaining = "";
        }
    }
    result.push_str(remaining);
    result
}

/// Parse model output for tool calls, trying each known format in order.
///
/// `tools` is the set of tools that were passed in the request.  When
/// provided, the parser will filter out any calls to functions not in
/// the tool set.
///
/// Returns a `ToolCallParseResult` which may contain zero or more parsed
/// calls.
pub fn parse_tool_calls(raw_output: &str, tools: Option<&[Tool]>) -> ToolCallParseResult {
    // Strip thinking blocks first
    let cleaned = strip_thinking(raw_output);
    let text = cleaned.trim();

    if text.is_empty() {
        return ToolCallParseResult::none(raw_output.to_string());
    }

    // Try each format in order of specificity (most distinctive markers first)
    let parsers: &[fn(&str) -> Option<ToolCallParseResult>] = &[
        formats::try_hermes,
        formats::try_mistral_nemo,
        formats::try_functionary_v31,
        formats::try_functionary_v32,
        formats::try_llama3,
        formats::try_generic_json,
    ];

    for parser in parsers {
        if let Some(mut result) = parser(text) {
            // Filter tool calls to only those in the provided tool set
            if let Some(tools) = tools {
                result.tool_calls = filter_by_tools(result.tool_calls, tools);
            }
            if result.has_tool_calls() {
                return result;
            }
        }
    }

    ToolCallParseResult::none(raw_output.to_string())
}

/// Filter parsed tool calls to only include functions that exist in the
/// provided tool definitions.
fn filter_by_tools(calls: Vec<ParsedToolCall>, tools: &[Tool]) -> Vec<ParsedToolCall> {
    let tool_names: std::collections::HashSet<&str> =
        tools.iter().map(|t| t.function.name.as_str()).collect();

    calls
        .into_iter()
        .filter(|c| tool_names.contains(c.name.as_str()))
        .collect()
}

/// Generate a unique tool call ID in the format `call_` + random alphanumeric.
pub fn generate_tool_call_id() -> String {
    let id = uuid::Uuid::new_v4().to_string().replace('-', "");
    format!("call_{}", &id[..24])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::types::request::{FunctionDefinition, Tool};

    fn make_tool(name: &str) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: None,
                parameters: None,
            },
        }
    }

    #[test]
    fn parse_hermes_format() {
        let output =
            r#"<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>"#;
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn parse_with_thinking_blocks() {
        let output = r#"<think>Let me check the weather API.</think>
<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>"#;
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn parse_filters_unknown_tools() {
        let output = r#"<tool_call>{"name": "unknown_fn", "arguments": {}}</tool_call>"#;
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(!result.has_tool_calls());
    }

    #[test]
    fn parse_no_tools_provided_accepts_all() {
        let output = r#"<tool_call>{"name": "any_fn", "arguments": {"x": 1}}</tool_call>"#;
        let result = parse_tool_calls(output, None);
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "any_fn");
    }

    #[test]
    fn parse_plain_text_returns_none() {
        let output = "Hello, how can I help you?";
        let result = parse_tool_calls(output, None);
        assert!(!result.has_tool_calls());
        assert_eq!(result.content, output);
    }

    #[test]
    fn parse_mistral_nemo_format() {
        let output = r#"[TOOL_CALLS] [{"name": "search", "arguments": {"query": "rust"}}]"#;
        let tools = vec![make_tool("search")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn parse_functionary_v31_format() {
        let output = r#"<function=get_weather>{"location": "Berlin"}</function>"#;
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn parse_generic_json_format() {
        let output = r#"{"name": "calc", "arguments": {"expr": "2+2"}}"#;
        let tools = vec![make_tool("calc")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "calc");
    }

    #[test]
    fn generate_tool_call_id_format() {
        let id = generate_tool_call_id();
        assert!(id.starts_with("call_"));
        assert_eq!(id.len(), 29); // "call_" (5) + 24 hex chars
    }

    #[test]
    fn strip_thinking_removes_think_blocks() {
        let input = "<think>reasoning here</think>actual content";
        let result = strip_thinking(input);
        assert_eq!(result, "actual content");
    }

    #[test]
    fn strip_thinking_handles_multiple_blocks() {
        let input = "<think>first</think>middle<think>second</think>end";
        let result = strip_thinking(input);
        assert_eq!(result, "middleend");
    }

    #[test]
    fn strip_thinking_handles_unclosed_tag() {
        let input = "<think>unclosed thinking";
        let result = strip_thinking(input);
        assert_eq!(result, "");
    }

    #[test]
    fn strip_thinking_passes_through_no_tags() {
        let input = "no thinking tags here";
        let result = strip_thinking(input);
        assert_eq!(result, input);
    }
}
