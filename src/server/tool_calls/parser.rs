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

/// Strip thinking/reasoning blocks from model output before parsing.
///
/// Handles two formats:
/// - `<think>...</think>` — DeepSeek R1, Granite 3.3, etc.
/// - `<|channel>...<channel|>` — Gemma 4 thinking channels
///
/// Used by: tool_calls::parser
fn strip_thinking(text: &str) -> String {
    let pairs: &[(&str, &str)] = &[("<think>", "</think>"), ("<|channel>", "<channel|>")];

    let mut result = text.to_string();
    for &(open, close) in pairs {
        let mut new_result = String::with_capacity(result.len());
        let mut remaining = result.as_str();
        while let Some(start) = remaining.find(open) {
            new_result.push_str(&remaining[..start]);
            if let Some(end) = remaining[start..].find(close) {
                remaining = &remaining[start + end + close.len()..];
            } else {
                // Unclosed tag -- strip everything after it
                remaining = "";
            }
        }
        new_result.push_str(remaining);
        result = new_result;
    }
    result
}

/// Remove model-specific structural markers from content text.
///
/// Called when tool-call parsing was attempted but no tool calls were found,
/// or when cleaning content extracted alongside tool calls.  Strips markers
/// that would otherwise leak into the response content.
///
/// Used by: tool_calls::parser
fn clean_content_markers(text: &str) -> String {
    text.replace("<turn|>", "")
        .replace("<|turn>", "")
        .replace("<|think|>", "")
        .trim()
        .to_string()
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
        formats::try_granite, // <response><tool_call> — more specific than bare Hermes
        formats::try_gemma4,  // <|tool_call>call:... — pipe-delimited, before Hermes
        formats::try_hermes,  // <tool_call> — Hermes/Qwen/DeepSeek
        formats::try_mistral_nemo, // [TOOL_CALLS]
        formats::try_functionary_v31, // <function=name>
        formats::try_functionary_v32, // >>>name\n
        formats::try_llama3,  // {"name": ..., "parameters": ...}
        formats::try_generic_json, // {"name": ..., "arguments": ...}
        formats::try_command_r, // Action: / Action Input: — least specific
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

    // No tool calls found: return cleaned content (thinking blocks and
    // model-specific markers stripped) so callers never see raw control tokens.
    let content = clean_content_markers(text);
    ToolCallParseResult::none(content)
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
    fn parse_command_r_format() {
        let output = "Action: search\nAction Input: {\"query\": \"rust\"}";
        let tools = vec![make_tool("search")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn parse_granite_format() {
        let output = r#"<response><tool_call>{"name": "get_info", "arguments": {"id": 42}}</tool_call></response>"#;
        let tools = vec![make_tool("get_info")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "get_info");
    }

    #[test]
    fn parse_deepseek_format_via_hermes() {
        // DeepSeek uses <tool_call> (Hermes style) after <think> blocks
        let output = "<think>Reasoning step.</think>\n<tool_call>{\"name\": \"fn\", \"arguments\": {\"k\": \"v\"}}</tool_call>";
        let tools = vec![make_tool("fn")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "fn");
    }

    #[test]
    fn parse_gemma4_format() {
        let output = "<|tool_call>call:get_weather{location:<|\"|>Tokyo<|\"|>}<tool_call|>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["location"], "Tokyo");
        assert_eq!(
            result.format,
            Some(crate::server::tool_calls::ToolCallFormat::Gemma4)
        );
    }

    #[test]
    fn parse_gemma4_filters_unknown_tools() {
        let output = "<|tool_call>call:unknown_fn{key:<|\"|>val<|\"|>}<tool_call|>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(!result.has_tool_calls());
    }

    #[test]
    fn parse_granite_takes_precedence_over_hermes() {
        // Granite format wraps in <response>; must be matched by Granite handler, not Hermes
        let output =
            r#"<response><tool_call>{"name": "fn", "arguments": {}}</tool_call></response>"#;
        let tools = vec![make_tool("fn")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(
            result.format,
            Some(crate::server::tool_calls::ToolCallFormat::Granite)
        );
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

    // -- Gemma 4 channel/thinking block stripping --

    #[test]
    fn strip_thinking_removes_gemma4_channel() {
        let input = "<|channel>thought\nI should search.<channel|>actual content";
        let result = strip_thinking(input);
        assert_eq!(result, "actual content");
    }

    #[test]
    fn strip_thinking_handles_unclosed_channel() {
        let input = "<|channel>thought\nunclosed channel";
        let result = strip_thinking(input);
        assert_eq!(result, "");
    }

    #[test]
    fn strip_thinking_handles_both_think_and_channel() {
        let input = "<think>think block</think><|channel>thought\nchannel block<channel|>final";
        let result = strip_thinking(input);
        assert_eq!(result, "final");
    }

    // -- Gemma 4 full-path parse tests (issue #311) --

    #[test]
    fn gemma4_thinking_only() {
        let output = "<|channel>thought\nI should search.<channel|><turn|>";
        let result = parse_tool_calls(output, None);
        assert!(!result.has_tool_calls());
        assert_eq!(result.content, "");
    }

    #[test]
    fn gemma4_thinking_then_content() {
        let output = "<|channel>thought\nPlanning.<channel|>Here is the answer.<turn|>";
        let result = parse_tool_calls(output, None);
        assert!(!result.has_tool_calls());
        assert_eq!(result.content, "Here is the answer.");
    }

    #[test]
    fn gemma4_thinking_then_tool_call() {
        let output = "<|channel>thought\nI need to search.<channel|>\
                       <|tool_call>call:web_search{query:<|\"|>rust<|\"|>}<tool_call|><turn|>";
        let tools = vec![make_tool("web_search")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "web_search");
        assert_eq!(result.content, "");
    }

    #[test]
    fn gemma4_content_then_tool_call() {
        let output = "Let me search.\
                       <|tool_call>call:web_search{query:<|\"|>rust<|\"|>}<tool_call|><turn|>";
        let tools = vec![make_tool("web_search")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.content, "Let me search.");
    }

    #[test]
    fn gemma4_multiple_tool_calls() {
        let output = "<|tool_call>call:search{query:<|\"|>rust<|\"|>}<tool_call|>\
                       <|tool_call>call:calc{expr:<|\"|>2+2<|\"|>}<tool_call|><turn|>";
        let tools = vec![make_tool("search"), make_tool("calc")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.tool_calls[1].name, "calc");
    }

    #[test]
    fn gemma4_strips_trailing_turn() {
        // Content followed by <turn|> with no tool calls
        let output = "Here is the answer.<turn|>";
        let result = parse_tool_calls(output, None);
        assert!(!result.content.contains("<turn|>"));
        assert_eq!(result.content, "Here is the answer.");
    }

    #[test]
    fn gemma4_strips_trailing_turn_with_tools() {
        // Content + tool call + <turn|> — content must not contain <turn|>
        let output = "Content<|tool_call>call:fn{key:<|\"|>v<|\"|>}<tool_call|><turn|>";
        let tools = vec![make_tool("fn")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert!(!result.content.contains("<turn|>"));
        assert_eq!(result.content, "Content");
    }

    #[test]
    fn clean_content_markers_strips_all() {
        let input = "<|turn>content<turn|><|think|>";
        let result = clean_content_markers(input);
        assert_eq!(result, "content");
    }
}
