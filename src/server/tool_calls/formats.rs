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

//! Known tool call format definitions for popular model families.
//!
//! Each format handler attempts to parse the model output and returns
//! `Some(ToolCallParseResult)` on success or `None` if the format
//! does not match.
//!
//! Used by: tool_calls::parser

use super::types::{ParsedToolCall, ToolCallFormat, ToolCallParseResult};

/// Try parsing Hermes/Qwen format:
/// `<tool_call>{"name": "fn", "arguments": {...}}</tool_call>`
///
/// Multiple tool calls may appear sequentially.
pub fn try_hermes(text: &str) -> Option<ToolCallParseResult> {
    let tag_open = "<tool_call>";
    let tag_close = "</tool_call>";

    if !text.contains(tag_open) {
        return None;
    }

    let mut calls = Vec::new();
    let mut content = String::new();
    let mut remaining = text;

    // Collect content before the first tool_call tag
    if let Some(first_pos) = remaining.find(tag_open) {
        let before = remaining[..first_pos].trim();
        if !before.is_empty() {
            content = before.to_string();
        }
        remaining = &remaining[first_pos..];
    }

    while let Some(start) = remaining.find(tag_open) {
        let json_start = start + tag_open.len();
        let end = remaining[json_start..].find(tag_close);

        let json_str = if let Some(end_offset) = end {
            let s = &remaining[json_start..json_start + end_offset];
            remaining = &remaining[json_start + end_offset + tag_close.len()..];
            s
        } else {
            // No closing tag: take the rest
            let s = &remaining[json_start..];
            remaining = "";
            s
        };

        if let Some(call) = parse_hermes_json(json_str.trim()) {
            calls.push(call);
        }
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::Hermes),
        tool_calls: calls,
        content,
    })
}

/// Parse a single Hermes-format JSON object.
fn parse_hermes_json(json_str: &str) -> Option<ParsedToolCall> {
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = v.get("arguments")?;
    Some(ParsedToolCall {
        name,
        arguments: if arguments.is_string() {
            arguments.as_str().unwrap_or_default().to_string()
        } else {
            serde_json::to_string(arguments).ok()?
        },
    })
}

/// Try parsing Llama 3.x format:
/// `{"name": "fn", "parameters": {...}}`
/// Optionally preceded by `<|python_tag|>`
pub fn try_llama3(text: &str) -> Option<ToolCallParseResult> {
    let cleaned = text
        .trim()
        .strip_prefix("<|python_tag|>")
        .unwrap_or(text.trim())
        .trim();

    // May be a single call or array of calls
    if cleaned.starts_with('[') {
        let arr: Vec<serde_json::Value> = serde_json::from_str(cleaned).ok()?;
        let calls: Vec<ParsedToolCall> = arr.iter().filter_map(parse_llama3_value).collect();
        if calls.is_empty() {
            return None;
        }
        return Some(ToolCallParseResult {
            format: Some(ToolCallFormat::Llama3),
            tool_calls: calls,
            content: String::new(),
        });
    }

    if cleaned.starts_with('{') {
        let v: serde_json::Value = serde_json::from_str(cleaned).ok()?;
        let call = parse_llama3_value(&v)?;
        return Some(ToolCallParseResult {
            format: Some(ToolCallFormat::Llama3),
            tool_calls: vec![call],
            content: String::new(),
        });
    }

    None
}

fn parse_llama3_value(v: &serde_json::Value) -> Option<ParsedToolCall> {
    let name = v.get("name")?.as_str()?.to_string();
    // Llama uses "parameters" instead of "arguments"
    let args = v.get("parameters").or_else(|| v.get("arguments"))?;
    Some(ParsedToolCall {
        name,
        arguments: if args.is_string() {
            args.as_str().unwrap_or_default().to_string()
        } else {
            serde_json::to_string(args).ok()?
        },
    })
}

/// Try parsing Mistral Nemo format:
/// `[TOOL_CALLS] [{"name": "fn", "arguments": {...}}]`
pub fn try_mistral_nemo(text: &str) -> Option<ToolCallParseResult> {
    let trimmed = text.trim();
    let json_part = trimmed.strip_prefix("[TOOL_CALLS]")?.trim();

    let arr: Vec<serde_json::Value> = serde_json::from_str(json_part).ok()?;
    let calls: Vec<ParsedToolCall> = arr
        .iter()
        .filter_map(|v| {
            let name = v.get("name")?.as_str()?.to_string();
            let arguments = v.get("arguments")?;
            Some(ParsedToolCall {
                name,
                arguments: if arguments.is_string() {
                    arguments.as_str().unwrap_or_default().to_string()
                } else {
                    serde_json::to_string(arguments).ok()?
                },
            })
        })
        .collect();

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::MistralNemo),
        tool_calls: calls,
        content: String::new(),
    })
}

/// Try parsing Functionary v3.1 format:
/// `<function=fn_name>{"key": "val"}</function>`
pub fn try_functionary_v31(text: &str) -> Option<ToolCallParseResult> {
    let prefix = "<function=";
    if !text.contains(prefix) {
        return None;
    }

    let mut calls = Vec::new();
    let mut content = String::new();
    let mut remaining = text;

    if let Some(first_pos) = remaining.find(prefix) {
        let before = remaining[..first_pos].trim();
        if !before.is_empty() {
            content = before.to_string();
        }
        remaining = &remaining[first_pos..];
    }

    while let Some(start) = remaining.find(prefix) {
        let name_start = start + prefix.len();
        // Use if-let instead of ? to avoid early return from the entire
        // function when a single malformed tag is missing its closing '>'.
        let Some(name_end) = remaining[name_start..].find('>') else {
            // Malformed tag without '>': skip past the prefix and continue
            remaining = &remaining[name_start..];
            continue;
        };
        let name = remaining[name_start..name_start + name_end].to_string();

        let json_start = name_start + name_end + 1; // skip '>'
        let close_tag = "</function>";
        let json_end = remaining[json_start..].find(close_tag);

        let json_str = if let Some(end_offset) = json_end {
            let s = &remaining[json_start..json_start + end_offset];
            remaining = &remaining[json_start + end_offset + close_tag.len()..];
            s
        } else {
            let s = &remaining[json_start..];
            remaining = "";
            s
        };

        let arguments = json_str.trim().to_string();
        // Validate it's valid JSON
        if serde_json::from_str::<serde_json::Value>(&arguments).is_ok() {
            calls.push(ParsedToolCall { name, arguments });
        }
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::FunctionaryV31),
        tool_calls: calls,
        content,
    })
}

/// Try parsing Functionary v3.2 format:
/// `>>>fn_name\n{"key": "val"}`
pub fn try_functionary_v32(text: &str) -> Option<ToolCallParseResult> {
    let prefix = ">>>";
    if !text.contains(prefix) {
        return None;
    }

    let mut calls = Vec::new();
    let mut content = String::new();

    // Collect content before the first >>> marker
    if let Some(first_pos) = text.find(prefix) {
        let before = text[..first_pos].trim();
        if !before.is_empty() {
            content = before.to_string();
        }
    }

    for segment in text.split(prefix).skip(1) {
        let segment = segment.trim();
        if let Some(newline_pos) = segment.find('\n') {
            let name = segment[..newline_pos].trim().to_string();
            let json_str = segment[newline_pos + 1..].trim();

            // Skip "all" which is a common delimiter for general text
            if name == "all" || name.is_empty() {
                continue;
            }

            if serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
                calls.push(ParsedToolCall {
                    name,
                    arguments: json_str.to_string(),
                });
            }
        }
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::FunctionaryV32),
        tool_calls: calls,
        content,
    })
}

/// Try parsing generic JSON format:
/// `{"name": "fn", "arguments": {...}}` or `{"name": "fn", "parameters": {...}}`
///
/// Also handles arrays: `[{"name": ..., "arguments": ...}, ...]`
pub fn try_generic_json(text: &str) -> Option<ToolCallParseResult> {
    let trimmed = text.trim();

    // Try as array
    if trimmed.starts_with('[')
        && let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(trimmed)
    {
        let calls: Vec<ParsedToolCall> = arr.iter().filter_map(parse_generic_value).collect();
        if !calls.is_empty() {
            return Some(ToolCallParseResult {
                format: Some(ToolCallFormat::GenericJson),
                tool_calls: calls,
                content: String::new(),
            });
        }
    }

    // Try as single object
    if trimmed.starts_with('{')
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed)
        && let Some(call) = parse_generic_value(&v)
    {
        return Some(ToolCallParseResult {
            format: Some(ToolCallFormat::GenericJson),
            tool_calls: vec![call],
            content: String::new(),
        });
    }

    None
}

fn parse_generic_value(v: &serde_json::Value) -> Option<ParsedToolCall> {
    let name = v.get("name")?.as_str()?.to_string();
    let args = v.get("arguments").or_else(|| v.get("parameters"))?;
    Some(ParsedToolCall {
        name,
        arguments: if args.is_string() {
            args.as_str().unwrap_or_default().to_string()
        } else {
            serde_json::to_string(args).ok()?
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Hermes / Qwen --

    #[test]
    fn hermes_single_tool_call() {
        let text =
            r#"<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>"#;
        let result = try_hermes(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("Paris"));
        assert_eq!(result.format, Some(ToolCallFormat::Hermes));
    }

    #[test]
    fn hermes_multiple_tool_calls() {
        let text = r#"<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>"#;
        let result = try_hermes(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "a");
        assert_eq!(result.tool_calls[1].name, "b");
    }

    #[test]
    fn hermes_with_content_before() {
        let text = r#"Let me check the weather.
<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>"#;
        let result = try_hermes(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.contains("check the weather"));
    }

    #[test]
    fn hermes_no_match() {
        assert!(try_hermes("Hello, world!").is_none());
    }

    // -- Llama 3.x --

    #[test]
    fn llama3_single_call() {
        let text = r#"{"name": "search", "parameters": {"query": "rust"}}"#;
        let result = try_llama3(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
        assert!(result.tool_calls[0].arguments.contains("rust"));
    }

    #[test]
    fn llama3_with_python_tag() {
        let text = r#"<|python_tag|>{"name": "calc", "parameters": {"expr": "2+2"}}"#;
        let result = try_llama3(text).unwrap();
        assert_eq!(result.tool_calls[0].name, "calc");
    }

    #[test]
    fn llama3_array() {
        let text = r#"[{"name": "a", "parameters": {}}, {"name": "b", "parameters": {"x": 1}}]"#;
        let result = try_llama3(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
    }

    // -- Mistral Nemo --

    #[test]
    fn mistral_nemo_single_call() {
        let text = r#"[TOOL_CALLS] [{"name": "get_time", "arguments": {"tz": "UTC"}}]"#;
        let result = try_mistral_nemo(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_time");
    }

    #[test]
    fn mistral_nemo_no_prefix() {
        assert!(try_mistral_nemo(r#"[{"name": "x", "arguments": {}}]"#).is_none());
    }

    // -- Functionary v3.1 --

    #[test]
    fn functionary_v31_single_call() {
        let text = r#"<function=get_weather>{"location": "Berlin"}</function>"#;
        let result = try_functionary_v31(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("Berlin"));
    }

    #[test]
    fn functionary_v31_no_match() {
        assert!(try_functionary_v31("Hello!").is_none());
    }

    #[test]
    fn functionary_v31_malformed_trailing_tag_preserves_prior_calls() {
        // A trailing malformed tag (no '>' at all) must not discard already-parsed calls.
        // Before the fix, the '?' operator would abort the entire function, returning None.
        let text =
            r#"<function=get_weather>{"location": "Berlin"}</function><function=broken_no_close"#;
        let result = try_functionary_v31(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("Berlin"));
    }

    // -- Functionary v3.2 --

    #[test]
    fn functionary_v32_single_call() {
        let text = ">>>get_weather\n{\"location\": \"London\"}";
        let result = try_functionary_v32(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn functionary_v32_skips_all_segment() {
        let text = ">>>all\nHello\n>>>get_weather\n{\"location\": \"Tokyo\"}";
        let result = try_functionary_v32(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    // -- Generic JSON --

    #[test]
    fn generic_json_single() {
        let text = r#"{"name": "fn1", "arguments": {"a": 1}}"#;
        let result = try_generic_json(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "fn1");
    }

    #[test]
    fn generic_json_array() {
        let text = r#"[{"name": "a", "arguments": {}}, {"name": "b", "parameters": {"x": 1}}]"#;
        let result = try_generic_json(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
    }

    #[test]
    fn generic_json_not_a_tool_call() {
        assert!(try_generic_json(r#"{"key": "value"}"#).is_none());
    }
}
