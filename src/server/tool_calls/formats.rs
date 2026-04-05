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
///
/// Only matches `>>>` at the start of a line (position 0 or after `\n`).
/// This prevents false positives when `>>>` appears mid-line (e.g. in shell
/// output or blockquotes).
pub fn try_functionary_v32(text: &str) -> Option<ToolCallParseResult> {
    let trimmed = text.trim();
    // Guard: >>> must appear at line start (position 0 or after a newline)
    if !trimmed.starts_with(">>>") && !trimmed.contains("\n>>>") {
        return None;
    }

    let mut calls = Vec::new();
    let mut content = String::new();
    let mut found_first = false;

    // Walk line by line, treating each ">>>" line-start as a segment header.
    // Lines before the first ">>>" header become the `content` prefix.
    let mut current_name: Option<String> = None;
    let mut current_args_lines: Vec<&str> = Vec::new();

    for line in trimmed.lines() {
        if let Some(stripped) = line.strip_prefix(">>>") {
            // Flush any pending call
            if let Some(name) = current_name.take() {
                let json_str = current_args_lines.join("\n");
                let json_str = json_str.trim();
                if serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
                    calls.push(ParsedToolCall {
                        name,
                        arguments: json_str.to_string(),
                    });
                }
                current_args_lines.clear();
            }

            found_first = true;
            let name = stripped.trim().to_string();
            // Skip "all" which is a common delimiter for general text
            if name != "all" && !name.is_empty() {
                current_name = Some(name);
            }
        } else if found_first {
            if current_name.is_some() {
                current_args_lines.push(line);
            }
        } else {
            // Content before the first >>> marker
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(line);
        }
    }

    // Flush last pending call
    if let Some(name) = current_name.take() {
        let json_str = current_args_lines.join("\n");
        let json_str = json_str.trim().to_string();
        if serde_json::from_str::<serde_json::Value>(&json_str).is_ok() {
            calls.push(ParsedToolCall {
                name,
                arguments: json_str,
            });
        }
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::FunctionaryV32),
        tool_calls: calls,
        content: content.trim().to_string(),
    })
}

/// Try parsing Command R7B format:
/// `Action: function_name\nAction Input: {"key": "val"}`
///
/// Multiple calls may appear sequentially, each starting with `Action:`.
pub fn try_command_r(text: &str) -> Option<ToolCallParseResult> {
    // Guard: text must contain both the "Action:" and "Action Input:" markers.
    // Checking only "Action:" would cause false positives on normal prose
    // such as "Required Action: none".
    if !text.contains("Action:") || !text.contains("Action Input:") {
        return None;
    }

    let mut calls = Vec::new();
    let mut content = String::new();
    let mut found_first = false;
    let mut pending_name: Option<String> = None;

    for line in text.lines() {
        let trimmed_line = line.trim();
        if let Some(name) = trimmed_line.strip_prefix("Action:") {
            // Flush any pending action that had no Action Input line
            pending_name = Some(name.trim().to_string());
            if !found_first {
                found_first = true;
            }
        } else if let Some(json_part) = trimmed_line.strip_prefix("Action Input:") {
            if let Some(name) = pending_name.take() {
                let json_str = json_part.trim();
                if serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
                    calls.push(ParsedToolCall {
                        name,
                        arguments: json_str.to_string(),
                    });
                }
            }
        } else if !found_first {
            // Accumulate content before the first Action: marker
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(line);
        }
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::CommandR),
        tool_calls: calls,
        content: content.trim().to_string(),
    })
}

/// Try parsing Granite 3.3 format:
/// `<response><tool_call>{"name": ..., "arguments": ...}</tool_call></response>`
///
/// Strips the outer `<response>...</response>` wrapper and delegates to the
/// Hermes parser, which handles the inner `<tool_call>` tags.
pub fn try_granite(text: &str) -> Option<ToolCallParseResult> {
    let trimmed = text.trim();
    if !trimmed.contains("<response>") {
        return None;
    }

    // Extract the content inside <response>...</response>
    let start_tag = "<response>";
    let end_tag = "</response>";

    let inner_start = trimmed.find(start_tag)? + start_tag.len();
    let inner_end = trimmed[inner_start..].find(end_tag);

    let inner = if let Some(end_offset) = inner_end {
        &trimmed[inner_start..inner_start + end_offset]
    } else {
        // No closing tag: take everything after the opening tag
        &trimmed[inner_start..]
    };

    // Extract any content before the <response> tag
    let prefix = trimmed[..trimmed.find(start_tag).unwrap_or(0)].trim();

    // Delegate to the Hermes parser for the inner content
    let mut result = try_hermes(inner.trim())?;

    // Prepend any prefix content
    if !prefix.is_empty() {
        if result.content.is_empty() {
            result.content = prefix.to_string();
        } else {
            result.content = format!("{prefix}\n{}", result.content);
        }
    }

    // Override the format to Granite
    result.format = Some(ToolCallFormat::Granite);
    Some(result)
}

/// Try parsing Gemma 4 format:
/// `<|tool_call>call:function_name{key:<|"|>val<|"|>}<tool_call|>`
///
/// The pipe-delimited tags (`<|tool_call>` / `<tool_call|>`) distinguish this
/// format from Hermes's `<tool_call>` tag.  Multiple tool calls may appear
/// sequentially.  String values are delimited by `<|"|>` markers; non-string
/// values (numbers, booleans, null) are written without delimiters.
pub fn try_gemma4(text: &str) -> Option<ToolCallParseResult> {
    let tag_open = "<|tool_call>";
    let tag_close = "<tool_call|>";

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
        let block_start = start + tag_open.len();
        let block_end = remaining[block_start..].find(tag_close);

        let block = if let Some(end_offset) = block_end {
            let s = &remaining[block_start..block_start + end_offset];
            remaining = &remaining[block_start + end_offset + tag_close.len()..];
            s
        } else {
            // No closing tag: take the rest
            let s = &remaining[block_start..];
            remaining = "";
            s
        };

        if let Some(call) = parse_gemma4_block(block.trim()) {
            calls.push(call);
        }
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::Gemma4),
        tool_calls: calls,
        content,
    })
}

/// Parse a single Gemma 4 tool call block: `call:function_name{args_body}`
fn parse_gemma4_block(block: &str) -> Option<ParsedToolCall> {
    // Must start with "call:"
    let rest = block.strip_prefix("call:")?;

    // Find the opening brace for arguments
    let brace_pos = rest.find('{')?;
    let name = rest[..brace_pos].trim().to_string();
    if name.is_empty() {
        return None;
    }

    let args_with_brace = rest[brace_pos..].trim();
    let arguments = gemma4_args_to_json(args_with_brace)?;

    // Validate that the constructed JSON is well-formed before returning.
    // The Gemma4 format is built by string concatenation, so a malformed
    // model output could produce invalid JSON.  Other format parsers rely
    // on serde_json to parse the original text; we do the equivalent here.
    serde_json::from_str::<serde_json::Value>(&arguments).ok()?;

    Some(ParsedToolCall { name, arguments })
}

/// Convert Gemma 4 argument syntax to a valid JSON object string.
///
/// Input format: `{key:<|"|>string_val<|"|>, key2:42, key3:true}`
/// Output format: `{"key":"string_val","key2":42,"key3":true}`
fn gemma4_args_to_json(args: &str) -> Option<String> {
    // Strip outer braces
    let inner = args.strip_prefix('{')?.strip_suffix('}')?.trim();

    if inner.is_empty() {
        return Some("{}".to_string());
    }

    // Replace the string delimiter with a NUL placeholder that won't appear
    // in normal text, making it easy to detect string boundaries.
    const STR_DELIM: &str = "<|\"|\u{200b}>";
    let normalised = inner.replace("<|\"|>", STR_DELIM);

    let pairs = split_top_level_commas(&normalised);
    let mut json_pairs: Vec<String> = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }

        // Split on the first `:` to get key and value
        let colon_pos = pair.find(':')?;
        let key = pair[..colon_pos].trim();
        let value_raw = pair[colon_pos + 1..].trim();

        let json_key = format!("\"{}\"", escape_json_string(key));

        let json_value = if value_raw.starts_with(STR_DELIM) {
            // String value: strip delimiters and JSON-escape the content
            let without_open = value_raw.strip_prefix(STR_DELIM)?;
            let content = if let Some(end) = without_open.rfind(STR_DELIM) {
                &without_open[..end]
            } else {
                // Unterminated string: take everything
                without_open
            };
            format!("\"{}\"", escape_json_string(content))
        } else if value_raw.starts_with('{') {
            // Nested object: recurse
            // Restore the original delimiters before recursing
            let restored = value_raw.replace(STR_DELIM, "<|\"|>");
            gemma4_args_to_json(&restored)?
        } else if value_raw.starts_with('[') {
            // Array: restore delimiters and parse via serde_json
            let restored = value_raw.replace(STR_DELIM, "<|\"|>");
            gemma4_array_to_json(&restored)?
        } else {
            // Number, boolean, null — pass through as-is
            value_raw.to_string()
        };

        json_pairs.push(format!("{json_key}:{json_value}"));
    }

    Some(format!("{{{}}}", json_pairs.join(",")))
}

/// Escape a string for inclusion inside JSON double quotes.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Convert a Gemma 4 array literal to a JSON array string.
///
/// Arrays use the same `<|"|>` string escaping as objects.
fn gemma4_array_to_json(arr: &str) -> Option<String> {
    let inner = arr.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some("[]".to_string());
    }

    const STR_DELIM: &str = "<|\"|\u{200b}>";
    let normalised = inner.replace("<|\"|>", STR_DELIM);
    let items = split_top_level_commas(&normalised);
    let mut json_items: Vec<String> = Vec::with_capacity(items.len());

    for item in items {
        let item = item.trim();
        if item.starts_with(STR_DELIM) {
            let without_open = item.strip_prefix(STR_DELIM)?;
            let content = if let Some(end) = without_open.rfind(STR_DELIM) {
                &without_open[..end]
            } else {
                without_open
            };
            json_items.push(format!("\"{}\"", escape_json_string(content)));
        } else if item.starts_with('{') {
            let restored = item.replace(STR_DELIM, "<|\"|>");
            json_items.push(gemma4_args_to_json(&restored)?);
        } else if item.starts_with('[') {
            let restored = item.replace(STR_DELIM, "<|\"|>");
            json_items.push(gemma4_array_to_json(&restored)?);
        } else {
            json_items.push(item.to_string());
        }
    }

    Some(format!("[{}]", json_items.join(",")))
}

/// Split a comma-separated sequence at the top level only
/// (i.e., not inside `{}`, `[]` brackets, or string delimiters).
///
/// String regions are bounded by the `STR_DELIM` placeholder that
/// `gemma4_args_to_json` / `gemma4_array_to_json` substitute for the
/// original `<|"|>` markers.  Commas inside string regions are ignored.
fn split_top_level_commas(s: &str) -> Vec<&str> {
    const STR_DELIM: &str = "<|\"|\u{200b}>";
    let delim_bytes = STR_DELIM.as_bytes();

    let mut parts = Vec::new();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut last = 0;

    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Check for string delimiter boundary
        if i + delim_bytes.len() <= bytes.len() && &bytes[i..i + delim_bytes.len()] == delim_bytes {
            in_string = !in_string;
            i += delim_bytes.len();
            continue;
        }

        if !in_string {
            match bytes[i] {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth -= 1,
                b',' if depth == 0 => {
                    parts.push(&s[last..i]);
                    last = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    if last < s.len() {
        parts.push(&s[last..]);
    }
    parts
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

    #[test]
    fn functionary_v32_multiple_calls() {
        let text = ">>>fn_a\n{\"x\": 1}\n>>>fn_b\n{\"y\": 2}";
        let result = try_functionary_v32(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "fn_a");
        assert_eq!(result.tool_calls[1].name, "fn_b");
    }

    #[test]
    fn functionary_v32_no_false_positive_mid_line() {
        // >>> appearing mid-line (e.g. shell output) must NOT match
        let text = "The output was: foo>>>bar\nsome more text";
        assert!(try_functionary_v32(text).is_none());
    }

    #[test]
    fn functionary_v32_no_false_positive_blockquote() {
        // >>> in a blockquote-like context mid-sentence must not match
        let text = "Here is an example: >>>function_name\n{}";
        assert!(try_functionary_v32(text).is_none());
    }

    #[test]
    fn functionary_v32_content_before_first_marker() {
        let text = "Some preamble\n>>>get_weather\n{\"location\": \"Berlin\"}";
        let result = try_functionary_v32(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.content, "Some preamble");
    }

    // -- Command R --

    #[test]
    fn command_r_single_call() {
        let text = "Action: get_weather\nAction Input: {\"location\": \"Paris\"}";
        let result = try_command_r(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("Paris"));
        assert_eq!(result.format, Some(ToolCallFormat::CommandR));
    }

    #[test]
    fn command_r_multiple_calls() {
        let text = "Action: fn_a\nAction Input: {\"x\": 1}\nAction: fn_b\nAction Input: {\"y\": 2}";
        let result = try_command_r(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "fn_a");
        assert_eq!(result.tool_calls[1].name, "fn_b");
    }

    #[test]
    fn command_r_with_content_before() {
        let text = "I need to check the weather.\nAction: get_weather\nAction Input: {\"city\": \"Tokyo\"}";
        let result = try_command_r(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.contains("check the weather"));
    }

    #[test]
    fn command_r_no_match() {
        assert!(try_command_r("Hello, world!").is_none());
        assert!(try_command_r("Action without colon").is_none());
    }

    #[test]
    fn command_r_invalid_json_skipped() {
        let text = "Action: fn\nAction Input: not-json";
        assert!(try_command_r(text).is_none());
    }

    // -- Granite 3.3 --

    #[test]
    fn granite_single_call() {
        let text = r#"<response><tool_call>{"name": "get_weather", "arguments": {"city": "Seoul"}}</tool_call></response>"#;
        let result = try_granite(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("Seoul"));
        assert_eq!(result.format, Some(ToolCallFormat::Granite));
    }

    #[test]
    fn granite_multiple_calls() {
        let text = r#"<response><tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call></response>"#;
        let result = try_granite(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "a");
        assert_eq!(result.tool_calls[1].name, "b");
    }

    #[test]
    fn granite_with_content_before_response_tag() {
        let text = r#"Let me help.<response><tool_call>{"name": "search", "arguments": {"q": "rust"}}</tool_call></response>"#;
        let result = try_granite(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.contains("Let me help"));
    }

    #[test]
    fn granite_no_match() {
        assert!(try_granite("Hello, world!").is_none());
        assert!(
            try_granite("<tool_call>{\"name\": \"fn\", \"arguments\": {}}</tool_call>").is_none()
        );
    }

    // -- DeepSeek V3/R1 (Hermes format with think blocks) --

    #[test]
    fn deepseek_tool_call_via_hermes_with_think_block() {
        // DeepSeek wraps reasoning in <think> blocks; tool calls use <tool_call> tags.
        // Hermes parser handles the tool calls after think-block stripping in parser.rs.
        let text = r#"<think>Let me call the weather API.</think>
<tool_call>{"name": "get_weather", "arguments": {"location": "Beijing"}}</tool_call>"#;

        // Strip think blocks as the parser does
        let cleaned: String = {
            let mut result = String::new();
            let mut remaining = text;
            while let Some(start) = remaining.find("<think>") {
                result.push_str(&remaining[..start]);
                if let Some(end) = remaining[start..].find("</think>") {
                    remaining = &remaining[start + end + "</think>".len()..];
                } else {
                    remaining = "";
                }
            }
            result.push_str(remaining);
            result
        };

        let result = try_hermes(cleaned.trim()).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("Beijing"));
    }

    // -- Gemma 4 --

    #[test]
    fn test_gemma4_no_args() {
        let text = "<|tool_call>call:get_time{}<tool_call|>";
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_time");
        assert_eq!(result.tool_calls[0].arguments, "{}");
        assert_eq!(result.format, Some(ToolCallFormat::Gemma4));
    }

    #[test]
    fn test_gemma4_string_arg() {
        let text = "<|tool_call>call:get_weather{location:<|\"|>Tokyo<|\"|>}<tool_call|>";
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["location"], "Tokyo");
    }

    #[test]
    fn test_gemma4_mixed_types() {
        // string + number + boolean
        let text =
            "<|tool_call>call:search{query:<|\"|>rust<|\"|>,limit:10,active:true}<tool_call|>";
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["query"], "rust");
        assert_eq!(args["limit"], 10);
        assert_eq!(args["active"], true);
    }

    #[test]
    fn test_gemma4_multiple_calls() {
        let text = "<|tool_call>call:get_time{}<tool_call|><|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>";
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "get_time");
        assert_eq!(result.tool_calls[1].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[1].arguments).unwrap();
        assert_eq!(args["location"], "Paris");
    }

    #[test]
    fn test_gemma4_with_content_before() {
        let text = "Let me check the weather.\n<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>";
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.contains("check the weather"));
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_gemma4_empty_string_value() {
        let text = "<|tool_call>call:fn{key:<|\"|><|\"|>}<tool_call|>";
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["key"], "");
    }

    #[test]
    fn test_gemma4_string_with_special_chars() {
        // Strings containing double quotes and backslashes
        let text = r#"<|tool_call>call:fn{msg:<|"|>hello "world" and \path\<|"|>}<tool_call|>"#;
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["msg"], r#"hello "world" and \path\"#);
    }

    #[test]
    fn test_gemma4_string_with_comma() {
        // Commas inside string values must not split the argument list
        let text = r#"<|tool_call>call:fn{msg:<|"|>hello, world<|"|>,count:3}<tool_call|>"#;
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["msg"], "hello, world");
        assert_eq!(args["count"], 3);
    }

    #[test]
    fn test_gemma4_no_match() {
        // Regular Hermes tag should not match
        assert!(try_gemma4(r#"<tool_call>{"name": "fn", "arguments": {}}</tool_call>"#).is_none());
        // Plain text should not match
        assert!(try_gemma4("Hello, world!").is_none());
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
