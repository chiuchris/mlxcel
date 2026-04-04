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

//! Internal types for tool call parsing and detection.
//!
//! Used by: tool_calls::parser, tool_calls::formats, routes/chat

/// A parsed tool call extracted from model output.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    /// Function name
    pub name: String,
    /// Arguments as a JSON string
    pub arguments: String,
}

/// The format detected in the model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolCallFormat {
    /// Hermes/Qwen: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
    Hermes,
    /// Llama 3.x: `{"name": ..., "parameters": ...}` possibly with `<|python_tag|>`
    Llama3,
    /// Mistral Nemo: `[TOOL_CALLS] [{"name": ..., "arguments": ...}]`
    MistralNemo,
    /// Functionary v3.1: `<function=name>{"key": "val"}</function>`
    FunctionaryV31,
    /// Functionary v3.2: `>>>name\n{"key": "val"}`
    FunctionaryV32,
    /// Generic JSON: raw `{"name": ..., "arguments": ...}` object
    GenericJson,
}

/// Result of parsing model output for tool calls.
#[derive(Debug, Clone)]
pub struct ToolCallParseResult {
    /// The detected format (if any)
    pub format: Option<ToolCallFormat>,
    /// Extracted tool calls
    pub tool_calls: Vec<ParsedToolCall>,
    /// Any text content before/outside the tool calls (may be empty)
    pub content: String,
}

impl ToolCallParseResult {
    /// Returns true if tool calls were found.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Create an empty result (no tool calls found).
    pub fn none(content: String) -> Self {
        Self {
            format: None,
            tool_calls: Vec::new(),
            content,
        }
    }
}
