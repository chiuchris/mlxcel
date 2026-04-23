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

//! Streaming content filter for Gemma 4 special tokens.
//!
//! Prevents model-internal structural tokens (`<|channel>`, `<|tool_call>`,
//! `<turn|>`, etc.) from leaking into SSE content deltas during streaming
//! generation.
//!
//! The filter operates as a state machine on decoded text fragments (after
//! `StreamingDecodeState` has resolved UTF-8 boundaries).  It buffers text
//! at delimiter boundaries to handle the case where a delimiter straddles
//! two decode fragments.
//!
//! Used by: routes/chat (streaming path)

/// Split output from [`StreamFilter::feed`] / [`StreamFilter::flush`].
///
/// `content` is text that should be emitted to the client as
/// `delta.content`; `reasoning` is text that belongs inside a scratchpad /
/// thinking block (Gemma 4 `<|channel>thought\n...<channel|>`) and should be
/// emitted as `delta.reasoning_content` so downstream routers and UIs can
/// display a "thinking" status without parsing model-specific markers. Both
/// are `None` when the fragment was fully buffered or fell inside a
/// tool-call block (tool calls are materialized by the non-streaming parser
/// path, not here).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilterOutput {
    pub content: Option<String>,
    pub reasoning: Option<String>,
}

impl FilterOutput {
    /// Convenience constructor for a content-only emit.
    pub fn content(text: String) -> Self {
        Self {
            content: Some(text),
            reasoning: None,
        }
    }

    /// Convenience constructor for a reasoning-only emit.
    pub fn reasoning(text: String) -> Self {
        Self {
            content: None,
            reasoning: Some(text),
        }
    }

    /// `true` when the filter produced no output at all.
    pub fn is_empty(&self) -> bool {
        self.content.is_none() && self.reasoning.is_none()
    }
}

/// Action to take when a delimiter is matched.
#[derive(Debug, Clone, Copy)]
enum DelimiterAction {
    /// Enter thinking state — suppress content until exit.
    EnterThinking,
    /// Exit thinking state — resume content emission.
    ExitThinking,
    /// Enter tool call state — suppress content until exit.
    EnterToolCall,
    /// Exit tool call state — resume content emission.
    ExitToolCall,
    /// Strip the delimiter but don't change state.
    Strip,
}

/// Gemma 4 special tokens and their associated actions.
///
/// The delimiter list is ordered by length (longest first) so that longer
/// matches take priority over shorter prefixes during scanning.
const GEMMA4_DELIMITERS: &[(&str, DelimiterAction)] = &[
    ("<|tool_call>", DelimiterAction::EnterToolCall),
    ("<tool_call|>", DelimiterAction::ExitToolCall),
    ("<|channel>", DelimiterAction::EnterThinking),
    ("<channel|>", DelimiterAction::ExitThinking),
    ("<|think|>", DelimiterAction::Strip),
    ("<|turn>", DelimiterAction::Strip),
    ("<turn|>", DelimiterAction::Strip),
];

/// The state of the streaming filter state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterState {
    /// Emitting content to the client.
    Content,
    /// Inside a thinking/channel block — suppress all text.
    Thinking,
    /// Inside a tool call block — suppress all text.
    ToolCall,
}

/// Streaming content filter that strips Gemma 4 structural tokens from
/// content deltas.
///
/// Feed decoded text fragments via [`feed()`](StreamFilter::feed).  The
/// filter returns only the content that should be emitted to the client.
/// Call [`flush()`](StreamFilter::flush) at the end of generation to emit
/// any remaining buffered content.
pub struct StreamFilter {
    state: FilterState,
    buffer: String,
    /// Length of the longest delimiter (for partial-match buffering).
    max_delim_len: usize,
}

impl Default for StreamFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamFilter {
    /// Create a new stream filter with Gemma 4 delimiter set.
    pub fn new() -> Self {
        let max_delim_len = GEMMA4_DELIMITERS
            .iter()
            .map(|(s, _)| s.len())
            .max()
            .unwrap_or(0);

        Self {
            state: FilterState::Content,
            buffer: String::new(),
            max_delim_len,
        }
    }

    /// Create a filter that starts inside a `Thinking` block.
    ///
    /// Use when the chat template's generation prompt primed an open
    /// `<|channel>thought\n` (Gemma 4 `enable_thinking=true` branch): the
    /// first emitted token is already reasoning content, not a regular
    /// response, so suppress it until the model emits `<channel|>` to
    /// transition back to `Content`. If the model never closes the block,
    /// the entire generation is suppressed — matching the non-streaming
    /// post-processor behavior in `routes::chat::strip_unclosed_primed_thinking`.
    pub fn new_primed_open_thinking() -> Self {
        let mut s = Self::new();
        s.state = FilterState::Thinking;
        s
    }

    /// Feed a decoded text fragment. Returns split output: `content` holds
    /// text that should be emitted as `delta.content`, `reasoning` holds
    /// text that should be emitted as `delta.reasoning_content`. Either (or
    /// both) may be `None` when the fragment is entirely buffered / suppressed.
    pub fn feed(&mut self, fragment: &str) -> FilterOutput {
        self.buffer.push_str(fragment);
        self.drain_buffer()
    }

    /// Flush any remaining buffered content at the end of generation.
    ///
    /// Emits the tail under the channel that matches the filter's current
    /// state: anything still buffered inside `Thinking` surfaces as
    /// `reasoning`, anything inside `Content` surfaces as `content`, and
    /// anything inside `ToolCall` is discarded (tool-call deltas are emitted
    /// via the parser path, not here).
    pub fn flush(&mut self) -> FilterOutput {
        if self.buffer.is_empty() {
            return FilterOutput::default();
        }
        let remaining = std::mem::take(&mut self.buffer);
        match self.state {
            FilterState::Content if !remaining.is_empty() => FilterOutput::content(remaining),
            FilterState::Thinking if !remaining.is_empty() => FilterOutput::reasoning(remaining),
            _ => FilterOutput::default(),
        }
    }

    /// Process the internal buffer: find delimiters, emit content/reasoning,
    /// and transition states.
    fn drain_buffer(&mut self) -> FilterOutput {
        let mut content = String::new();
        let mut reasoning = String::new();

        loop {
            if self.buffer.is_empty() {
                break;
            }

            match self.find_earliest_delimiter() {
                Some((pos, delim_len, action)) => {
                    // Text before the delimiter is attributed to the current
                    // state so thinking fragments surface as reasoning and
                    // regular fragments surface as content.
                    if pos > 0 {
                        match self.state {
                            FilterState::Content => content.push_str(&self.buffer[..pos]),
                            FilterState::Thinking => reasoning.push_str(&self.buffer[..pos]),
                            FilterState::ToolCall => {}
                        }
                    }

                    // Consume the delimiter and transition (drain in-place
                    // to avoid allocating a new String per delimiter hit).
                    let after = pos + delim_len;
                    self.buffer.drain(..after);
                    self.apply_action(action);
                }
                None => {
                    // No complete delimiter — check for a partial match
                    // at the tail and hold it back.
                    let safe_len = self.safe_emit_length();

                    if safe_len > 0 {
                        match self.state {
                            FilterState::Content => content.push_str(&self.buffer[..safe_len]),
                            FilterState::Thinking => reasoning.push_str(&self.buffer[..safe_len]),
                            FilterState::ToolCall => {}
                        }
                        self.buffer.drain(..safe_len);
                    }
                    break;
                }
            }
        }

        FilterOutput {
            content: if content.is_empty() { None } else { Some(content) },
            reasoning: if reasoning.is_empty() {
                None
            } else {
                Some(reasoning)
            },
        }
    }

    /// Find the earliest complete delimiter in the buffer.
    ///
    /// Returns `(byte_position, delimiter_len, action)`.
    fn find_earliest_delimiter(&self) -> Option<(usize, usize, DelimiterAction)> {
        let mut earliest: Option<(usize, usize, DelimiterAction)> = None;

        for &(delim, action) in GEMMA4_DELIMITERS {
            if let Some(pos) = self.buffer.find(delim) {
                match earliest {
                    Some((best_pos, best_len, _)) => {
                        // Pick earliest position; on tie, pick longest delimiter
                        if pos < best_pos || (pos == best_pos && delim.len() > best_len) {
                            earliest = Some((pos, delim.len(), action));
                        }
                    }
                    None => {
                        earliest = Some((pos, delim.len(), action));
                    }
                }
            }
        }

        earliest
    }

    /// Find how many bytes at the start of the buffer are safe to emit,
    /// i.e. no suffix of the buffer could be the start of a delimiter.
    fn safe_emit_length(&self) -> usize {
        let buf = &self.buffer;
        let buf_len = buf.len();
        if buf_len == 0 {
            return 0;
        }

        // Only check positions near the end (within max_delim_len - 1 bytes)
        let check_from = buf_len.saturating_sub(self.max_delim_len.saturating_sub(1));

        for (i, _) in buf.char_indices() {
            if i < check_from {
                continue;
            }
            let suffix = &buf[i..];
            for &(delim, _) in GEMMA4_DELIMITERS {
                // A suffix is a partial match if it's a proper prefix of a
                // delimiter (shorter, and the delimiter starts with it).
                if suffix.len() < delim.len() && delim.starts_with(suffix) {
                    return i;
                }
            }
        }

        buf_len
    }

    /// Apply a delimiter action to transition the state machine.
    fn apply_action(&mut self, action: DelimiterAction) {
        match action {
            DelimiterAction::EnterThinking => self.state = FilterState::Thinking,
            DelimiterAction::ExitThinking => self.state = FilterState::Content,
            DelimiterAction::EnterToolCall => self.state = FilterState::ToolCall,
            DelimiterAction::ExitToolCall => self.state = FilterState::Content,
            DelimiterAction::Strip => { /* consume delimiter, no state change */ }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Basic passthrough --

    #[test]
    fn no_special_tokens_passthrough() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("Hello world").content, Some("Hello world".to_string()));
    }

    #[test]
    fn empty_fragment() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("").content, None);
    }

    // -- Thinking blocks --

    #[test]
    fn thinking_only() {
        let mut f = StreamFilter::new();
        let result = f.feed("<|channel>thought\nI should search.<channel|><turn|>");
        assert_eq!(result.content, None);
    }

    #[test]
    fn thinking_then_content() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("<|channel>thought\nPlanning.<channel|>Here is the answer.").content, Some("Here is the answer.".to_string())
        );
    }

    #[test]
    fn thinking_then_content_with_turn() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("<|channel>thought\nPlanning.<channel|>Here is the answer.<turn|>").content, Some("Here is the answer.".to_string())
        );
    }

    #[test]
    fn thinking_then_tool_call() {
        let mut f = StreamFilter::new();
        let input = "<|channel>thought\nI need to search.<channel|>\
                      <|tool_call>call:web_search{query:<|\"|>rust<|\"|>}<tool_call|><turn|>";
        assert_eq!(f.feed(input).content, None);
    }

    // -- Tool call blocks --

    #[test]
    fn content_then_tool_call() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("Let me search.<|tool_call>call:web_search{query:<|\"|>rust<|\"|>}<tool_call|>").content, Some("Let me search.".to_string())
        );
    }

    #[test]
    fn tool_call_only() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|tool_call>call:fn{}<tool_call|>").content, None);
    }

    // -- Turn markers --

    #[test]
    fn strips_trailing_turn() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("Answer").content, Some("Answer".to_string()));
        assert_eq!(f.feed("<turn|>").content, None);
    }

    #[test]
    fn strips_leading_turn() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|turn>Hello").content, Some("Hello".to_string()));
    }

    #[test]
    fn strips_think_marker() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("Before<|think|>After").content, Some("BeforeAfter".to_string())
        );
    }

    // -- Partial delimiter buffering --

    #[test]
    fn partial_delimiter_at_end() {
        let mut f = StreamFilter::new();
        // Fragment ends mid-delimiter "<|to" — could be "<|tool_call>"
        assert_eq!(f.feed("Hello <|to").content, Some("Hello ".to_string()));
        // Next fragment completes the delimiter
        assert_eq!(f.feed("ol_call>rest<tool_call|>").content, None);
    }

    #[test]
    fn partial_delimiter_resolves_to_content() {
        let mut f = StreamFilter::new();
        // "<|" at the end could start a delimiter
        assert_eq!(f.feed("test <|").content, Some("test ".to_string()));
        // Next fragment shows it's not a delimiter
        assert_eq!(f.feed("not_a_delim").content, Some("<|not_a_delim".to_string()));
    }

    #[test]
    fn partial_channel_delimiter() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("Hello <|cha").content, Some("Hello ".to_string()));
        // Completes the channel delimiter
        assert_eq!(
            f.feed("nnel>thought\nthinking<channel|>World").content, Some("World".to_string())
        );
    }

    #[test]
    fn angle_bracket_mid_text_passes_through() {
        let mut f = StreamFilter::new();
        // "< 3" does NOT match any delimiter prefix (none start with "< ")
        // so the entire text passes through immediately.
        assert_eq!(f.feed("2 < 3").content, Some("2 < 3".to_string()));
    }

    #[test]
    fn trailing_angle_bracket_buffered() {
        let mut f = StreamFilter::new();
        // A lone '<' at the very end IS held back (could start any delimiter)
        assert_eq!(f.feed("end<").content, Some("end".to_string()));
        // Next fragment resolves it — not a delimiter
        assert_eq!(f.feed(" more").content, Some("< more".to_string()));
    }

    // -- Multi-fragment streaming --

    #[test]
    fn token_by_token_content() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("Hello").content, Some("Hello".to_string()));
        assert_eq!(f.feed(" ").content, Some(" ".to_string()));
        assert_eq!(f.feed("world").content, Some("world".to_string()));
    }

    #[test]
    fn token_by_token_thinking_then_content() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|channel>").content, None);
        assert_eq!(f.feed("thought\n").content, None);
        assert_eq!(f.feed("I am thinking.").content, None);
        assert_eq!(f.feed("<channel|>").content, None);
        assert_eq!(
            f.feed("Here is my answer.").content, Some("Here is my answer.".to_string())
        );
    }

    #[test]
    fn token_by_token_tool_call() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|tool_call>").content, None);
        assert_eq!(f.feed("call:search{").content, None);
        assert_eq!(f.feed("query:<|\"|>test<|\"|>}").content, None);
        assert_eq!(f.feed("<tool_call|>").content, None);
        // Back in content state
        assert_eq!(f.flush().content, None);
    }

    // -- Flush --

    #[test]
    fn flush_emits_buffered_content() {
        let mut f = StreamFilter::new();
        // '<' is held back as potential delimiter start
        assert_eq!(f.feed("end<").content, Some("end".to_string()));
        // Flush at end of generation: emit the held-back '<'
        assert_eq!(f.flush().content, Some("<".to_string()));
    }

    #[test]
    fn flush_in_thinking_suppresses() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|channel>unclosed thinking").content, None);
        // Still in thinking state at flush — suppress
        assert_eq!(f.flush().content, None);
    }

    // -- No false positives --

    #[test]
    fn angle_brackets_in_normal_text() {
        let mut f = StreamFilter::new();
        // HTML-like content should pass through
        assert_eq!(
            f.feed("<div>hello</div>").content, Some("<div>hello</div>".to_string())
        );
    }

    #[test]
    fn pipe_in_normal_text() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("a | b | c").content, Some("a | b | c".to_string()));
    }

    // -- Reasoning split (FilterOutput.content vs .reasoning) --

    #[test]
    fn thinking_block_surfaces_as_reasoning_not_content() {
        // Inside a `<|channel>thought…<channel|>` block, fragments belong to
        // `reasoning` so the server can emit them as `delta.reasoning_content`
        // SSE chunks. Content after the close stays in `content`.
        let mut f = StreamFilter::new();
        let out = f.feed("<|channel>thought\nReasoning text<channel|>Answer text");
        assert_eq!(out.reasoning.as_deref(), Some("thought\nReasoning text"));
        assert_eq!(out.content.as_deref(), Some("Answer text"));
    }

    #[test]
    fn primed_open_thinking_starts_in_reasoning_state() {
        // When the chat template primed an open `<|channel>thought\n` at the
        // end of the generation prompt, the model's first emitted tokens are
        // already inside the thinking channel. Starting the filter in
        // `Thinking` state routes them to `reasoning`.
        let mut f = StreamFilter::new_primed_open_thinking();
        let out = f.feed("I am thinking<channel|>the answer");
        assert_eq!(out.reasoning.as_deref(), Some("I am thinking"));
        assert_eq!(out.content.as_deref(), Some("the answer"));
    }

    #[test]
    fn primed_open_thinking_unclosed_suppresses_content_entirely() {
        // Model ran out of budget inside the primed channel — never emits
        // `<channel|>`. The filter keeps every token in `reasoning` and
        // leaves `content` empty, including on flush.
        let mut f = StreamFilter::new_primed_open_thinking();
        let out = f.feed("reasoning continues");
        assert_eq!(out.content, None);
        assert_eq!(out.reasoning.as_deref(), Some("reasoning continues"));

        let flushed = f.flush();
        assert_eq!(flushed.content, None);
        // Buffer was already drained inside feed, so flush has nothing left
        // to surface. Either None or empty is acceptable.
        assert!(
            flushed.reasoning.as_deref().unwrap_or("").is_empty(),
            "flush must not leak buffered reasoning as content"
        );
    }

    #[test]
    fn primed_open_thinking_token_by_token() {
        // Fragment-by-fragment streaming: every pre-close token is reasoning,
        // every post-close token is content.
        let mut f = StreamFilter::new_primed_open_thinking();
        assert_eq!(f.feed("Let me").reasoning.as_deref(), Some("Let me"));
        assert_eq!(f.feed(" think.").reasoning.as_deref(), Some(" think."));
        // Close marker transitions state; nothing emitted on the marker itself.
        let closing = f.feed("<channel|>");
        assert!(closing.reasoning.is_none());
        assert!(closing.content.is_none());
        // After the close, fragments are content.
        assert_eq!(f.feed("Done.").content.as_deref(), Some("Done."));
    }

    #[test]
    fn flush_in_content_state_emits_content_only() {
        // Pre-existing behavior: when the filter is in the Content state at
        // flush, any buffered tail surfaces as content (not reasoning).
        let mut f = StreamFilter::new();
        // '<' is held back as a potential delimiter start.
        assert_eq!(f.feed("end<").content.as_deref(), Some("end"));
        let flushed = f.flush();
        assert_eq!(flushed.content.as_deref(), Some("<"));
        assert!(flushed.reasoning.is_none());
    }
}
