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

    /// Feed a decoded text fragment.  Returns content to emit as a
    /// `delta.content` chunk, or `None` if the text was suppressed.
    pub fn feed(&mut self, fragment: &str) -> Option<String> {
        self.buffer.push_str(fragment);
        self.drain_buffer()
    }

    /// Flush any remaining buffered content at the end of generation.
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        let remaining = std::mem::take(&mut self.buffer);
        if self.state == FilterState::Content && !remaining.is_empty() {
            Some(remaining)
        } else {
            None
        }
    }

    /// Process the internal buffer: find delimiters, emit content, and
    /// transition states.
    fn drain_buffer(&mut self) -> Option<String> {
        let mut output = String::new();

        loop {
            if self.buffer.is_empty() {
                break;
            }

            match self.find_earliest_delimiter() {
                Some((pos, delim_len, action)) => {
                    // Text before the delimiter
                    if pos > 0 && self.state == FilterState::Content {
                        output.push_str(&self.buffer[..pos]);
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

                    if safe_len > 0 && self.state == FilterState::Content {
                        output.push_str(&self.buffer[..safe_len]);
                    }
                    if safe_len > 0 {
                        self.buffer.drain(..safe_len);
                    }
                    break;
                }
            }
        }

        if output.is_empty() {
            None
        } else {
            Some(output)
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
        assert_eq!(f.feed("Hello world"), Some("Hello world".to_string()));
    }

    #[test]
    fn empty_fragment() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed(""), None);
    }

    // -- Thinking blocks --

    #[test]
    fn thinking_only() {
        let mut f = StreamFilter::new();
        let result = f.feed("<|channel>thought\nI should search.<channel|><turn|>");
        assert_eq!(result, None);
    }

    #[test]
    fn thinking_then_content() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("<|channel>thought\nPlanning.<channel|>Here is the answer."),
            Some("Here is the answer.".to_string())
        );
    }

    #[test]
    fn thinking_then_content_with_turn() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("<|channel>thought\nPlanning.<channel|>Here is the answer.<turn|>"),
            Some("Here is the answer.".to_string())
        );
    }

    #[test]
    fn thinking_then_tool_call() {
        let mut f = StreamFilter::new();
        let input = "<|channel>thought\nI need to search.<channel|>\
                      <|tool_call>call:web_search{query:<|\"|>rust<|\"|>}<tool_call|><turn|>";
        assert_eq!(f.feed(input), None);
    }

    // -- Tool call blocks --

    #[test]
    fn content_then_tool_call() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("Let me search.<|tool_call>call:web_search{query:<|\"|>rust<|\"|>}<tool_call|>"),
            Some("Let me search.".to_string())
        );
    }

    #[test]
    fn tool_call_only() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|tool_call>call:fn{}<tool_call|>"), None);
    }

    // -- Turn markers --

    #[test]
    fn strips_trailing_turn() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("Answer"), Some("Answer".to_string()));
        assert_eq!(f.feed("<turn|>"), None);
    }

    #[test]
    fn strips_leading_turn() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|turn>Hello"), Some("Hello".to_string()));
    }

    #[test]
    fn strips_think_marker() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("Before<|think|>After"),
            Some("BeforeAfter".to_string())
        );
    }

    // -- Partial delimiter buffering --

    #[test]
    fn partial_delimiter_at_end() {
        let mut f = StreamFilter::new();
        // Fragment ends mid-delimiter "<|to" — could be "<|tool_call>"
        assert_eq!(f.feed("Hello <|to"), Some("Hello ".to_string()));
        // Next fragment completes the delimiter
        assert_eq!(f.feed("ol_call>rest<tool_call|>"), None);
    }

    #[test]
    fn partial_delimiter_resolves_to_content() {
        let mut f = StreamFilter::new();
        // "<|" at the end could start a delimiter
        assert_eq!(f.feed("test <|"), Some("test ".to_string()));
        // Next fragment shows it's not a delimiter
        assert_eq!(f.feed("not_a_delim"), Some("<|not_a_delim".to_string()));
    }

    #[test]
    fn partial_channel_delimiter() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("Hello <|cha"), Some("Hello ".to_string()));
        // Completes the channel delimiter
        assert_eq!(
            f.feed("nnel>thought\nthinking<channel|>World"),
            Some("World".to_string())
        );
    }

    #[test]
    fn angle_bracket_mid_text_passes_through() {
        let mut f = StreamFilter::new();
        // "< 3" does NOT match any delimiter prefix (none start with "< ")
        // so the entire text passes through immediately.
        assert_eq!(f.feed("2 < 3"), Some("2 < 3".to_string()));
    }

    #[test]
    fn trailing_angle_bracket_buffered() {
        let mut f = StreamFilter::new();
        // A lone '<' at the very end IS held back (could start any delimiter)
        assert_eq!(f.feed("end<"), Some("end".to_string()));
        // Next fragment resolves it — not a delimiter
        assert_eq!(f.feed(" more"), Some("< more".to_string()));
    }

    // -- Multi-fragment streaming --

    #[test]
    fn token_by_token_content() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("Hello"), Some("Hello".to_string()));
        assert_eq!(f.feed(" "), Some(" ".to_string()));
        assert_eq!(f.feed("world"), Some("world".to_string()));
    }

    #[test]
    fn token_by_token_thinking_then_content() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|channel>"), None);
        assert_eq!(f.feed("thought\n"), None);
        assert_eq!(f.feed("I am thinking."), None);
        assert_eq!(f.feed("<channel|>"), None);
        assert_eq!(
            f.feed("Here is my answer."),
            Some("Here is my answer.".to_string())
        );
    }

    #[test]
    fn token_by_token_tool_call() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|tool_call>"), None);
        assert_eq!(f.feed("call:search{"), None);
        assert_eq!(f.feed("query:<|\"|>test<|\"|>}"), None);
        assert_eq!(f.feed("<tool_call|>"), None);
        // Back in content state
        assert_eq!(f.flush(), None);
    }

    // -- Flush --

    #[test]
    fn flush_emits_buffered_content() {
        let mut f = StreamFilter::new();
        // '<' is held back as potential delimiter start
        assert_eq!(f.feed("end<"), Some("end".to_string()));
        // Flush at end of generation: emit the held-back '<'
        assert_eq!(f.flush(), Some("<".to_string()));
    }

    #[test]
    fn flush_in_thinking_suppresses() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|channel>unclosed thinking"), None);
        // Still in thinking state at flush — suppress
        assert_eq!(f.flush(), None);
    }

    // -- No false positives --

    #[test]
    fn angle_brackets_in_normal_text() {
        let mut f = StreamFilter::new();
        // HTML-like content should pass through
        assert_eq!(
            f.feed("<div>hello</div>"),
            Some("<div>hello</div>".to_string())
        );
    }

    #[test]
    fn pipe_in_normal_text() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("a | b | c"), Some("a | b | c".to_string()));
    }
}
