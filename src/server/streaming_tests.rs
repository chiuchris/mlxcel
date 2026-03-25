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

use serde::{Serialize, Serializer};
use serde_json::Value;

use super::{DONE_MARKER, payload_channel, serialize_json_data};
use crate::server::types::{ChatCompletionChunk, CompletionChunk};

#[derive(Serialize)]
struct TestPayload<'a> {
    token: &'a str,
}

struct FailingPayload;

impl Serialize for FailingPayload {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Err(serde::ser::Error::custom("boom"))
    }
}

#[test]
fn blocking_sse_sender_sends_json_text_and_done_in_order() {
    let (sender, mut rx) = payload_channel(4);

    sender.json(&TestPayload { token: "hello" }).unwrap();
    sender.text("plain-text");
    sender.done();

    assert_eq!(
        rx.blocking_recv().unwrap().unwrap(),
        r#"{"token":"hello"}"#.to_string()
    );
    assert_eq!(rx.blocking_recv().unwrap().unwrap(), "plain-text");
    assert_eq!(rx.blocking_recv().unwrap().unwrap(), DONE_MARKER);
}

#[test]
fn done_marker_matches_openai_stream_terminator() {
    assert_eq!(DONE_MARKER, "[DONE]");
}

#[test]
fn serialize_json_data_returns_errors_instead_of_panicking() {
    let err = serialize_json_data(&FailingPayload)
        .unwrap_err()
        .to_string();
    assert!(err.contains("boom"));
}

// ── OpenAI streaming format compliance tests ──────────────────────────────────

/// `finish_reason` must always be serialized in content chunks as `null` (not
/// omitted). Strict clients such as opencode, Continue, and Cursor reject
/// chunks that lack the key entirely.
#[test]
fn content_chunk_serializes_finish_reason_as_null() {
    let chunk = ChatCompletionChunk::content(
        "chatcmpl-test".to_string(),
        "model".to_string(),
        "hello".to_string(),
    );
    let json = serde_json::to_value(&chunk).unwrap();
    let choice = &json["choices"][0];
    assert!(
        choice.get("finish_reason").is_some(),
        "finish_reason key must be present in content chunks"
    );
    assert_eq!(
        choice["finish_reason"],
        Value::Null,
        "finish_reason must be null in content chunks"
    );
}

/// The `system_fingerprint` field must always be present in every chunk (even
/// if null) to satisfy clients that rely on it for state tracking.
#[test]
fn content_chunk_has_system_fingerprint_field() {
    let chunk = ChatCompletionChunk::content(
        "chatcmpl-test".to_string(),
        "model".to_string(),
        "hello".to_string(),
    );
    let json = serde_json::to_value(&chunk).unwrap();
    assert!(
        json.get("system_fingerprint").is_some(),
        "system_fingerprint key must be present"
    );
    assert_eq!(json["system_fingerprint"], Value::Null);
}

/// The `logprobs` field must always be present in each choice (even if null).
#[test]
fn content_chunk_choice_has_logprobs_field() {
    let chunk = ChatCompletionChunk::content(
        "chatcmpl-test".to_string(),
        "model".to_string(),
        "hello".to_string(),
    );
    let json = serde_json::to_value(&chunk).unwrap();
    let choice = &json["choices"][0];
    assert!(
        choice.get("logprobs").is_some(),
        "logprobs key must be present in each choice"
    );
    assert_eq!(choice["logprobs"], Value::Null);
}

/// Usage chunks must carry an empty `choices` array and a populated `usage`
/// object, per the OpenAI streaming spec for `include_usage`.
#[test]
fn usage_chunk_has_empty_choices_and_populated_usage() {
    let chunk =
        ChatCompletionChunk::usage("chatcmpl-test".to_string(), "model".to_string(), 10, 20);
    let json = serde_json::to_value(&chunk).unwrap();

    let choices = json["choices"].as_array().unwrap();
    assert!(
        choices.is_empty(),
        "usage chunk must have empty choices array"
    );

    assert_eq!(json["usage"]["prompt_tokens"], 10);
    assert_eq!(json["usage"]["completion_tokens"], 20);
    assert_eq!(json["usage"]["total_tokens"], 30);
}

/// Initial chunk must have role="assistant" in delta and null finish_reason.
#[test]
fn initial_chunk_has_role_and_null_finish_reason() {
    let chunk = ChatCompletionChunk::initial("chatcmpl-test".to_string(), "model".to_string());
    let json = serde_json::to_value(&chunk).unwrap();
    let choice = &json["choices"][0];
    assert_eq!(choice["delta"]["role"], "assistant");
    assert_eq!(choice["finish_reason"], Value::Null);
}

/// Finish chunk must carry the finish reason string (not null).
#[test]
fn finish_chunk_has_stop_finish_reason() {
    let chunk = ChatCompletionChunk::finish(
        "chatcmpl-test".to_string(),
        "model".to_string(),
        "stop".to_string(),
    );
    let json = serde_json::to_value(&chunk).unwrap();
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
}

/// Same compliance checks for text-completion streaming chunks.
#[test]
fn completion_content_chunk_serializes_finish_reason_as_null() {
    let chunk = CompletionChunk::content(
        "cmpl-test".to_string(),
        "model".to_string(),
        "hello".to_string(),
    );
    let json = serde_json::to_value(&chunk).unwrap();
    let choice = &json["choices"][0];
    assert!(
        choice.get("finish_reason").is_some(),
        "finish_reason must be present in content chunks"
    );
    assert_eq!(choice["finish_reason"], Value::Null);
}

#[test]
fn completion_usage_chunk_has_empty_choices_and_populated_usage() {
    let chunk = CompletionChunk::usage("cmpl-test".to_string(), "model".to_string(), 5, 15);
    let json = serde_json::to_value(&chunk).unwrap();
    let choices = json["choices"].as_array().unwrap();
    assert!(choices.is_empty());
    assert_eq!(json["usage"]["prompt_tokens"], 5);
    assert_eq!(json["usage"]["completion_tokens"], 15);
    assert_eq!(json["usage"]["total_tokens"], 20);
}
