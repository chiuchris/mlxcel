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

use super::{
    apply_user_chat_template, generated_suffix, generation_stats_from_duration, resolve_cli_prompt,
};
use mlxcel::server::chat_template::ChatTemplateProcessor;
use std::time::Duration;

#[test]
fn generation_stats_from_duration_uses_elapsed_time_for_decode_rate() {
    let stats = generation_stats_from_duration(12, 6, Duration::from_secs(2));

    assert_eq!(stats.prompt_tokens, 12);
    assert_eq!(stats.generated_tokens, 6);
    assert_eq!(stats.decode_time_ms, 2000.0);
    assert_eq!(stats.decode_tok_per_sec, 3.0);
}

#[test]
fn generation_stats_from_duration_handles_zero_elapsed_time() {
    let stats = generation_stats_from_duration(4, 2, Duration::ZERO);

    assert_eq!(stats.decode_time_ms, 0.0);
    assert_eq!(stats.decode_tok_per_sec, 0.0);
}

#[test]
fn apply_user_chat_template_wraps_prompt_as_user_message() {
    let processor = ChatTemplateProcessor::with_template(
        "{{ messages[0].role }}: {{ messages[0].content }}".to_string(),
    );

    let rendered = apply_user_chat_template(&processor, "Hello");

    assert_eq!(rendered, "user: Hello");
}

#[test]
fn resolve_cli_prompt_skips_template_when_disabled() {
    let processor = ChatTemplateProcessor::with_template("wrapped".to_string());

    let prompt = resolve_cli_prompt("Hello", true, Some(&processor));

    assert_eq!(prompt, "Hello");
}

#[test]
fn resolve_cli_prompt_falls_back_on_template_errors() {
    let processor = ChatTemplateProcessor::with_template("{% if %}".to_string());

    let prompt = resolve_cli_prompt("Hello", false, Some(&processor));

    assert_eq!(prompt, "Hello");
}

#[test]
fn generated_suffix_strips_prompt_prefix() {
    assert_eq!(generated_suffix("Hello, world", "Hello"), ", world");
}

#[test]
fn generated_suffix_falls_back_when_prefix_is_missing() {
    assert_eq!(generated_suffix("world", "Hello"), "world");
}
