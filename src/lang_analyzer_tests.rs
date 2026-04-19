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

//! Integration tests for `mlxcel_core::lang_analyzer`.
//!
//! These tests verify the public API of the lang_analyzer module from the
//! perspective of a downstream consumer (the main mlxcel crate). Unit-level
//! tests live in `mlxcel-core/src/lang_analyzer.rs`; this file validates the
//! same invariants via the public API surface.

use mlxcel_core::lang_analyzer::{Script, classify_token, is_numeric, is_punctuation, is_whitespace};

// ============================================================================
// classify_token — §10.1 acceptance criteria strings
// ============================================================================

#[test]
fn lang_analyze_pure_hangul() {
    let result = classify_token("한국어");
    assert_eq!(result.as_slice(), &[Script::Hangul]);
}

#[test]
fn lang_analyze_pure_han() {
    let result = classify_token("中文");
    assert_eq!(result.as_slice(), &[Script::Han]);
}

#[test]
fn lang_analyze_hangul_and_han() {
    let result = classify_token("韓국어");
    assert!(result.contains(&Script::Hangul));
    assert!(result.contains(&Script::Han));
    assert_eq!(result.len(), 2);
}

#[test]
fn lang_analyze_pure_hiragana() {
    let result = classify_token("ひらがな");
    assert_eq!(result.as_slice(), &[Script::Hiragana]);
}

#[test]
fn lang_analyze_pure_katakana() {
    let result = classify_token("カタカナ");
    assert_eq!(result.as_slice(), &[Script::Katakana]);
}

#[test]
fn lang_analyze_hiragana_and_han() {
    let result = classify_token("日本語のひらがな");
    assert!(result.contains(&Script::Hiragana));
    assert!(result.contains(&Script::Han));
    assert_eq!(result.len(), 2);
}

#[test]
fn lang_analyze_pure_latin() {
    let result = classify_token("hello");
    assert_eq!(result.as_slice(), &[Script::Latin]);
}

#[test]
fn lang_analyze_pure_cyrillic() {
    let result = classify_token("Привет");
    assert_eq!(result.as_slice(), &[Script::Cyrillic]);
}

#[test]
fn lang_analyze_numeric_returns_empty_scripts() {
    let result = classify_token("12345");
    assert!(result.is_empty(), "numeric string should yield no scripts; got {result:?}");
}

#[test]
fn lang_analyze_numeric_is_numeric_flag() {
    assert!(is_numeric("12345"));
    assert!(!is_numeric("hello"));
}

#[test]
fn lang_analyze_punctuation_returns_empty_scripts() {
    let result = classify_token(",.!?");
    assert!(result.is_empty(), "punctuation string should yield no scripts; got {result:?}");
}

#[test]
fn lang_analyze_punctuation_is_punctuation_flag() {
    assert!(is_punctuation(",.!?"));
    assert!(!is_punctuation("hello"));
}

#[test]
fn lang_analyze_whitespace_is_whitespace_flag() {
    assert!(is_whitespace("   "));
    assert!(!is_whitespace("hello"));
}

#[test]
fn lang_analyze_bpe_prefix_strips_prefix() {
    // ▁hello — ▁ is a SentencePiece BPE prefix, should be ignored.
    let result = classify_token("▁hello");
    assert_eq!(result.as_slice(), &[Script::Latin]);
}

#[test]
fn lang_analyze_bpe_prefix_alone_returns_empty() {
    let result = classify_token("▁");
    assert!(result.is_empty());
    assert!(is_whitespace("▁"), "BPE prefix should count as whitespace");
}
