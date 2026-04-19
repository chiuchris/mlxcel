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

use mlxcel_core::lang_analyzer::{
    CURRENT_VERSION, Script, TokenLanguageIndex, classify_token, is_numeric, is_punctuation,
    is_whitespace,
};

// ============================================================================
// B2 — classify_token — §10.1 acceptance criteria strings
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

// ============================================================================
// B3 — TokenLanguageIndex helper
// ============================================================================

/// Minimal 100-token WordLevel tokenizer JSON for unit tests.
/// Contains 3 special tokens and 97 regular tokens, including some Latin
/// words and numeric strings for flag coverage.
const MOCK_TOKENIZER_100_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<s>",   "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "</s>",  "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "hello": 3, "world": 4, "test": 5, "the": 6, "and": 7,
      "123": 8, "456": 9, "789": 10,
      "token11": 11, "token12": 12, "token13": 13, "token14": 14,
      "token15": 15, "token16": 16, "token17": 17, "token18": 18,
      "token19": 19, "token20": 20, "token21": 21, "token22": 22,
      "token23": 23, "token24": 24, "token25": 25, "token26": 26,
      "token27": 27, "token28": 28, "token29": 29, "token30": 30,
      "token31": 31, "token32": 32, "token33": 33, "token34": 34,
      "token35": 35, "token36": 36, "token37": 37, "token38": 38,
      "token39": 39, "token40": 40, "token41": 41, "token42": 42,
      "token43": 43, "token44": 44, "token45": 45, "token46": 46,
      "token47": 47, "token48": 48, "token49": 49, "token50": 50,
      "token51": 51, "token52": 52, "token53": 53, "token54": 54,
      "token55": 55, "token56": 56, "token57": 57, "token58": 58,
      "token59": 59, "token60": 60, "token61": 61, "token62": 62,
      "token63": 63, "token64": 64, "token65": 65, "token66": 66,
      "token67": 67, "token68": 68, "token69": 69, "token70": 70,
      "token71": 71, "token72": 72, "token73": 73, "token74": 74,
      "token75": 75, "token76": 76, "token77": 77, "token78": 78,
      "token79": 79, "token80": 80, "token81": 81, "token82": 82,
      "token83": 83, "token84": 84, "token85": 85, "token86": 86,
      "token87": 87, "token88": 88, "token89": 89, "token90": 90,
      "token91": 91, "token92": 92, "token93": 93, "token94": 94,
      "token95": 95, "token96": 96, "token97": 97, "token98": 98,
      "token99": 99
    },
    "unk_token": "<unk>"
  }
}"#;

/// Slight variation of the mock tokenizer — different vocab hash.
const MOCK_TOKENIZER_ALT_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<pad>": 0, "alt_token": 1
    },
    "unk_token": "<pad>"
  }
}"#;

fn make_tokenizer(json: &str) -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("failed to parse mock tokenizer JSON")
}

// ============================================================================
// B3 — TokenLanguageIndex unit tests (§ acceptance criteria)
// ============================================================================

#[test]
fn build_populates_all_tokens() {
    let tok = make_tokenizer(MOCK_TOKENIZER_100_JSON);
    let idx = TokenLanguageIndex::build(&tok, MOCK_TOKENIZER_100_JSON.as_bytes())
        .expect("build should succeed");
    assert_eq!(idx.tokens.len(), 100, "expected 100 token entries");
}

#[test]
fn build_inverts_by_script() {
    let tok = make_tokenizer(MOCK_TOKENIZER_100_JSON);
    let idx = TokenLanguageIndex::build(&tok, MOCK_TOKENIZER_100_JSON.as_bytes())
        .expect("build should succeed");

    // For every script in by_script, verify each listed token id actually
    // has that script in its TokenScriptInfo.
    for (script, ids) in &idx.by_script {
        for &token_id in ids {
            let info = &idx.tokens[token_id as usize];
            assert!(
                info.scripts.contains(script),
                "token id {token_id} in by_script[{script:?}] but missing script in info.scripts"
            );
        }
    }
}

#[test]
fn build_flags_special_tokens() {
    let tok = make_tokenizer(MOCK_TOKENIZER_100_JSON);
    let idx = TokenLanguageIndex::build(&tok, MOCK_TOKENIZER_100_JSON.as_bytes())
        .expect("build should succeed");

    // Ids 0, 1, 2 are special tokens in the mock tokenizer.
    for special_id in [0u32, 1, 2] {
        assert!(
            idx.tokens[special_id as usize].is_special,
            "token id {special_id} should be flagged as special"
        );
    }
    // A regular token (id 3, "hello") should NOT be special.
    assert!(
        !idx.tokens[3].is_special,
        "token id 3 ('hello') should not be flagged as special"
    );
}

#[test]
fn build_vocab_hash_deterministic() {
    let tok = make_tokenizer(MOCK_TOKENIZER_100_JSON);
    let bytes = MOCK_TOKENIZER_100_JSON.as_bytes();
    let idx1 = TokenLanguageIndex::build(&tok, bytes).expect("first build failed");
    let idx2 = TokenLanguageIndex::build(&tok, bytes).expect("second build failed");
    assert_eq!(
        idx1.vocab_hash, idx2.vocab_hash,
        "vocab_hash must be deterministic for the same tokenizer"
    );
}

#[test]
fn build_vocab_hash_differs_across_tokenizers() {
    let tok1 = make_tokenizer(MOCK_TOKENIZER_100_JSON);
    let tok2 = make_tokenizer(MOCK_TOKENIZER_ALT_JSON);

    let idx1 = TokenLanguageIndex::build(&tok1, MOCK_TOKENIZER_100_JSON.as_bytes())
        .expect("first build failed");
    let idx2 = TokenLanguageIndex::build(&tok2, MOCK_TOKENIZER_ALT_JSON.as_bytes())
        .expect("second build failed");

    assert_ne!(
        idx1.vocab_hash, idx2.vocab_hash,
        "vocab_hash must differ for different tokenizer.json bytes"
    );
}

#[test]
fn version_constant_is_one() {
    assert_eq!(CURRENT_VERSION, 1, "CURRENT_VERSION must be 1");

    // Also verify the built index carries the correct version.
    let tok = make_tokenizer(MOCK_TOKENIZER_100_JSON);
    let idx = TokenLanguageIndex::build(&tok, MOCK_TOKENIZER_100_JSON.as_bytes())
        .expect("build failed");
    assert_eq!(idx.version, 1);
}

// ============================================================================
// B3 — Integration criterion: real tokenizer smoke test
// ============================================================================

/// Smoke test that loads a real tokenizer.json from a model in models/ and
/// builds a TokenLanguageIndex from it. This validates the full end-to-end
/// path that B4 (disk cache) and B5 (LangBiasSet conversion) will consume.
///
/// The test is skipped gracefully when no model directory is present so that
/// it does not break CI environments without downloaded models.
#[test]
fn build_index_from_real_tokenizer_smoke() {
    // Try multiple candidate tokenizer paths — use the first one found.
    let candidates = [
        "models/smollm-135m-4bit/tokenizer.json",
        "models/Qwen2.5-7B-Instruct-4bit/tokenizer.json",
        "models/Meta-Llama-3.1-8B-Instruct-4bit/tokenizer.json",
    ];

    let found = candidates.iter().find(|p| std::path::Path::new(p).exists());
    let Some(path) = found else {
        // No model downloaded — skip gracefully.
        eprintln!("[skip] no model tokenizer.json found; skipping real-tokenizer smoke test");
        return;
    };

    let bytes = std::fs::read(path).expect("failed to read tokenizer.json");
    let tok = tokenizers::Tokenizer::from_bytes(&bytes).expect("failed to parse tokenizer.json");

    let vocab_size = tok.get_vocab_size(true);
    assert!(vocab_size > 0, "tokenizer must have a non-empty vocabulary");

    let idx = TokenLanguageIndex::build(&tok, &bytes).expect("build should succeed on real tokenizer");

    assert_eq!(idx.tokens.len(), vocab_size, "index must have one entry per vocab token");
    assert_eq!(idx.version, CURRENT_VERSION);
    assert_eq!(
        idx.vocab_hash.len(),
        16,
        "vocab_hash must be exactly 16 hex chars"
    );

    // Verify the by_script inversion is consistent.
    for (script, ids) in &idx.by_script {
        for &token_id in ids {
            let info = &idx.tokens[token_id as usize];
            assert!(
                info.scripts.contains(script),
                "real tokenizer: token id {token_id} in by_script[{script:?}] but missing from info.scripts"
            );
        }
    }

    eprintln!(
        "[ok] built index from {path}: vocab={vocab_size}, by_script entries={}",
        idx.by_script.len()
    );
}
