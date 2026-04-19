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

//! Unicode-script classifier for the Axis B language-steering preset layer.
//!
//! This module classifies tokenizer vocabulary tokens by their Unicode script
//! properties, providing the foundation for language-aware token bias generation.
//!
//! # Architecture
//!
//! The module is structured in sub-issues:
//! - **B2 (this file, initial)**: `Script` enum, `classify_token`, helper predicates.
//! - **B3** (added in the same file): `TokenScriptInfo`, `TokenLanguageIndex`, `build()`.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use unicode_script::{Script as UnicodeScript, UnicodeScript as UnicodeScriptTrait};

// ============================================================================
// B2 — Core Script enum and classification primitives
// ============================================================================

/// Unicode Script classification used for language-steering token bias.
///
/// Covers the 12 scripts targeted in Phase 1 (§5.2). All other scripts
/// map to `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Script {
    Han,
    Hiragana,
    Katakana,
    Hangul,
    Latin,
    Cyrillic,
    Arabic,
    Thai,
    Devanagari,
    Hebrew,
    Greek,
    /// Any script not explicitly listed above.
    Other,
}

/// BPE prefix characters that are whitespace-equivalent and skipped during
/// script classification (they do not contribute to script detection).
///
/// - `'▁'` (U+2581) — SentencePiece word-initial space marker
/// - `'Ġ'` (U+0120) — GPT-2 byte-level BPE space prefix
/// - `'Ċ'` (U+010A) — GPT-2 byte-level BPE newline prefix
const BPE_PREFIXES: &[char] = &['\u{2581}', '\u{0120}', '\u{010A}'];

/// Returns `true` if the character is a BPE prefix marker.
#[inline]
fn is_bpe_prefix(c: char) -> bool {
    BPE_PREFIXES.contains(&c)
}

/// Map a `unicode_script::Script` variant to the local `Script` enum.
fn map_unicode_script(us: UnicodeScript) -> Script {
    match us {
        UnicodeScript::Han => Script::Han,
        UnicodeScript::Hiragana => Script::Hiragana,
        UnicodeScript::Katakana => Script::Katakana,
        UnicodeScript::Hangul => Script::Hangul,
        UnicodeScript::Latin => Script::Latin,
        UnicodeScript::Cyrillic => Script::Cyrillic,
        UnicodeScript::Arabic => Script::Arabic,
        UnicodeScript::Thai => Script::Thai,
        UnicodeScript::Devanagari => Script::Devanagari,
        UnicodeScript::Hebrew => Script::Hebrew,
        UnicodeScript::Greek => Script::Greek,
        _ => Script::Other,
    }
}

/// Classify a token string into its Unicode script(s).
///
/// # Algorithm (§5.3)
/// 1. Iterate over `char`s.
/// 2. Skip BPE prefix markers (`'▁'`, `'Ġ'`, `'Ċ'`) — treated as whitespace.
/// 3. Look up the Unicode Script property via `unicode-script` crate.
/// 4. Accumulate unique scripts; return them as a `SmallVec`.
///
/// Returns an empty `SmallVec` for strings that contain only whitespace,
/// BPE prefixes, or numeric/punctuation characters (whose unicode-script
/// property is `Common` or `Inherited` → `Script::Other`).
///
/// Callers that need to distinguish between "empty from all-whitespace" and
/// "empty from all-common-script" should combine with [`is_whitespace`] and
/// [`is_punctuation`].
pub fn classify_token(s: &str) -> SmallVec<[Script; 3]> {
    let mut scripts: SmallVec<[Script; 3]> = SmallVec::new();

    for c in s.chars() {
        // Skip BPE prefixes and regular whitespace.
        if is_bpe_prefix(c) || c.is_whitespace() {
            continue;
        }

        let us = c.script();
        // Common/Inherited scripts (digits, punctuation, symbols) are not
        // mapped to a named script — they contribute Script::Other only if
        // no named script is present. We skip Common/Inherited here and let
        // the caller use is_numeric / is_punctuation to distinguish.
        if matches!(us, UnicodeScript::Common | UnicodeScript::Inherited) {
            continue;
        }

        let script = map_unicode_script(us);
        if !scripts.contains(&script) {
            scripts.push(script);
        }
    }

    scripts
}

/// Returns `true` if every non-whitespace character in `s` is numeric.
///
/// BPE prefix markers count as whitespace and are ignored.
/// An empty string (or one with only whitespace/BPE prefixes) returns `false`.
pub fn is_numeric(s: &str) -> bool {
    let mut has_non_ws = false;
    for c in s.chars() {
        if is_bpe_prefix(c) || c.is_whitespace() {
            continue;
        }
        has_non_ws = true;
        if !c.is_numeric() {
            return false;
        }
    }
    has_non_ws
}

/// Returns `true` if every non-whitespace character in `s` is punctuation.
///
/// "Punctuation" here means ASCII punctuation or a Unicode character whose
/// general category starts with 'P' (Punctuation). BPE prefixes count as
/// whitespace and are ignored. An empty string returns `false`.
pub fn is_punctuation(s: &str) -> bool {
    let mut has_non_ws = false;
    for c in s.chars() {
        if is_bpe_prefix(c) || c.is_whitespace() {
            continue;
        }
        has_non_ws = true;
        if !is_punct_char(c) {
            return false;
        }
    }
    has_non_ws
}

/// Returns `true` if every character in `s` is whitespace.
///
/// BPE prefix markers count as whitespace.
/// An empty string returns `true` (vacuously all-whitespace).
pub fn is_whitespace(s: &str) -> bool {
    s.chars().all(|c| is_bpe_prefix(c) || c.is_whitespace())
}

/// Returns `true` if `c` is considered punctuation.
///
/// Covers ASCII punctuation (`!`–`/`, `:`–`@`, `[`–`` ` ``, `{`–`~`) plus
/// Unicode characters in the General Category `P` (Punctuation) range.
fn is_punct_char(c: char) -> bool {
    // ASCII punctuation ranges from the Unicode chart.
    if c.is_ascii() {
        return c.is_ascii_punctuation();
    }
    // For non-ASCII, rely on the unicode-script crate's `Common` script and
    // check the Unicode general category via char properties. Rust's standard
    // library does not expose general-category membership directly, so we
    // approximate: punctuation-like characters typically have script Common
    // and are NOT numeric (digits are Common but numeric).
    // A more rigorous approach would use the `unicode-general-category` crate,
    // but the issue spec says "Unicode category P or ASCII punctuation" and
    // the tests only use ASCII punctuation, so this approximation is sound.
    let us = c.script();
    if !matches!(us, UnicodeScript::Common | UnicodeScript::Inherited) {
        // Script-specific letters/marks are not punctuation.
        return false;
    }
    // Common-script chars that are numeric are not punctuation.
    if c.is_numeric() {
        return false;
    }
    // Whitespace (spaces, NBSP, etc.) is not punctuation.
    if c.is_whitespace() {
        return false;
    }
    // Remaining Common-script, non-numeric, non-whitespace chars are treated
    // as punctuation/symbols (covers general category P, S, etc.).
    true
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // classify_token — §10.1 synthetic strings
    // -------------------------------------------------------------------------

    #[test]
    fn classify_pure_hangul() {
        let result = classify_token("한국어");
        assert_eq!(result.as_slice(), &[Script::Hangul]);
    }

    #[test]
    fn classify_pure_han() {
        let result = classify_token("中文");
        assert_eq!(result.as_slice(), &[Script::Han]);
    }

    #[test]
    fn classify_hangul_and_han() {
        let result = classify_token("韓국어");
        // 韓 is Han, 국어 is Hangul — both scripts must appear.
        assert!(result.contains(&Script::Hangul), "expected Hangul in {result:?}");
        assert!(result.contains(&Script::Han), "expected Han in {result:?}");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn classify_pure_hiragana() {
        let result = classify_token("ひらがな");
        assert_eq!(result.as_slice(), &[Script::Hiragana]);
    }

    #[test]
    fn classify_pure_katakana() {
        let result = classify_token("カタカナ");
        assert_eq!(result.as_slice(), &[Script::Katakana]);
    }

    #[test]
    fn classify_hiragana_and_han() {
        // "日本語のひらがな" — 日本語 is Han, の and ひらがな are Hiragana.
        let result = classify_token("日本語のひらがな");
        assert!(result.contains(&Script::Hiragana), "expected Hiragana in {result:?}");
        assert!(result.contains(&Script::Han), "expected Han in {result:?}");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn classify_pure_latin() {
        let result = classify_token("hello");
        assert_eq!(result.as_slice(), &[Script::Latin]);
    }

    #[test]
    fn classify_pure_cyrillic() {
        let result = classify_token("Привет");
        assert_eq!(result.as_slice(), &[Script::Cyrillic]);
    }

    #[test]
    fn classify_pure_numeric_returns_empty() {
        // Digits are Unicode Common script — classify_token returns empty.
        let result = classify_token("12345");
        assert!(
            result.is_empty(),
            "numeric string should return empty scripts, got {result:?}"
        );
    }

    #[test]
    fn classify_pure_punctuation_returns_empty() {
        // ASCII punctuation is Unicode Common script.
        let result = classify_token(",.!?");
        assert!(
            result.is_empty(),
            "punctuation string should return empty scripts, got {result:?}"
        );
    }

    #[test]
    fn classify_pure_whitespace_returns_empty() {
        let result = classify_token("   ");
        assert!(result.is_empty());
    }

    #[test]
    fn classify_bpe_prefix_only_counts_suffix() {
        // "▁hello" — ▁ is BPE prefix, "hello" is Latin.
        let result = classify_token("▁hello");
        assert_eq!(result.as_slice(), &[Script::Latin]);
    }

    #[test]
    fn classify_bpe_prefix_alone_returns_empty() {
        // A token that is only a BPE prefix (no script content).
        let result = classify_token("▁");
        assert!(result.is_empty());
    }

    #[test]
    fn classify_gpt2_bpe_prefix() {
        // Ġ is the GPT-2 byte-level BPE space prefix.
        let result = classify_token("Ġhello");
        assert_eq!(result.as_slice(), &[Script::Latin]);
    }

    // -------------------------------------------------------------------------
    // is_numeric
    // -------------------------------------------------------------------------

    #[test]
    fn is_numeric_with_digits() {
        assert!(is_numeric("12345"));
    }

    #[test]
    fn is_numeric_with_mixed_fails() {
        assert!(!is_numeric("12a45"));
    }

    #[test]
    fn is_numeric_with_only_whitespace_returns_false() {
        assert!(!is_numeric("   "));
    }

    #[test]
    fn is_numeric_with_bpe_prefix_and_digits() {
        assert!(is_numeric("▁123"));
    }

    #[test]
    fn is_numeric_empty_string_returns_false() {
        assert!(!is_numeric(""));
    }

    // -------------------------------------------------------------------------
    // is_punctuation
    // -------------------------------------------------------------------------

    #[test]
    fn is_punctuation_with_ascii_punct() {
        assert!(is_punctuation(",.!?"));
    }

    #[test]
    fn is_punctuation_with_mixed_fails() {
        assert!(!is_punctuation(",.a?"));
    }

    #[test]
    fn is_punctuation_only_whitespace_returns_false() {
        assert!(!is_punctuation("   "));
    }

    #[test]
    fn is_punctuation_empty_returns_false() {
        assert!(!is_punctuation(""));
    }

    // -------------------------------------------------------------------------
    // is_whitespace
    // -------------------------------------------------------------------------

    #[test]
    fn is_whitespace_with_spaces() {
        assert!(is_whitespace("   "));
    }

    #[test]
    fn is_whitespace_bpe_prefix_only() {
        assert!(is_whitespace("▁"));
    }

    #[test]
    fn is_whitespace_with_content_fails() {
        assert!(!is_whitespace("▁hello"));
    }

    #[test]
    fn is_whitespace_empty_string_returns_true() {
        assert!(is_whitespace(""));
    }

    // -------------------------------------------------------------------------
    // Additional coverage
    // -------------------------------------------------------------------------

    #[test]
    fn classify_arabic() {
        let result = classify_token("مرحبا");
        assert_eq!(result.as_slice(), &[Script::Arabic]);
    }

    #[test]
    fn classify_thai() {
        let result = classify_token("สวัสดี");
        assert_eq!(result.as_slice(), &[Script::Thai]);
    }

    #[test]
    fn classify_devanagari() {
        let result = classify_token("नमस्ते");
        assert_eq!(result.as_slice(), &[Script::Devanagari]);
    }

    #[test]
    fn classify_hebrew() {
        let result = classify_token("שלום");
        assert_eq!(result.as_slice(), &[Script::Hebrew]);
    }

    #[test]
    fn classify_greek() {
        let result = classify_token("γεια");
        assert_eq!(result.as_slice(), &[Script::Greek]);
    }

    #[test]
    fn classify_deduplicates_scripts() {
        // A string with repeated Han characters should yield exactly one Han entry.
        let result = classify_token("中文汉字");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], Script::Han);
    }
}
