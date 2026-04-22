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

use super::*;

fn tokens(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

#[test]
fn digest_is_stable_for_identical_inputs() {
    let toks = tokens(16);
    let a = PromptCacheKey::new_full("m", Some("l"), "tpl", Some("s"), &toks);
    let b = PromptCacheKey::new_full("m", Some("l"), "tpl", Some("s"), &toks);
    assert_eq!(a.digest(), b.digest());
}

#[test]
fn digest_changes_when_model_id_differs() {
    let toks = tokens(16);
    let a = PromptCacheKey::new_full("model-a", None, "tpl", None, &toks);
    let b = PromptCacheKey::new_full("model-b", None, "tpl", None, &toks);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn digest_changes_when_lora_differs() {
    let toks = tokens(16);
    let a = PromptCacheKey::new_full("m", None, "tpl", None, &toks);
    let b = PromptCacheKey::new_full("m", Some("lora-1"), "tpl", None, &toks);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn digest_changes_when_template_sig_differs() {
    let toks = tokens(16);
    let a = PromptCacheKey::new_full("m", None, "tpl-a", None, &toks);
    let b = PromptCacheKey::new_full("m", None, "tpl-b", None, &toks);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn digest_changes_when_session_key_differs() {
    let toks = tokens(16);
    let a = PromptCacheKey::new_full("m", None, "tpl", None, &toks);
    let b = PromptCacheKey::new_full("m", None, "tpl", Some("session-1"), &toks);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn digest_changes_with_each_extra_token() {
    let a_tokens = tokens(16);
    let b_tokens = tokens(17);
    let a = PromptCacheKey::new_full("m", None, "tpl", None, &a_tokens);
    let b = PromptCacheKey::new_full("m", None, "tpl", None, &b_tokens);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn prefix_len_saturates_at_token_length() {
    let toks = tokens(8);
    let k = PromptCacheKey::new_prefix("m", None, "tpl", None, &toks, 64);
    assert_eq!(k.effective_prefix_len(), 8);
}

#[test]
fn shorter_prefix_digest_matches_truncated_full_digest() {
    // A prefix-keyed digest over tokens[..8] must equal the full digest of
    // a 8-token input, because only the first `prefix_len` tokens are
    // hashed.
    let full = tokens(16);
    let truncated = tokens(8);

    let a = PromptCacheKey::new_prefix("m", None, "tpl", None, &full, 8);
    let b = PromptCacheKey::new_full("m", None, "tpl", None, &truncated);
    assert_eq!(a.digest(), b.digest());
}

#[test]
fn length_prefix_prevents_boundary_collisions() {
    // Without length prefixes, ("ab", "c") and ("a", "bc") concatenate to
    // the same string. Length prefixes + domain separator must distinguish
    // them.
    let toks = tokens(8);
    let a = PromptCacheKey::new_full("ab", None, "c", None, &toks);
    let b = PromptCacheKey::new_full("a", None, "bc", None, &toks);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn hex_roundtrip_length() {
    let toks = tokens(4);
    let d = PromptCacheKey::new_full("m", None, "tpl", None, &toks).digest();
    assert_eq!(d.to_hex().len(), 64);
    assert_eq!(d.short_hex().len(), 16);
}

#[test]
fn digest_treats_none_and_empty_string_identically() {
    // The length-prefixed wire format emits `[len=0][]` for both, so this
    // is intentional. Callers that need to distinguish "no session" from
    // "empty session id" must normalize upstream.
    let toks = tokens(16);
    let a = PromptCacheKey::new_full("m", None, "tpl", None, &toks);
    let b = PromptCacheKey::new_full("m", Some(""), "tpl", Some(""), &toks);
    assert_eq!(a.digest(), b.digest());
}

#[test]
fn token_order_matters() {
    let forward: Vec<i32> = (0..16).collect();
    let reversed: Vec<i32> = (0..16).rev().collect();
    let a = PromptCacheKey::new_full("m", None, "tpl", None, &forward);
    let b = PromptCacheKey::new_full("m", None, "tpl", None, &reversed);
    assert_ne!(a.digest(), b.digest());
}
