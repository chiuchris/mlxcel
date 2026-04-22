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

// ---------------------------------------------------------------------------
// Issue #422 — session-key resolution
// ---------------------------------------------------------------------------

#[test]
fn resolve_session_key_prefers_prompt_cache_key() {
    let got = resolve_session_key(Some("pck"), Some("user-1"));
    assert_eq!(got, "pck");
}

#[test]
fn resolve_session_key_falls_back_to_user() {
    let got = resolve_session_key(None, Some("user-1"));
    assert_eq!(got, "user-1");
}

#[test]
fn resolve_session_key_uses_anonymous_sentinel_when_both_absent() {
    let got = resolve_session_key(None, None);
    assert_eq!(got, ANONYMOUS_SESSION_SENTINEL);
    // The sentinel is non-empty so it distinguishes from a `None` session.
    assert!(!got.is_empty());
}

#[test]
fn resolve_session_key_treats_empty_prompt_cache_key_as_absent() {
    let got = resolve_session_key(Some(""), Some("user-1"));
    assert_eq!(got, "user-1");
}

#[test]
fn resolve_session_key_treats_empty_user_as_absent() {
    let got = resolve_session_key(None, Some(""));
    assert_eq!(got, ANONYMOUS_SESSION_SENTINEL);
}

#[test]
fn resolve_session_key_all_empty_returns_sentinel() {
    let got = resolve_session_key(Some(""), Some(""));
    assert_eq!(got, ANONYMOUS_SESSION_SENTINEL);
}

// ---------------------------------------------------------------------------
// Issue #422 — template signature
// ---------------------------------------------------------------------------

fn sample_tool(name: &str) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: crate::server::types::request::FunctionDefinition {
            name: name.to_string(),
            description: Some(format!("{name} description")),
            parameters: Some(serde_json::json!({"type": "object"})),
        },
    }
}

#[test]
fn template_sig_is_stable_for_identical_inputs() {
    let kw = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": true}"#).unwrap();
    let a = template_sig("tpl", &kw, None, None);
    let b = template_sig("tpl", &kw, None, None);
    assert_eq!(a, b);
    assert_eq!(a.len(), 64, "must be 64-char hex");
}

#[test]
fn template_sig_changes_when_template_source_changes() {
    let kw = ChatTemplateKwargs::new();
    let a = template_sig("template-a", &kw, None, None);
    let b = template_sig("template-b", &kw, None, None);
    assert_ne!(a, b);
}

#[test]
fn template_sig_changes_when_kwargs_change() {
    let kw_a = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": true}"#).unwrap();
    let kw_b = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": false}"#).unwrap();
    let a = template_sig("tpl", &kw_a, None, None);
    let b = template_sig("tpl", &kw_b, None, None);
    assert_ne!(a, b);
}

#[test]
fn template_sig_canonicalizes_kwargs_key_order() {
    // {"a":1,"b":2} and {"b":2,"a":1} — canonicalization must collapse
    // these to the same digest so map-insertion-order drift is absorbed.
    let kw_a = ChatTemplateKwargs::from_json_str(r#"{"a": 1, "b": 2}"#).unwrap();
    let kw_b = ChatTemplateKwargs::from_json_str(r#"{"b": 2, "a": 1}"#).unwrap();
    let a = template_sig("tpl", &kw_a, None, None);
    let b = template_sig("tpl", &kw_b, None, None);
    assert_eq!(a, b, "kwargs key order must not affect the signature");
}

#[test]
fn template_sig_changes_when_tool_choice_mode_changes() {
    let kw = ChatTemplateKwargs::new();
    let a = template_sig(
        "tpl",
        &kw,
        Some(&ToolChoice::Mode("auto".to_string())),
        None,
    );
    let b = template_sig(
        "tpl",
        &kw,
        Some(&ToolChoice::Mode("none".to_string())),
        None,
    );
    assert_ne!(a, b);
}

#[test]
fn template_sig_distinguishes_absent_from_auto_tool_choice() {
    let kw = ChatTemplateKwargs::new();
    let absent = template_sig("tpl", &kw, None, None);
    let auto = template_sig(
        "tpl",
        &kw,
        Some(&ToolChoice::Mode("auto".to_string())),
        None,
    );
    assert_ne!(absent, auto);
}

#[test]
fn template_sig_changes_when_tools_added() {
    let kw = ChatTemplateKwargs::new();
    let tools_a: Vec<Tool> = vec![];
    let tools_b = vec![sample_tool("get_weather")];
    let a = template_sig("tpl", &kw, None, Some(&tools_a));
    let b = template_sig("tpl", &kw, None, Some(&tools_b));
    assert_ne!(a, b);
}

#[test]
fn template_sig_changes_when_tool_removed() {
    let kw = ChatTemplateKwargs::new();
    let two = vec![sample_tool("get_weather"), sample_tool("send_email")];
    let one = vec![sample_tool("get_weather")];
    let a = template_sig("tpl", &kw, None, Some(&two));
    let b = template_sig("tpl", &kw, None, Some(&one));
    assert_ne!(a, b);
}

#[test]
fn template_sig_changes_when_tools_reordered() {
    // tools_digest is order-preserving by design; reordering must change
    // the signature.
    let kw = ChatTemplateKwargs::new();
    let forward = vec![sample_tool("a"), sample_tool("b")];
    let reversed = vec![sample_tool("b"), sample_tool("a")];
    let a = template_sig("tpl", &kw, None, Some(&forward));
    let b = template_sig("tpl", &kw, None, Some(&reversed));
    assert_ne!(
        a, b,
        "tool reordering must invalidate the template signature"
    );
}

#[test]
fn template_sig_treats_empty_tools_and_none_tools_identically() {
    // Both map to the same "no tools" marker in the digest.
    let kw = ChatTemplateKwargs::new();
    let empty: Vec<Tool> = vec![];
    let a = template_sig("tpl", &kw, None, None);
    let b = template_sig("tpl", &kw, None, Some(&empty));
    assert_eq!(a, b);
}

#[test]
fn tools_digest_changes_with_parameters() {
    // The function.parameters JSON Schema participates in the digest.
    let mut tool_a = sample_tool("weather");
    let mut tool_b = sample_tool("weather");
    tool_a.function.parameters =
        Some(serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}}}));
    tool_b.function.parameters =
        Some(serde_json::json!({"type": "object", "properties": {"b": {"type": "string"}}}));
    let a = tools_digest(Some(&[tool_a]));
    let b = tools_digest(Some(&[tool_b]));
    assert_ne!(a, b);
}

#[test]
fn tools_digest_canonicalizes_parameter_key_order() {
    // Re-ordering JSON object keys inside parameters must not change the digest.
    let mut tool_a = sample_tool("fn");
    let mut tool_b = sample_tool("fn");
    tool_a.function.parameters = Some(serde_json::json!({"a": 1, "b": 2}));
    tool_b.function.parameters = Some(serde_json::json!({"b": 2, "a": 1}));
    let a = tools_digest(Some(&[tool_a]));
    let b = tools_digest(Some(&[tool_b]));
    assert_eq!(a, b);
}

#[test]
fn template_sig_changes_with_tool_name() {
    // Two tools with the same type and description but different names must
    // still yield different signatures.
    let kw = ChatTemplateKwargs::new();
    let a = template_sig("tpl", &kw, None, Some(&[sample_tool("alpha")]));
    let b = template_sig("tpl", &kw, None, Some(&[sample_tool("beta")]));
    assert_ne!(a, b);
}

#[test]
fn template_sig_fits_into_prompt_cache_key() {
    // Smoke: signature hex can slot straight into the cache key's
    // `template_sig` field.
    let kw = ChatTemplateKwargs::new();
    let sig = template_sig("tpl", &kw, None, None);
    let toks = tokens(4);
    let key = PromptCacheKey::new_full("m", None, sig.as_str(), None, &toks);
    assert_eq!(key.template_sig, sig.as_str());
    // Computes a digest without panicking.
    let _ = key.digest();
}
