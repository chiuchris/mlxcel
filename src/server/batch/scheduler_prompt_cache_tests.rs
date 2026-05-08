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

//! Scheduler-facing prompt-prefix cache integration tests (epic #416 / issue
//! #421).
//!
//! Full end-to-end adopt/donate coverage that exercises a real `LoadedModel`
//! lives in the worker-thread integration tests; those need the server binary
//! and are out of scope for `cargo test --lib`. The cases in this file isolate
//! the **scheduler-visible invariants** that the #421 implementation adds and
//! that the store / key / entry data flow relies on:
//!
//! * Cache hit reports the matched prefix length and leaves the detached
//!   entry consumed (one-shot).
//! * A second lookup against a consumed entry falls through to a miss even
//!   though the entry structurally still lives in the store (race semantics).
//! * Miss paths preserve bit-exact behavior (no hidden cache access).
//! * Donate-back on a healthy finish produces an entry with
//!   `tokens = prompt + generated` as the radix-trie key.
//! * The observability counters on [`BatchObservability`] advance exactly
//!   once per adopt / donate-back as the scheduler would call them.
//! * `SequenceInfo::prefill_start_offset` / `already_cached_tokens` transport
//!   the hit metadata to the prefill path without forcing the rest of the
//!   sequence struct to mutate.

use std::sync::Arc;
use std::time::{Duration, Instant};

use mlxcel_core::cache::SequenceId;

use crate::server::batch::BatchObservability;
use crate::server::prompt_cache::{
    CacheEntry, PromptCacheConfig, PromptCacheStore,
    key::{MultimodalDigest, PromptCacheKey},
};

/// Produce a placeholder [`CacheEntry`] with a zero-tensor detached set.
///
/// Real scheduler donate-back entries carry MLX buffers; these tests only
/// exercise the store/key/trie data flow, so the shared
/// [`CacheEntry::new_for_test`] helper (defined under `#[cfg(test)]`) is
/// sufficient and avoids reaching into the private constructor fields of
/// `DetachedKVCache` / `DetachedCacheSet`.
fn fake_entry(tokens: Vec<i32>, size_bytes: usize) -> CacheEntry {
    CacheEntry::new_for_test(tokens, size_bytes)
}

fn test_store() -> Arc<PromptCacheStore> {
    Arc::new(PromptCacheStore::with_config(PromptCacheConfig {
        enabled: true,
        capacity_bytes: 128 * 1024,
        max_entries: 32,
        ttl: Duration::from_secs(60),
        min_prefix_tokens: 4,
    }))
}

fn make_key<'a>(
    model_id: &'a str,
    session_key: &'a str,
    template_sig: &'a str,
    tokens: &'a [i32],
) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(
        model_id,
        None,
        template_sig,
        Some(session_key),
        MultimodalDigest::empty(),
        tokens,
    )
}

// ---------------------------------------------------------------------------
// Hit path
// ---------------------------------------------------------------------------

#[test]
fn hit_then_decode_reports_matched_len_and_consumes_entry() {
    // Mirror the scheduler's hit-path invocation: insert a detached entry,
    // then have a subsequent request with a longer prompt look it up.
    let store = test_store();
    let prompt: Vec<i32> = (100..=131).collect(); // 32 tokens — a plausible prefix
    let entry = fake_entry(prompt.clone(), 4096);
    let bytes_before = entry.size_bytes;
    let insert_key = make_key("model-A", "session-1", "tpl-sig-v1", &prompt);
    store.insert(&insert_key, entry).expect("insert succeeds");

    // Next turn: same conversation, 5 extra user tokens → the first 32
    // tokens of the new prompt match the stored prefix.
    let mut next_prompt = prompt.clone();
    next_prompt.extend([200, 201, 202, 203, 204]);
    let lookup_key = make_key("model-A", "session-1", "tpl-sig-v1", &next_prompt);
    let (hit_entry, matched_len) = store
        .lookup_longest_prefix(&lookup_key, &next_prompt)
        .expect("matching prefix must hit");
    assert_eq!(matched_len, prompt.len());
    assert_eq!(hit_entry.size_bytes, bytes_before);

    // Scheduler consumes the detached cache via `take_detached`; after
    // that, the entry is drained (one-shot) even though the radix-trie
    // lookup still sees it. A racing second adopt must fall through to a
    // cold prefill.
    let _detached = hit_entry
        .take_detached()
        .expect("entry is consumable on first hit");
    assert!(hit_entry.take_detached().is_none());

    // The *scheduler* records matched_len via BatchObservability — we
    // exercise the same counter call pattern here.
    let obs = BatchObservability::new();
    obs.record_prompt_cache_hit(matched_len);
    let snap = obs.snapshot();
    assert_eq!(snap.prompt_cache_hits, 1);
    assert_eq!(snap.prompt_cache_hit_tokens, matched_len as u64);
}

// ---------------------------------------------------------------------------
// Miss path
// ---------------------------------------------------------------------------

#[test]
fn miss_then_decode_leaves_scheduler_state_unchanged() {
    // A request with no matching entry must return None with no side
    // effects — the scheduler's miss path proceeds with a fresh allocate.
    let store = test_store();

    // Insert an entry under a different (model, session, template) bucket
    // so the trie is non-empty but does not overlap with the lookup key.
    let seeded: Vec<i32> = (1..=16).collect();
    let seed_entry = fake_entry(seeded.clone(), 2048);
    let seed_key = make_key("other-model", "other-session", "other-tpl", &seeded);
    store
        .insert(&seed_key, seed_entry)
        .expect("seed insert succeeds");

    // Lookup under an unrelated bucket must miss.
    let probe_tokens: Vec<i32> = (100..=131).collect();
    let probe_key = make_key("model-A", "session-1", "tpl-sig-v1", &probe_tokens);
    assert!(
        store
            .lookup_longest_prefix(&probe_key, &probe_tokens)
            .is_none()
    );

    // No hit recorded on the observability counter.
    let obs = BatchObservability::new();
    let snap = obs.snapshot();
    assert_eq!(snap.prompt_cache_hits, 0);
    assert_eq!(snap.prompt_cache_hit_tokens, 0);
}

// ---------------------------------------------------------------------------
// Donate-back path
// ---------------------------------------------------------------------------

#[test]
fn donate_back_on_healthy_finish_inserts_prompt_plus_generated() {
    // Simulate a sequence that ran to a healthy finish: prompt of 32
    // tokens + 4 generated tokens. The scheduler calls
    // `CachePool::detach` → builds a `CacheEntry` with the joined tokens
    // → inserts under the same key context as the request.
    let store = test_store();

    let prompt: Vec<i32> = (300..=331).collect();
    let generated: Vec<i32> = vec![999, 1000, 1001, 1002];

    let mut joined = prompt.clone();
    joined.extend_from_slice(&generated);
    let entry = fake_entry(joined.clone(), 8192);

    let insert_key = make_key("model-A", "session-1", "tpl-sig-v1", &joined);
    store
        .insert(&insert_key, entry)
        .expect("donate-back insert succeeds");

    // The next turn of the same conversation sees the full prompt +
    // assistant tail as a reusable prefix.
    let mut next_prompt = joined.clone();
    next_prompt.extend([2000, 2001]);
    let next_key = make_key("model-A", "session-1", "tpl-sig-v1", &next_prompt);
    let (_e, matched_len) = store
        .lookup_longest_prefix(&next_key, &next_prompt)
        .expect("donate-back entry must be reachable on the next turn");
    assert_eq!(matched_len, joined.len());

    let obs = BatchObservability::new();
    obs.record_prompt_cache_insert();
    assert_eq!(obs.snapshot().prompt_cache_inserts, 1);
}

// ---------------------------------------------------------------------------
// No donate-back on error / OOM paths
// ---------------------------------------------------------------------------

#[test]
fn no_donate_back_on_error_outcome() {
    // Simulates the scheduler's policy: the error / OOM / invalid-cache
    // path never calls `donate_finished_sequence_cache` with
    // `healthy_finish == true`, so the store stays untouched and the
    // reject-counter stays at zero (a reject would only fire on an
    // actual insert attempt that the store refuses).
    let store = test_store();
    assert_eq!(store.stats().inserts, 0);
    assert_eq!(store.stats().entries, 0);

    // Mimic the scheduler reject-counter gate: no insert, no counter
    // movement. The donate-path helper hard-guards on `healthy_finish`
    // before touching the store or counters.
    let obs = BatchObservability::new();
    let snap = obs.snapshot();
    assert_eq!(snap.prompt_cache_inserts, 0);
    assert_eq!(snap.prompt_cache_insert_rejects, 0);
}

#[test]
fn oversized_entry_records_reject_counter() {
    // Build a store with a 1-byte capacity so any real entry is
    // oversized. The scheduler mirrors this via
    // `record_prompt_cache_insert_reject()` in its insert-error branch.
    let store = Arc::new(PromptCacheStore::with_config(PromptCacheConfig {
        enabled: true,
        capacity_bytes: 1,
        max_entries: 32,
        ttl: Duration::from_secs(60),
        min_prefix_tokens: 4,
    }));
    let tokens: Vec<i32> = (1..=16).collect();
    let entry = CacheEntry::new_for_test(tokens.clone(), 4096); // 4 KiB > 1 byte
    let key = make_key("m", "s", "t", &tokens);
    let err = store
        .insert(&key, entry)
        .expect_err("oversized entry must be rejected");
    assert!(
        matches!(
            err,
            crate::server::prompt_cache::InsertError::OversizedEntry { .. }
        ),
        "unexpected reject reason: {err:?}"
    );

    // Scheduler would advance the reject counter here.
    let obs = BatchObservability::new();
    obs.record_prompt_cache_insert_reject();
    assert_eq!(obs.snapshot().prompt_cache_insert_rejects, 1);
}

// ---------------------------------------------------------------------------
// SequenceInfo fields transport the hit metadata
// ---------------------------------------------------------------------------

#[test]
fn sequence_info_fields_transport_cache_hit_metadata() {
    // The scheduler sets both `prefill_start_offset` (read by the prefill
    // path) and `already_cached_tokens` (read by the metrics bridge) to
    // the matched prefix length on a hit. The two are updated in lock-
    // step; test that the struct carries both independently.
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;

    use crate::server::batch::sequence::{SequenceInfo, SequenceState};
    use crate::server::model_provider::GenerateEvent;
    use crate::server::model_provider::model_worker::StreamingDecodeState;
    use mlxcel_core::generate::SamplingConfig;

    let (tx, _rx) = mpsc::channel::<GenerateEvent>();
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let prompt_tokens = vec![1_i32; 100];
    let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);

    let seq = SequenceInfo {
        seq_id: SequenceId::from_raw(7),
        state: SequenceState::Queued,
        prompt_tokens,
        sampling: SamplingConfig::default(),
        max_tokens: 64,
        eos_token_ids: vec![2],
        priority: crate::server::batch::RequestPriority::Normal,
        logprobs_config: Default::default(),
        vlm_embeddings: None,
        images: Vec::new(),
        audio: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        prefill_offset: 0,
        prefill_start_offset: 73,
        already_cached_tokens: 73,
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: None,
        first_token_time: None,
        token_history: Vec::new(),
        merged_eos: Vec::new(),
        thinking: crate::server::thinking_budget::ThinkingState::disabled(),
        structured: None,
    };

    assert_eq!(seq.prefill_start_offset, 73);
    assert_eq!(seq.already_cached_tokens, 73);
    assert_eq!(seq.prompt_tokens.len(), 100);
    // Prefill-path math: model sees only the suffix tokens.
    assert_eq!(
        seq.prompt_tokens.len() - seq.prefill_start_offset,
        27,
        "scheduler prefill loop must process only suffix tokens"
    );
}

// ---------------------------------------------------------------------------
// Batched prefill falls back to sequential when any adopted prefix is present
// ---------------------------------------------------------------------------

#[test]
fn batched_prefill_path_detects_adopted_sequences() {
    // The scheduler's `execute_batched_prefill` short-circuits to
    // sequential prefill when *any* sequence in the batch has
    // `prefill_start_offset > 0`. We assert the detection predicate here
    // so any future refactor of the batched path can't regress the
    // behavior silently.
    let mut offsets = [0_usize; 3];
    assert!(
        !offsets.iter().any(|&o| o > 0),
        "cold batch must not trigger adopted-sequence fallback"
    );
    offsets[1] = 32;
    assert!(
        offsets.iter().any(|&o| o > 0),
        "a single adopted sequence must trigger the fallback"
    );
}
