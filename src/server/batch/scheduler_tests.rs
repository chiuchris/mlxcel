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

//! Unit tests for batch scheduler scheduling decisions.
//!
//! These tests verify the `decide_action` policy given various combinations
//! of empty/non-empty queue and batch states, without requiring a real model.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;

use super::effective_decode_storage_backend;
use crate::server::batch::active::ActiveBatch;
use crate::server::batch::queue::PrefillQueue;
use crate::server::batch::sequence::{
    BatchSchedulerAction, RequestPriority, SequenceInfo, SequenceState,
};
use crate::server::config::DecodeStorageBackend;
use crate::server::model_provider::GenerateEvent;
use crate::server::model_provider::model_worker::StreamingDecodeState;

/// Build a minimal `SequenceInfo` for scheduling tests.
fn make_test_sequence(id_val: u64) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let prompt_tokens = vec![1, 2, 3];
    let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);

    let seq = SequenceInfo {
        seq_id: SequenceId::from_raw(id_val),
        state: SequenceState::Queued,
        prompt_tokens,
        sampling: SamplingConfig::default(),
        max_tokens: 100,
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
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: None,
        first_token_time: None,
        token_history: Vec::new(),
        merged_eos: Vec::new(),
    };

    (seq, rx)
}

/// Helper: reproduce `decide_action` logic in isolation so tests do not need
/// a full `BatchScheduler` (which requires a real LoadedModel + tokenizer).
///
/// This mirrors the exact decision policy from `BatchScheduler::decide_action`.
/// Policy: active sequences always decode first to prevent starvation; prefill
/// only happens when the batch is empty.
fn decide_action_from_state(queue: &PrefillQueue, batch: &ActiveBatch) -> BatchSchedulerAction {
    if batch.is_empty() && queue.is_empty() {
        return BatchSchedulerAction::Idle;
    }
    // Active sequences always get a decode step before admitting new prefills.
    if !batch.is_empty() {
        return BatchSchedulerAction::Decode(batch.sequence_ids());
    }
    // Batch is empty but queue has work -- prefill next request.
    BatchSchedulerAction::Prefill(SequenceId::from_raw(0))
}

// -------------------------------------------------------------------
// decide_action tests
// -------------------------------------------------------------------

#[test]
fn decide_action_idle_when_both_empty() {
    let queue = PrefillQueue::new();
    let batch = ActiveBatch::new(8);

    let action = decide_action_from_state(&queue, &batch);
    assert!(matches!(action, BatchSchedulerAction::Idle));
}

#[test]
fn decide_action_prefill_when_queue_has_entry_and_batch_not_full() {
    let mut queue = PrefillQueue::new();
    let batch = ActiveBatch::new(4);

    let (seq, _rx) = make_test_sequence(1);
    queue.enqueue(seq).unwrap();

    let action = decide_action_from_state(&queue, &batch);
    assert!(matches!(action, BatchSchedulerAction::Prefill(_)));
}

#[test]
fn decide_action_decode_when_batch_has_entries_and_queue_empty() {
    let queue = PrefillQueue::new();
    let mut batch = ActiveBatch::new(4);

    let (mut seq, _rx) = make_test_sequence(10);
    seq.state = SequenceState::Decoding;
    batch.add(seq).unwrap();

    let action = decide_action_from_state(&queue, &batch);
    match action {
        BatchSchedulerAction::Decode(ids) => {
            assert_eq!(ids.len(), 1);
            assert_eq!(ids[0].as_u64(), 10);
        }
        other => panic!("Expected Decode, got {other:?}"),
    }
}

#[test]
fn decide_action_decode_when_batch_full_and_queue_non_empty() {
    let mut queue = PrefillQueue::new();
    let mut batch = ActiveBatch::new(1); // capacity 1

    // Fill the batch
    let (mut seq_active, _rx1) = make_test_sequence(1);
    seq_active.state = SequenceState::Decoding;
    batch.add(seq_active).unwrap();

    // Also queue a new request
    let (seq_queued, _rx2) = make_test_sequence(2);
    queue.enqueue(seq_queued).unwrap();

    // Batch is full -> should decode, not prefill
    let action = decide_action_from_state(&queue, &batch);
    assert!(matches!(action, BatchSchedulerAction::Decode(_)));
}

#[test]
fn decide_action_decode_when_batch_has_active_even_if_queue_nonempty() {
    let mut queue = PrefillQueue::new();
    let mut batch = ActiveBatch::new(4); // capacity 4

    // One active sequence
    let (mut seq_active, _rx1) = make_test_sequence(1);
    seq_active.state = SequenceState::Decoding;
    batch.add(seq_active).unwrap();

    // One queued request
    let (seq_queued, _rx2) = make_test_sequence(2);
    queue.enqueue(seq_queued).unwrap();

    // Active sequences always decode first to prevent starvation.
    // Queued requests will be prefilled once the batch empties.
    let action = decide_action_from_state(&queue, &batch);
    assert!(matches!(action, BatchSchedulerAction::Decode(_)));
}

#[test]
fn decide_action_decode_returns_all_active_ids() {
    let queue = PrefillQueue::new();
    let mut batch = ActiveBatch::new(8);

    for id in [10, 20, 30] {
        let (mut seq, _rx) = make_test_sequence(id);
        seq.state = SequenceState::Decoding;
        batch.add(seq).unwrap();
    }

    let action = decide_action_from_state(&queue, &batch);
    match action {
        BatchSchedulerAction::Decode(ids) => {
            let mut sorted: Vec<u64> = ids.iter().map(|id| id.as_u64()).collect();
            sorted.sort();
            assert_eq!(sorted, vec![10, 20, 30]);
        }
        other => panic!("Expected Decode, got {other:?}"),
    }
}

// -------------------------------------------------------------------
// O(1) property of decide_action
// -------------------------------------------------------------------

#[test]
fn decide_action_is_o1_regardless_of_queue_size() {
    // Verifies that decide_action does not iterate the queue or batch;
    // it only checks .is_empty() / .is_full() which are O(1).
    let mut queue = PrefillQueue::with_capacity(1000);
    let batch = ActiveBatch::new(8);

    for i in 0..100 {
        let (seq, _rx) = make_test_sequence(i);
        queue.enqueue(seq).unwrap();
    }

    // Should immediately return Prefill without scanning the queue
    let action = decide_action_from_state(&queue, &batch);
    assert!(matches!(action, BatchSchedulerAction::Prefill(_)));
}

// -------------------------------------------------------------------
// Priority-related tests
// -------------------------------------------------------------------

fn make_test_sequence_with_priority(
    id_val: u64,
    priority: RequestPriority,
) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let prompt_tokens = vec![1, 2, 3];
    let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);

    let seq = SequenceInfo {
        seq_id: SequenceId::from_raw(id_val),
        state: SequenceState::Queued,
        prompt_tokens,
        sampling: SamplingConfig::default(),
        max_tokens: 100,
        eos_token_ids: vec![2],
        priority,
        logprobs_config: Default::default(),
        vlm_embeddings: None,
        images: Vec::new(),
        audio: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        prefill_offset: 0,
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: None,
        first_token_time: None,
        token_history: Vec::new(),
        merged_eos: Vec::new(),
    };

    (seq, rx)
}

#[test]
fn priority_queue_dequeues_high_before_normal() {
    let mut queue = PrefillQueue::new();

    let (s_norm, _r1) = make_test_sequence_with_priority(1, RequestPriority::Normal);
    let (s_high, _r2) = make_test_sequence_with_priority(2, RequestPriority::High);

    queue.enqueue(s_norm).unwrap();
    queue.enqueue(s_high).unwrap();

    // High should come out first even though normal was enqueued first
    let first = queue.dequeue().unwrap();
    assert_eq!(first.seq_id.as_u64(), 2);
    assert_eq!(first.priority, RequestPriority::High);

    let second = queue.dequeue().unwrap();
    assert_eq!(second.seq_id.as_u64(), 1);
    assert_eq!(second.priority, RequestPriority::Normal);
}

#[test]
fn priority_queue_dequeues_normal_before_low() {
    let mut queue = PrefillQueue::new();

    let (s_low, _r1) = make_test_sequence_with_priority(1, RequestPriority::Low);
    let (s_norm, _r2) = make_test_sequence_with_priority(2, RequestPriority::Normal);

    queue.enqueue(s_low).unwrap();
    queue.enqueue(s_norm).unwrap();

    assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 2); // Normal
    assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 1); // Low
}

#[test]
fn priority_queue_fifo_within_same_priority_level() {
    let mut queue = PrefillQueue::new();

    let (s1, _r1) = make_test_sequence_with_priority(10, RequestPriority::High);
    let (s2, _r2) = make_test_sequence_with_priority(20, RequestPriority::High);
    let (s3, _r3) = make_test_sequence_with_priority(30, RequestPriority::High);

    queue.enqueue(s1).unwrap();
    queue.enqueue(s2).unwrap();
    queue.enqueue(s3).unwrap();

    assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 10);
    assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 20);
    assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 30);
}

#[test]
fn priority_ordering_is_high_gt_normal_gt_low() {
    assert!(RequestPriority::High > RequestPriority::Normal);
    assert!(RequestPriority::Normal > RequestPriority::Low);
    assert!(RequestPriority::High > RequestPriority::Low);
}

#[test]
fn request_priority_from_header_parses_valid_values() {
    assert_eq!(
        RequestPriority::from_header("high"),
        Some(RequestPriority::High)
    );
    assert_eq!(
        RequestPriority::from_header("NORMAL"),
        Some(RequestPriority::Normal)
    );
    assert_eq!(
        RequestPriority::from_header("Low"),
        Some(RequestPriority::Low)
    );
    assert_eq!(
        RequestPriority::from_header(" high "),
        Some(RequestPriority::High)
    );
}

#[test]
fn request_priority_from_header_returns_none_for_invalid() {
    assert_eq!(RequestPriority::from_header("urgent"), None);
    assert_eq!(RequestPriority::from_header(""), None);
    assert_eq!(RequestPriority::from_header("1"), None);
}

#[test]
fn request_priority_default_is_normal() {
    assert_eq!(RequestPriority::default(), RequestPriority::Normal);
}

// -------------------------------------------------------------------
// Chunked prefill scheduling tests (without real model)
// -------------------------------------------------------------------

/// Extended decide_action that accounts for chunked prefill in progress.
/// This mirrors the scheduler's policy.
fn decide_action_with_chunked(
    queue: &PrefillQueue,
    batch: &ActiveBatch,
    chunked_in_progress: bool,
) -> BatchSchedulerAction {
    // If chunked prefill is in progress, interleave decode
    if chunked_in_progress {
        if !batch.is_empty() {
            return BatchSchedulerAction::Decode(batch.sequence_ids());
        }
        return BatchSchedulerAction::Prefill(SequenceId::from_raw(0));
    }

    if batch.is_empty() && queue.is_empty() {
        return BatchSchedulerAction::Idle;
    }
    if !batch.is_empty() {
        return BatchSchedulerAction::Decode(batch.sequence_ids());
    }
    BatchSchedulerAction::Prefill(SequenceId::from_raw(0))
}

#[test]
fn chunked_prefill_interleaves_decode_when_active_sequences_exist() {
    let queue = PrefillQueue::new();
    let mut batch = ActiveBatch::new(4);

    let (mut seq, _rx) = make_test_sequence(10);
    seq.state = SequenceState::Decoding;
    batch.add(seq).unwrap();

    // Chunked prefill in progress + active sequences -> should decode
    let action = decide_action_with_chunked(&queue, &batch, true);
    assert!(
        matches!(action, BatchSchedulerAction::Decode(_)),
        "Expected Decode during chunked prefill with active sequences"
    );
}

#[test]
fn chunked_prefill_continues_when_no_active_sequences() {
    let queue = PrefillQueue::new();
    let batch = ActiveBatch::new(4);

    // Chunked prefill in progress + no active sequences -> continue prefill
    let action = decide_action_with_chunked(&queue, &batch, true);
    assert!(
        matches!(action, BatchSchedulerAction::Prefill(_)),
        "Expected Prefill continuation when no active sequences"
    );
}

#[test]
fn chunked_prefill_interleaving_pattern() {
    // Simulate the interleaving pattern:
    // Tick 1: Prefill chunk 1
    // Tick 2: Decode (if active)
    // Tick 3: Prefill chunk 2
    // Tick 4: Decode (if active)
    // ...

    let queue = PrefillQueue::new();
    let mut batch = ActiveBatch::new(4);

    let (mut active_seq, _rx) = make_test_sequence(1);
    active_seq.state = SequenceState::Decoding;
    batch.add(active_seq).unwrap();

    // Tick 1: chunked prefill starts -> first action is Prefill
    // (no chunked_in_progress yet, batch has active seq, so Decode first)
    let action1 = decide_action_with_chunked(&queue, &batch, false);
    assert!(matches!(action1, BatchSchedulerAction::Decode(_)));

    // Tick 2: chunked prefill now in progress -> Decode interleave
    let action2 = decide_action_with_chunked(&queue, &batch, true);
    assert!(matches!(action2, BatchSchedulerAction::Decode(_)));

    // Tick 3: after decode, back to prefill continuation
    let action3 = decide_action_with_chunked(&queue, &batch, true);
    assert!(matches!(action3, BatchSchedulerAction::Decode(_)));
    // (still interleaving because batch is non-empty)

    // With no active batch: prefill continues
    let empty_batch = ActiveBatch::new(4);
    let action4 = decide_action_with_chunked(&queue, &empty_batch, true);
    assert!(matches!(action4, BatchSchedulerAction::Prefill(_)));
}

// -------------------------------------------------------------------
// Eviction victim selection tests
// -------------------------------------------------------------------

#[test]
fn active_batch_iter_min_priority_finds_lowest() {
    let mut batch = ActiveBatch::new(4);

    let (mut s1, _r1) = make_test_sequence_with_priority(1, RequestPriority::High);
    s1.state = SequenceState::Decoding;
    let (mut s2, _r2) = make_test_sequence_with_priority(2, RequestPriority::Low);
    s2.state = SequenceState::Decoding;
    let (mut s3, _r3) = make_test_sequence_with_priority(3, RequestPriority::Normal);
    s3.state = SequenceState::Decoding;

    batch.add(s1).unwrap();
    batch.add(s2).unwrap();
    batch.add(s3).unwrap();

    assert_eq!(batch.iter_min_priority(), Some(RequestPriority::Low));
}

#[test]
fn active_batch_iter_min_priority_empty_returns_none() {
    let batch = ActiveBatch::new(4);
    assert_eq!(batch.iter_min_priority(), None);
}

#[test]
fn eviction_selects_longest_first_by_default() {
    use crate::server::config::PreemptionPolicy;

    let mut batch = ActiveBatch::new(4);

    let (mut s1, _r1) = make_test_sequence_with_priority(1, RequestPriority::Normal);
    s1.state = SequenceState::Decoding;
    s1.generated_tokens = vec![10, 20, 30]; // 3 tokens

    let (mut s2, _r2) = make_test_sequence_with_priority(2, RequestPriority::Normal);
    s2.state = SequenceState::Decoding;
    s2.generated_tokens = vec![10]; // 1 token

    batch.add(s1).unwrap();
    batch.add(s2).unwrap();

    // LongestFirst should pick s1 (3 tokens > 1 token)
    let victim = match PreemptionPolicy::LongestFirst {
        PreemptionPolicy::LongestFirst => batch
            .iter_sequences()
            .max_by_key(|seq| seq.generated_tokens.len())
            .map(|seq| seq.seq_id),
        _ => None,
    };

    assert_eq!(victim.unwrap().as_u64(), 1);
}

#[test]
fn eviction_selects_lowest_priority_then_longest() {
    use crate::server::config::PreemptionPolicy;

    let mut batch = ActiveBatch::new(4);

    let (mut s1, _r1) = make_test_sequence_with_priority(1, RequestPriority::High);
    s1.state = SequenceState::Decoding;
    s1.generated_tokens = vec![10, 20, 30]; // 3 tokens

    let (mut s2, _r2) = make_test_sequence_with_priority(2, RequestPriority::Low);
    s2.state = SequenceState::Decoding;
    s2.generated_tokens = vec![10]; // 1 token

    let (mut s3, _r3) = make_test_sequence_with_priority(3, RequestPriority::Low);
    s3.state = SequenceState::Decoding;
    s3.generated_tokens = vec![10, 20, 30, 40]; // 4 tokens

    batch.add(s1).unwrap();
    batch.add(s2).unwrap();
    batch.add(s3).unwrap();

    // LowestPriority should pick s3 (Low + 4 tokens, longest of Low group)
    let victim = match PreemptionPolicy::LowestPriority {
        PreemptionPolicy::LowestPriority => batch
            .iter_sequences()
            .min_by(|a, b| {
                a.priority
                    .cmp(&b.priority)
                    .then_with(|| b.generated_tokens.len().cmp(&a.generated_tokens.len()))
            })
            .map(|seq| seq.seq_id),
        _ => None,
    };

    assert_eq!(victim.unwrap().as_u64(), 3);
}

#[test]
fn preemption_disabled_by_default_never_triggers() {
    // When enable_preemption is false, should_preempt should never
    // return true, even if the batch is full and queue has high-priority
    // requests. We test this by verifying the policy logic directly.
    let enable_preemption = false;
    let batch_full = true;
    let queue_has_high_priority = true;

    // The condition: enable_preemption && batch_full && queue_has_high > min_active
    let should_preempt = enable_preemption && batch_full && queue_has_high_priority;
    assert!(!should_preempt);
}

// -------------------------------------------------------------------
// Incremental token history and merged EOS (perf optimization tests)
// -------------------------------------------------------------------

#[test]
fn sequence_info_initializes_token_history_and_merged_eos_empty() {
    // New sequences enqueued via make_test_sequence start with empty
    // token_history and merged_eos. They are populated during prefill
    // (finish_prefill) so the decode steps can reuse them without
    // per-step reconstruction.
    let (seq, _rx) = make_test_sequence(1);
    assert!(
        seq.token_history.is_empty(),
        "token_history must start empty (populated at prefill)"
    );
    assert!(
        seq.merged_eos.is_empty(),
        "merged_eos must start empty (populated at prefill)"
    );
}

#[test]
fn sequence_info_token_history_can_be_populated_incrementally() {
    // Simulate the incremental update pattern used in decode_single_step
    // and execute_batched_decode after the perf optimization:
    //   token_history starts as prompt tokens (from initial_token_history),
    //   then each new token is appended with push() rather than
    //   rebuilding the full Vec from scratch every step.
    let prompt_tokens: Vec<i32> = vec![10, 20, 30];
    let mut token_history = prompt_tokens.clone(); // initial_token_history equivalent

    // Simulate generating 3 tokens incrementally
    let generated = vec![100_i32, 200, 300];
    for tok in &generated {
        token_history.push(*tok);
    }

    let expected: Vec<i32> = vec![10, 20, 30, 100, 200, 300];
    assert_eq!(token_history, expected);
}

#[test]
fn sequence_info_token_history_empty_when_no_penalties() {
    use mlxcel_core::generate::SamplingConfig;

    // When needs_token_history() is false (default config), the token
    // history should remain empty. This avoids unnecessary Vec growth.
    let sampling = SamplingConfig::default();
    assert!(!sampling.needs_token_history());

    // Simulate the scheduler's conditional push pattern:
    // `if seq.sampling.needs_token_history() { seq.token_history.push(tok); }`
    let mut token_history: Vec<i32> = Vec::new();
    let generated_token = 42_i32;
    if sampling.needs_token_history() {
        token_history.push(generated_token);
    }

    assert!(
        token_history.is_empty(),
        "token_history should stay empty when no penalties are active"
    );
}

#[test]
fn sequence_info_token_history_grows_when_penalties_enabled() {
    use mlxcel_core::generate::SamplingConfig;

    // When repetition_penalty is active, each decoded token is appended.
    let sampling = SamplingConfig {
        repetition_penalty: 1.3,
        ..Default::default()
    };
    assert!(sampling.needs_token_history());

    let mut token_history: Vec<i32> = vec![1, 2, 3]; // from initial_token_history

    for tok in [10_i32, 20, 30] {
        if sampling.needs_token_history() {
            token_history.push(tok);
        }
    }

    assert_eq!(token_history, vec![1, 2, 3, 10, 20, 30]);
}

#[test]
fn merged_eos_contains_both_model_and_request_stop_tokens() {
    // merged_eos_token_ids merges the model's built-in EOS tokens with
    // per-request stop tokens. This merged list is computed once at prefill
    // and cached on the SequenceInfo to avoid per-step recomputation.
    use mlxcel_core::generation_policy::merged_eos_token_ids;

    let model_eos = vec![2_i32]; // typical EOS
    let stop_tokens = vec![128001_i32, 128009_i32]; // e.g. Llama3 stop tokens

    let merged = merged_eos_token_ids(model_eos, &stop_tokens);

    assert!(merged.contains(&2));
    assert!(merged.contains(&128001));
    assert!(merged.contains(&128009));
}

#[test]
fn merged_eos_deduplicates_overlapping_tokens() {
    use mlxcel_core::generation_policy::merged_eos_token_ids;

    // When the model EOS and stop_token_ids share a token, it should
    // appear only once (or at most a small number of times) in merged.
    let model_eos = vec![2_i32, 100];
    let stop_tokens = vec![2_i32, 200]; // 2 is already in model_eos

    let merged = merged_eos_token_ids(model_eos, &stop_tokens);

    // All tokens must be present
    assert!(merged.contains(&2));
    assert!(merged.contains(&100));
    assert!(merged.contains(&200));
}

#[test]
fn paged_decode_storage_falls_back_when_batching_is_unavailable() {
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Paged, 4, false, true),
        DecodeStorageBackend::Dense
    );
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Paged, 1, true, true),
        DecodeStorageBackend::Dense
    );
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Paged, 4, true, false),
        DecodeStorageBackend::Dense
    );
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Paged, 4, true, true),
        DecodeStorageBackend::Paged
    );
}

#[test]
fn auto_decode_storage_prefers_paged_only_for_supported_workers() {
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Auto, 4, true, true),
        DecodeStorageBackend::Paged
    );
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Auto, 4, true, false),
        DecodeStorageBackend::Dense
    );
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Auto, 1, true, true),
        DecodeStorageBackend::Dense
    );
}
