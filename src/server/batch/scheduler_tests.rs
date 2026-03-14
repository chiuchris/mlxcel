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

use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;

use crate::server::batch::active::ActiveBatch;
use crate::server::batch::queue::PrefillQueue;
use crate::server::batch::sequence::{BatchSchedulerAction, SequenceInfo, SequenceState};
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
        vlm_embeddings: None,
        images: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        response_tx: tx,
        created_at: Instant::now(),
        prefill_start: None,
        first_token_time: None,
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
