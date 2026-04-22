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

//! Priority queue for requests awaiting prefill.
//!
//! [`PrefillQueue`] uses three priority lanes (high, normal, low) backed by
//! `VecDeque`s. Within each lane, insertion order is preserved (FIFO).
//! The scheduler drains entries starting from the highest-priority non-empty
//! lane. An optional capacity limit (across all lanes) prevents unbounded
//! memory growth under sustained load.

use std::collections::VecDeque;
use std::sync::atomic::Ordering;

use super::sequence::{RequestPriority, SequenceInfo};

/// Default maximum queue depth when no explicit limit is given.
const DEFAULT_MAX_QUEUE_SIZE: usize = 1024;

/// Priority-aware queue of sequences waiting for prefill.
///
/// Requests are routed into one of three lanes based on their
/// [`RequestPriority`]. The scheduler always dequeues from the
/// highest-priority non-empty lane first, ensuring that high-priority
/// requests are prefilled before lower-priority ones.
///
/// Within each lane, FIFO order is preserved so older requests of the
/// same priority are served first.
///
/// An optional `max_size` bound prevents unbounded memory growth. When the
/// queue is full, `enqueue` returns the rejected `SequenceInfo` (boxed) so
/// the caller can send an appropriate error response.
///
/// **Starvation note:** This queue uses strict priority ordering without
/// aging. Under sustained high-priority load, lower-priority requests
/// may wait indefinitely. Callers should apply server-level timeouts
/// to bound worst-case latency for low-priority requests.
pub struct PrefillQueue {
    high: VecDeque<SequenceInfo>,
    normal: VecDeque<SequenceInfo>,
    low: VecDeque<SequenceInfo>,
    max_size: usize,
}

impl PrefillQueue {
    /// Create an empty prefill queue with the default capacity limit (1024).
    pub fn new() -> Self {
        Self {
            high: VecDeque::new(),
            normal: VecDeque::new(),
            low: VecDeque::new(),
            max_size: DEFAULT_MAX_QUEUE_SIZE,
        }
    }

    /// Create an empty prefill queue with a custom capacity limit.
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            high: VecDeque::new(),
            normal: VecDeque::new(),
            low: VecDeque::new(),
            max_size,
        }
    }

    /// Push a sequence into the appropriate priority lane.
    ///
    /// Returns `Err(Box<seq>)` if the total queue (across all lanes) is at
    /// capacity, giving the caller ownership back so it can respond with a
    /// "server busy" error.
    pub fn enqueue(&mut self, seq: SequenceInfo) -> Result<(), Box<SequenceInfo>> {
        if self.len() >= self.max_size {
            return Err(Box::new(seq));
        }
        match seq.priority {
            RequestPriority::High => self.high.push_back(seq),
            RequestPriority::Normal => self.normal.push_back(seq),
            RequestPriority::Low => self.low.push_back(seq),
        }
        Ok(())
    }

    /// Pop the highest-priority, oldest sequence from the queue.
    ///
    /// Drains from the high lane first, then normal, then low.
    pub fn dequeue(&mut self) -> Option<SequenceInfo> {
        self.high
            .pop_front()
            .or_else(|| self.normal.pop_front())
            .or_else(|| self.low.pop_front())
    }

    /// Peek at the priority of the next sequence that would be dequeued.
    pub fn peek_priority(&self) -> Option<RequestPriority> {
        if !self.high.is_empty() {
            Some(RequestPriority::High)
        } else if !self.normal.is_empty() {
            Some(RequestPriority::Normal)
        } else if !self.low.is_empty() {
            Some(RequestPriority::Low)
        } else {
            None
        }
    }

    /// Total number of sequences across all priority lanes.
    pub fn len(&self) -> usize {
        self.high.len() + self.normal.len() + self.low.len()
    }

    /// Returns `true` when all lanes are empty.
    pub fn is_empty(&self) -> bool {
        self.high.is_empty() && self.normal.is_empty() && self.low.is_empty()
    }

    /// Returns `true` when the total queue has reached its capacity limit.
    pub fn is_full(&self) -> bool {
        self.len() >= self.max_size
    }

    /// Maximum number of entries this queue will hold (across all lanes).
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Remove and return all queued sequences whose cancellation flag is set.
    ///
    /// Scans all priority lanes and removes sequences where
    /// `cancelled.load(Relaxed)` is `true`. The caller is responsible for
    /// cleaning up cache pool entries and notifying the response channel.
    pub fn drain_cancelled(&mut self) -> Vec<SequenceInfo> {
        let mut cancelled = Vec::new();
        Self::drain_cancelled_from_lane(&mut self.high, &mut cancelled);
        Self::drain_cancelled_from_lane(&mut self.normal, &mut cancelled);
        Self::drain_cancelled_from_lane(&mut self.low, &mut cancelled);
        cancelled
    }

    /// Helper: remove cancelled entries from a single lane, preserving order.
    fn drain_cancelled_from_lane(lane: &mut VecDeque<SequenceInfo>, out: &mut Vec<SequenceInfo>) {
        let mut i = 0;
        while i < lane.len() {
            if lane[i].cancelled.load(Ordering::Relaxed) {
                // remove() returns Option but index is valid since i < len.
                if let Some(seq) = lane.remove(i) {
                    out.push(seq);
                }
                // Do not increment i because remove shifts elements left.
            } else {
                i += 1;
            }
        }
    }
}

impl Default for PrefillQueue {
    fn default() -> Self {
        Self::new()
    }
}

// We cannot derive Debug because SequenceInfo contains non-Debug fields.
impl std::fmt::Debug for PrefillQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillQueue")
            .field("high", &self.high.len())
            .field("normal", &self.normal.len())
            .field("low", &self.low.len())
            .field("max_size", &self.max_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::batch::sequence::SequenceState;
    use crate::server::model_provider::GenerateEvent;
    use crate::server::model_provider::model_worker::StreamingDecodeState;
    use mlxcel_core::cache::SequenceId;
    use mlxcel_core::generate::SamplingConfig;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;
    use std::time::Instant;

    /// Build a minimal `SequenceInfo` for testing with a given priority.
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
            prefill_start_offset: 0,
            already_cached_tokens: 0,
            response_tx: tx,
            cancelled: Arc::new(AtomicBool::new(false)),
            created_at: Instant::now(),
            prefill_start: None,
            first_token_time: None,
            token_history: Vec::new(),
            merged_eos: Vec::new(),
            thinking: crate::server::thinking_budget::ThinkingState::disabled(),
        };

        (seq, rx)
    }

    fn make_test_sequence(id_val: u64) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
        make_test_sequence_with_priority(id_val, RequestPriority::Normal)
    }

    #[test]
    fn new_queue_is_empty() {
        let queue = PrefillQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert!(!queue.is_full());
    }

    #[test]
    fn enqueue_increases_len() {
        let mut queue = PrefillQueue::new();
        let (seq, _rx) = make_test_sequence(0);
        assert!(queue.enqueue(seq).is_ok());
        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());
    }

    #[test]
    fn dequeue_returns_none_when_empty() {
        let mut queue = PrefillQueue::new();
        assert!(queue.dequeue().is_none());
    }

    #[test]
    fn fifo_ordering_within_same_priority() {
        let mut queue = PrefillQueue::new();

        let (s1, _r1) = make_test_sequence(10);
        let (s2, _r2) = make_test_sequence(20);
        let (s3, _r3) = make_test_sequence(30);

        assert!(queue.enqueue(s1).is_ok());
        assert!(queue.enqueue(s2).is_ok());
        assert!(queue.enqueue(s3).is_ok());

        assert_eq!(queue.len(), 3);

        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 10);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 20);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 30);
        assert!(queue.is_empty());
    }

    #[test]
    fn priority_ordering_high_before_normal_before_low() {
        let mut queue = PrefillQueue::new();

        // Enqueue in reverse priority order
        let (s_low, _r1) = make_test_sequence_with_priority(1, RequestPriority::Low);
        let (s_norm, _r2) = make_test_sequence_with_priority(2, RequestPriority::Normal);
        let (s_high, _r3) = make_test_sequence_with_priority(3, RequestPriority::High);

        assert!(queue.enqueue(s_low).is_ok());
        assert!(queue.enqueue(s_norm).is_ok());
        assert!(queue.enqueue(s_high).is_ok());

        // Should dequeue high first, then normal, then low
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 3); // High
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 2); // Normal
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 1); // Low
        assert!(queue.is_empty());
    }

    #[test]
    fn priority_with_fifo_within_lanes() {
        let mut queue = PrefillQueue::new();

        let (s_high1, _r1) = make_test_sequence_with_priority(10, RequestPriority::High);
        let (s_norm1, _r2) = make_test_sequence_with_priority(20, RequestPriority::Normal);
        let (s_high2, _r3) = make_test_sequence_with_priority(11, RequestPriority::High);
        let (s_norm2, _r4) = make_test_sequence_with_priority(21, RequestPriority::Normal);

        assert!(queue.enqueue(s_high1).is_ok());
        assert!(queue.enqueue(s_norm1).is_ok());
        assert!(queue.enqueue(s_high2).is_ok());
        assert!(queue.enqueue(s_norm2).is_ok());

        // High lane: 10, 11 (FIFO), then Normal lane: 20, 21 (FIFO)
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 10);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 11);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 20);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 21);
    }

    #[test]
    fn peek_priority_reflects_head() {
        let mut queue = PrefillQueue::new();
        assert!(queue.peek_priority().is_none());

        let (s_low, _r1) = make_test_sequence_with_priority(1, RequestPriority::Low);
        queue.enqueue(s_low).unwrap();
        assert_eq!(queue.peek_priority(), Some(RequestPriority::Low));

        let (s_high, _r2) = make_test_sequence_with_priority(2, RequestPriority::High);
        queue.enqueue(s_high).unwrap();
        assert_eq!(queue.peek_priority(), Some(RequestPriority::High));
    }

    #[test]
    fn interleaved_enqueue_dequeue() {
        let mut queue = PrefillQueue::new();

        let (s1, _r1) = make_test_sequence(1);
        assert!(queue.enqueue(s1).is_ok());

        let d1 = queue.dequeue().unwrap();
        assert_eq!(d1.seq_id.as_u64(), 1);

        let (s2, _r2) = make_test_sequence(2);
        let (s3, _r3) = make_test_sequence(3);
        assert!(queue.enqueue(s2).is_ok());
        assert!(queue.enqueue(s3).is_ok());

        assert_eq!(queue.len(), 2);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 2);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 3);
        assert!(queue.is_empty());
    }

    #[test]
    fn default_creates_empty_queue() {
        let queue = PrefillQueue::default();
        assert!(queue.is_empty());
        assert_eq!(queue.max_size(), DEFAULT_MAX_QUEUE_SIZE);
    }

    #[test]
    fn capacity_enforcement_across_lanes() {
        let mut queue = PrefillQueue::with_capacity(2);

        let (s1, _r1) = make_test_sequence_with_priority(1, RequestPriority::High);
        let (s2, _r2) = make_test_sequence_with_priority(2, RequestPriority::Low);
        let (s3, _r3) = make_test_sequence_with_priority(3, RequestPriority::Normal);

        assert!(queue.enqueue(s1).is_ok());
        assert!(!queue.is_full());

        assert!(queue.enqueue(s2).is_ok());
        assert!(queue.is_full());

        // Third enqueue should fail regardless of priority
        let rejected = queue.enqueue(s3);
        assert!(rejected.is_err());
        let returned_seq = rejected.unwrap_err();
        assert_eq!(returned_seq.seq_id.as_u64(), 3);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn capacity_reopens_after_dequeue() {
        let mut queue = PrefillQueue::with_capacity(1);

        let (s1, _r1) = make_test_sequence(1);
        assert!(queue.enqueue(s1).is_ok());
        assert!(queue.is_full());

        queue.dequeue();
        assert!(!queue.is_full());

        let (s2, _r2) = make_test_sequence(2);
        assert!(queue.enqueue(s2).is_ok());
    }

    #[test]
    fn drain_cancelled_removes_only_cancelled_sequences() {
        use std::sync::atomic::Ordering;

        let mut queue = PrefillQueue::new();

        let (s1, _r1) = make_test_sequence(1);
        let (s2, _r2) = make_test_sequence(2);
        let (s3, _r3) = make_test_sequence(3);

        // Mark s2 as cancelled before enqueueing
        s2.cancelled.store(true, Ordering::Relaxed);

        queue.enqueue(s1).unwrap();
        queue.enqueue(s2).unwrap();
        queue.enqueue(s3).unwrap();
        assert_eq!(queue.len(), 3);

        let drained = queue.drain_cancelled();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].seq_id.as_u64(), 2);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn drain_cancelled_returns_empty_when_none_cancelled() {
        let mut queue = PrefillQueue::new();

        let (s1, _r1) = make_test_sequence(10);
        let (s2, _r2) = make_test_sequence(20);
        queue.enqueue(s1).unwrap();
        queue.enqueue(s2).unwrap();

        let drained = queue.drain_cancelled();
        assert!(drained.is_empty());
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn drain_cancelled_removes_across_all_priority_lanes() {
        use std::sync::atomic::Ordering;

        let mut queue = PrefillQueue::new();

        let (s_high, _r1) = make_test_sequence_with_priority(1, RequestPriority::High);
        let (s_norm, _r2) = make_test_sequence_with_priority(2, RequestPriority::Normal);
        let (s_low, _r3) = make_test_sequence_with_priority(3, RequestPriority::Low);

        // Cancel the high and low priority sequences
        s_high.cancelled.store(true, Ordering::Relaxed);
        s_low.cancelled.store(true, Ordering::Relaxed);

        queue.enqueue(s_high).unwrap();
        queue.enqueue(s_norm).unwrap();
        queue.enqueue(s_low).unwrap();

        let drained = queue.drain_cancelled();
        assert_eq!(drained.len(), 2);
        let drained_ids: Vec<u64> = drained.iter().map(|s| s.seq_id.as_u64()).collect();
        assert!(drained_ids.contains(&1));
        assert!(drained_ids.contains(&3));

        // Only the normal priority sequence should remain
        assert_eq!(queue.len(), 1);
        let remaining = queue.dequeue().unwrap();
        assert_eq!(remaining.seq_id.as_u64(), 2);
    }
}
