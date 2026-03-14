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

//! FIFO queue for requests awaiting prefill.
//!
//! [`PrefillQueue`] is a simple `VecDeque`-backed queue that preserves
//! insertion order. The batch scheduler drains entries one at a time whenever
//! the active decode batch has capacity. An optional capacity limit prevents
//! unbounded memory growth under sustained load.

use std::collections::VecDeque;

use super::sequence::SequenceInfo;

/// Default maximum queue depth when no explicit limit is given.
const DEFAULT_MAX_QUEUE_SIZE: usize = 1024;

/// First-in, first-out queue of sequences waiting for prefill.
///
/// New requests are pushed to the back; the scheduler pops from the front.
/// This guarantees that older requests are served first, preventing starvation
/// under load.
///
/// An optional `max_size` bound prevents unbounded memory growth. When the
/// queue is full, `enqueue` returns the rejected `SequenceInfo` (boxed) so
/// the caller can send an appropriate error response.
pub struct PrefillQueue {
    queue: VecDeque<SequenceInfo>,
    max_size: usize,
}

impl PrefillQueue {
    /// Create an empty prefill queue with the default capacity limit (1024).
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            max_size: DEFAULT_MAX_QUEUE_SIZE,
        }
    }

    /// Create an empty prefill queue with a custom capacity limit.
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            max_size,
        }
    }

    /// Push a sequence to the back of the queue.
    ///
    /// Returns `Err(Box<seq>)` if the queue is at capacity, giving the caller
    /// ownership back so it can respond with a "server busy" error.
    pub fn enqueue(&mut self, seq: SequenceInfo) -> Result<(), Box<SequenceInfo>> {
        if self.queue.len() >= self.max_size {
            return Err(Box::new(seq));
        }
        self.queue.push_back(seq);
        Ok(())
    }

    /// Pop the oldest sequence from the front of the queue.
    pub fn dequeue(&mut self) -> Option<SequenceInfo> {
        self.queue.pop_front()
    }

    /// Number of sequences currently waiting.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` when the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns `true` when the queue has reached its capacity limit.
    pub fn is_full(&self) -> bool {
        self.queue.len() >= self.max_size
    }

    /// Maximum number of entries this queue will hold.
    pub fn max_size(&self) -> usize {
        self.max_size
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
            .field("len", &self.queue.len())
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
    use std::sync::mpsc;
    use std::time::Instant;

    /// Build a minimal `SequenceInfo` for testing.
    ///
    /// The `StreamingDecodeState` requires a tokenizer, but for queue-level
    /// tests we only need structural correctness, not actual decode behavior.
    /// We use a stub tokenizer that returns empty strings.
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
    fn fifo_ordering() {
        let mut queue = PrefillQueue::new();

        let (s1, _r1) = make_test_sequence(10);
        let (s2, _r2) = make_test_sequence(20);
        let (s3, _r3) = make_test_sequence(30);

        assert!(queue.enqueue(s1).is_ok());
        assert!(queue.enqueue(s2).is_ok());
        assert!(queue.enqueue(s3).is_ok());

        assert_eq!(queue.len(), 3);

        let d1 = queue.dequeue().unwrap();
        assert_eq!(d1.seq_id.as_u64(), 10);

        let d2 = queue.dequeue().unwrap();
        assert_eq!(d2.seq_id.as_u64(), 20);

        let d3 = queue.dequeue().unwrap();
        assert_eq!(d3.seq_id.as_u64(), 30);

        assert!(queue.is_empty());
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
    fn capacity_enforcement() {
        let mut queue = PrefillQueue::with_capacity(2);

        let (s1, _r1) = make_test_sequence(1);
        let (s2, _r2) = make_test_sequence(2);
        let (s3, _r3) = make_test_sequence(3);

        assert!(queue.enqueue(s1).is_ok());
        assert!(!queue.is_full());

        assert!(queue.enqueue(s2).is_ok());
        assert!(queue.is_full());

        // Third enqueue should fail and return the sequence back
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
}
