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

//! Batch observability metrics for continuous batching.
//!
//! [`BatchObservability`] extends the existing [`super::super::state::BatchMetrics`]
//! with detailed counters for prefill, decode, and cache utilization. All fields
//! are atomic for lock-free reads from HTTP handlers and writes from the single
//! scheduler thread.
//!
//! The scheduler updates these counters at key lifecycle points (prefill start,
//! decode step, sequence completion) and HTTP handlers read them from `/health`
//! and `/metrics` endpoints without locking.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use serde::Serialize;

/// Detailed observability counters for the batch scheduler.
///
/// These complement the coarser [`BatchMetrics`] already tracked in
/// `server/state.rs`. All operations use `Ordering::Relaxed` because
/// exactness is not required for monitoring -- slight staleness is
/// acceptable and avoids unnecessary memory barriers on the hot path.
pub struct BatchObservability {
    // -- Counters (monotonically increasing) --
    /// Total sequences that entered the prefill stage.
    pub sequences_started: AtomicU64,
    /// Total sequences that completed generation (stop or length).
    pub sequences_completed: AtomicU64,
    /// Cumulative prompt tokens processed across all prefills.
    pub total_prefill_tokens: AtomicU64,
    /// Cumulative decode tokens generated across all sequences.
    pub total_decode_tokens: AtomicU64,
    /// Number of prefill chunks processed (chunked prefill only).
    pub prefill_chunks_processed: AtomicU64,
    /// Number of decode steps executed (one per tick per batch).
    pub decode_steps_processed: AtomicU64,

    // -- Gauges (point-in-time values set by the scheduler) --
    /// Number of sequences currently in the active decode batch.
    pub current_batch_size: AtomicUsize,
    /// Number of requests waiting in the prefill queue.
    pub current_queue_depth: AtomicUsize,
    /// Number of active cache entries in the pool.
    pub cache_pool_active: AtomicUsize,
    /// Estimated memory bytes held by active KV caches.
    pub cache_pool_bytes: AtomicU64,
}

impl Default for BatchObservability {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchObservability {
    /// Create a zeroed observability instance.
    pub fn new() -> Self {
        Self {
            sequences_started: AtomicU64::new(0),
            sequences_completed: AtomicU64::new(0),
            total_prefill_tokens: AtomicU64::new(0),
            total_decode_tokens: AtomicU64::new(0),
            prefill_chunks_processed: AtomicU64::new(0),
            decode_steps_processed: AtomicU64::new(0),
            current_batch_size: AtomicUsize::new(0),
            current_queue_depth: AtomicUsize::new(0),
            cache_pool_active: AtomicUsize::new(0),
            cache_pool_bytes: AtomicU64::new(0),
        }
    }

    // -- Counter increments (called by the scheduler thread) --

    /// Record that a sequence has started prefill.
    pub fn record_prefill_start(&self, prompt_len: usize) {
        self.sequences_started.fetch_add(1, Ordering::Relaxed);
        self.total_prefill_tokens
            .fetch_add(prompt_len as u64, Ordering::Relaxed);
    }

    /// Record that a chunked prefill chunk was processed.
    pub fn record_prefill_chunk(&self) {
        self.prefill_chunks_processed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record that one decode step was executed for a batch of the given size.
    pub fn record_decode_step(&self, batch_size: usize) {
        self.decode_steps_processed.fetch_add(1, Ordering::Relaxed);
        self.total_decode_tokens
            .fetch_add(batch_size as u64, Ordering::Relaxed);
    }

    /// Record that a sequence has completed generation.
    pub fn record_sequence_completed(&self) {
        self.sequences_completed.fetch_add(1, Ordering::Relaxed);
    }

    // -- Gauge updates (called by the scheduler thread) --

    /// Update the point-in-time gauge values.
    pub fn update_gauges(
        &self,
        batch_size: usize,
        queue_depth: usize,
        cache_active: usize,
        cache_bytes: u64,
    ) {
        self.current_batch_size.store(batch_size, Ordering::Relaxed);
        self.current_queue_depth
            .store(queue_depth, Ordering::Relaxed);
        self.cache_pool_active
            .store(cache_active, Ordering::Relaxed);
        self.cache_pool_bytes.store(cache_bytes, Ordering::Relaxed);
    }

    // -- Snapshot for HTTP handlers --

    /// Create a serializable snapshot of the current observability state.
    pub fn snapshot(&self) -> ObservabilitySnapshot {
        ObservabilitySnapshot {
            sequences_started: self.sequences_started.load(Ordering::Relaxed),
            sequences_completed: self.sequences_completed.load(Ordering::Relaxed),
            total_prefill_tokens: self.total_prefill_tokens.load(Ordering::Relaxed),
            total_decode_tokens: self.total_decode_tokens.load(Ordering::Relaxed),
            prefill_chunks_processed: self.prefill_chunks_processed.load(Ordering::Relaxed),
            decode_steps_processed: self.decode_steps_processed.load(Ordering::Relaxed),
            current_batch_size: self.current_batch_size.load(Ordering::Relaxed),
            current_queue_depth: self.current_queue_depth.load(Ordering::Relaxed),
            cache_pool_active: self.cache_pool_active.load(Ordering::Relaxed),
            cache_pool_bytes: self.cache_pool_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Serializable point-in-time snapshot of batch observability metrics.
///
/// Returned by [`BatchObservability::snapshot()`] and included in the
/// `/health` endpoint response when the batch scheduler is active.
#[derive(Debug, Clone, Serialize)]
pub struct ObservabilitySnapshot {
    pub sequences_started: u64,
    pub sequences_completed: u64,
    pub total_prefill_tokens: u64,
    pub total_decode_tokens: u64,
    pub prefill_chunks_processed: u64,
    pub decode_steps_processed: u64,
    pub current_batch_size: usize,
    pub current_queue_depth: usize,
    pub cache_pool_active: usize,
    pub cache_pool_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_observability_starts_zeroed() {
        let obs = BatchObservability::new();
        let snap = obs.snapshot();
        assert_eq!(snap.sequences_started, 0);
        assert_eq!(snap.sequences_completed, 0);
        assert_eq!(snap.total_prefill_tokens, 0);
        assert_eq!(snap.total_decode_tokens, 0);
        assert_eq!(snap.prefill_chunks_processed, 0);
        assert_eq!(snap.decode_steps_processed, 0);
        assert_eq!(snap.current_batch_size, 0);
        assert_eq!(snap.current_queue_depth, 0);
        assert_eq!(snap.cache_pool_active, 0);
        assert_eq!(snap.cache_pool_bytes, 0);
    }

    #[test]
    fn record_prefill_increments_counters() {
        let obs = BatchObservability::new();
        obs.record_prefill_start(128);
        obs.record_prefill_start(256);
        let snap = obs.snapshot();
        assert_eq!(snap.sequences_started, 2);
        assert_eq!(snap.total_prefill_tokens, 384);
    }

    #[test]
    fn record_decode_step_increments_counters() {
        let obs = BatchObservability::new();
        obs.record_decode_step(4);
        obs.record_decode_step(3);
        let snap = obs.snapshot();
        assert_eq!(snap.decode_steps_processed, 2);
        assert_eq!(snap.total_decode_tokens, 7);
    }

    #[test]
    fn record_completion_increments_counter() {
        let obs = BatchObservability::new();
        obs.record_sequence_completed();
        obs.record_sequence_completed();
        obs.record_sequence_completed();
        assert_eq!(obs.snapshot().sequences_completed, 3);
    }

    #[test]
    fn update_gauges_sets_values() {
        let obs = BatchObservability::new();
        obs.update_gauges(4, 10, 8, 1024 * 1024);
        let snap = obs.snapshot();
        assert_eq!(snap.current_batch_size, 4);
        assert_eq!(snap.current_queue_depth, 10);
        assert_eq!(snap.cache_pool_active, 8);
        assert_eq!(snap.cache_pool_bytes, 1024 * 1024);
    }

    #[test]
    fn snapshot_is_serializable() {
        let obs = BatchObservability::new();
        obs.record_prefill_start(100);
        obs.record_decode_step(2);
        let snap = obs.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"sequences_started\":1"));
        assert!(json.contains("\"total_decode_tokens\":2"));
    }
}
