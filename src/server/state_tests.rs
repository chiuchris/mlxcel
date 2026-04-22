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

use std::sync::atomic::Ordering;

use super::{BatchMetrics, Metrics};

#[test]
fn metrics_record_request_accumulates_counters() {
    let metrics = Metrics::new();

    metrics.record_request(10, 3, 120);
    metrics.record_request(4, 2, 30);

    assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 2);
    assert_eq!(metrics.prompt_tokens_total.load(Ordering::Relaxed), 14);
    assert_eq!(metrics.completion_tokens_total.load(Ordering::Relaxed), 5);
    assert_eq!(
        metrics.generation_time_ms_total.load(Ordering::Relaxed),
        150
    );
}

// ------------------------------------------------------------------
// BatchMetrics
// ------------------------------------------------------------------

#[test]
fn batch_metrics_initializes_to_zero() {
    let m = BatchMetrics::new();
    assert_eq!(m.active_count(), 0);
    assert_eq!(m.queue_depth(), 0);
    assert_eq!(m.total_sequences_processed.load(Ordering::Relaxed), 0);
    assert_eq!(m.total_tokens_generated.load(Ordering::Relaxed), 0);
}

#[test]
fn batch_metrics_set_active_count_is_reflected_by_getter() {
    let m = BatchMetrics::new();
    m.set_active_count(4);
    assert_eq!(m.active_count(), 4);
    m.set_active_count(0);
    assert_eq!(m.active_count(), 0);
}

#[test]
fn batch_metrics_set_queue_depth_is_reflected_by_getter() {
    let m = BatchMetrics::new();
    m.set_queue_depth(7);
    assert_eq!(m.queue_depth(), 7);
    m.set_queue_depth(0);
    assert_eq!(m.queue_depth(), 0);
}

#[test]
fn batch_metrics_record_sequence_completed_accumulates() {
    let m = BatchMetrics::new();
    m.record_sequence_completed(10);
    m.record_sequence_completed(25);
    assert_eq!(m.total_sequences_processed.load(Ordering::Relaxed), 2);
    assert_eq!(m.total_tokens_generated.load(Ordering::Relaxed), 35);
}

#[test]
fn batch_metrics_default_equals_new() {
    let a = BatchMetrics::new();
    let b = BatchMetrics::default();
    assert_eq!(a.active_count(), b.active_count());
    assert_eq!(a.queue_depth(), b.queue_depth());
}

// ------------------------------------------------------------------
// can_accept_request logic (tested via BatchMetrics + ServerConfig)
// ------------------------------------------------------------------

/// The admission-control predicate is:
///   queue_depth < max_queue_depth
/// Test this boundary without constructing a full AppState.
#[test]
fn admission_control_accepts_when_queue_below_limit() {
    let m = BatchMetrics::new();
    let max = 4usize;

    m.set_queue_depth(0);
    assert!(m.queue_depth() < max, "empty queue should accept");

    m.set_queue_depth(3);
    assert!(m.queue_depth() < max, "queue below limit should accept");
}

#[test]
fn admission_control_rejects_when_queue_at_or_above_limit() {
    let m = BatchMetrics::new();
    let max = 4usize;

    m.set_queue_depth(4);
    assert!((m.queue_depth() >= max), "queue at limit should reject");

    m.set_queue_depth(10);
    assert!((m.queue_depth() >= max), "queue above limit should reject");
}

// ------------------------------------------------------------------
// Prompt-prefix cache integration (issue #419)
// ------------------------------------------------------------------

/// `PromptCacheConfig` is wired onto `ServerConfig` with a sensible default
/// (enabled, 2 GiB budget, 1024 entries, 1 h TTL, min-prefix 32 tokens).
#[test]
fn server_config_carries_prompt_cache_defaults() {
    use super::super::prompt_cache::PromptCacheConfig;
    let cfg = super::super::ServerConfig::default();
    assert!(cfg.prompt_cache.enabled);
    assert_eq!(
        cfg.prompt_cache.capacity_bytes,
        PromptCacheConfig::DEFAULT_CAPACITY_BYTES
    );
    assert_eq!(
        cfg.prompt_cache.max_entries,
        PromptCacheConfig::DEFAULT_MAX_ENTRIES
    );
    assert_eq!(
        cfg.prompt_cache.ttl.as_secs(),
        PromptCacheConfig::DEFAULT_TTL_SECONDS
    );
    assert_eq!(
        cfg.prompt_cache.min_prefix_tokens,
        PromptCacheConfig::DEFAULT_MIN_PREFIX_TOKENS
    );
}

/// A `PromptCacheStore` behaves as `Send + Sync`, which is what lets us
/// plumb it into both `AppState` and `ModelProvider` via `Arc`.
#[test]
fn prompt_cache_store_is_send_sync_via_arc() {
    use std::sync::Arc;
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Arc<super::super::prompt_cache::PromptCacheStore>>();
}
