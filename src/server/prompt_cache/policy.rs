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

//! Eviction policy and configuration for the prompt prefix cache.
//!
//! The policy is intentionally small: LRU with a TTL escape hatch and both
//! byte-budget and entry-count caps. A production deployment can revisit
//! this layer without touching the store's public surface.

use std::fmt;
use std::time::Duration;

/// Runtime configuration for the prompt-prefix cache store.
///
/// CLI/env parsing for these fields is tracked in sub-issue #424. Until then
/// construct an instance via [`PromptCacheConfig::default`] or the explicit
/// constructor and hand it to [`super::PromptCacheStore::with_config`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PromptCacheConfig {
    /// Toggle the entire feature. When `false`, callers should skip
    /// constructing a [`super::PromptCacheStore`] entirely so no memory is
    /// reserved. The store itself also honors this value as an early-out.
    pub enabled: bool,
    /// Total byte budget across all entries. Inserts that would exceed this
    /// after eviction are rejected.
    pub capacity_bytes: usize,
    /// Upper bound on the number of live cache entries. Oldest entries are
    /// evicted first once this cap is hit.
    pub max_entries: usize,
    /// Time-to-live for an entry since its last successful lookup. Lazy
    /// sweep: entries are checked and possibly expired on lookup and on
    /// [`super::PromptCacheStore::evict_if_needed`].
    pub ttl: Duration,
    /// Minimum number of prompt tokens required before an entry is eligible
    /// to be inserted. Helps avoid polluting the cache with tiny prefixes
    /// that can't really amortize the detach/adopt overhead.
    pub min_prefix_tokens: usize,
}

impl PromptCacheConfig {
    /// Default capacity in bytes: 2 GiB.
    pub const DEFAULT_CAPACITY_BYTES: usize = 2 * 1024 * 1024 * 1024;
    /// Default maximum entry count.
    pub const DEFAULT_MAX_ENTRIES: usize = 1024;
    /// Default TTL: 3600 seconds.
    pub const DEFAULT_TTL_SECONDS: u64 = 3600;
    /// Default minimum prompt-prefix length before caching kicks in.
    pub const DEFAULT_MIN_PREFIX_TOKENS: usize = 32;

    /// Build a fully-specified config. Prefer [`PromptCacheConfig::default`]
    /// unless a caller has a reason to deviate.
    pub fn new(
        enabled: bool,
        capacity_bytes: usize,
        max_entries: usize,
        ttl: Duration,
        min_prefix_tokens: usize,
    ) -> Self {
        Self {
            enabled,
            capacity_bytes,
            max_entries,
            ttl,
            min_prefix_tokens,
        }
    }

    /// Config variant with the feature disabled. Safe to hand to the store
    /// constructor; the resulting store is a cheap no-op.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Whether the store should accept any insert at all.
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.capacity_bytes > 0 && self.max_entries > 0
    }
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            capacity_bytes: Self::DEFAULT_CAPACITY_BYTES,
            max_entries: Self::DEFAULT_MAX_ENTRIES,
            ttl: Duration::from_secs(Self::DEFAULT_TTL_SECONDS),
            min_prefix_tokens: Self::DEFAULT_MIN_PREFIX_TOKENS,
        }
    }
}

/// Aggregate store statistics, intended for both tests and the metrics
/// bridge tracked in sub-issue #423.
///
/// This is a pure snapshot: values are captured under the store's lock and
/// returned by value so callers can release the lock immediately.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PromptCacheStats {
    /// Number of live entries in the store.
    pub entries: usize,
    /// Sum of `size_bytes` across all live entries.
    pub bytes: usize,
    /// Lifetime count of successful inserts.
    pub inserts: u64,
    /// Lifetime count of inserts rejected because the single entry exceeded
    /// the configured byte budget.
    pub rejections_oversized: u64,
    /// Lifetime count of lookup calls.
    pub lookups: u64,
    /// Lifetime count of lookup calls that returned `Some(..)`.
    pub hits: u64,
    /// Lifetime count of entries evicted due to LRU pressure (entry-cap or
    /// byte-cap).
    pub evictions_lru: u64,
    /// Lifetime count of entries evicted because the TTL expired.
    pub evictions_ttl: u64,
}

impl fmt::Display for PromptCacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "entries={} bytes={} inserts={} hits={}/{} lru_evict={} ttl_evict={} reject_oversized={}",
            self.entries,
            self.bytes,
            self.inserts,
            self.hits,
            self.lookups,
            self.evictions_lru,
            self.evictions_ttl,
            self.rejections_oversized,
        )
    }
}
