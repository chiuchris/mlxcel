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

//! Shared LRU store for detached KV caches.
//!
//! This module provides [`PromptCacheStore`], the cross-request
//! prompt-prefix cache. The store is thread-safe via a single
//! `Arc<RwLock<Inner>>`: concurrent lookups take a read lock and match
//! prefixes, while inserts/evictions take an exclusive write lock.
//!
//! Sub-issue #420 will replace the current linear scan inside
//! [`PromptCacheStore::lookup_longest_prefix`] with a radix trie. The
//! public API is stable and tested against that eventual replacement.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use thiserror::Error;

use super::entry::CacheEntry;
use super::key::{PromptCacheKey, PromptCacheKeyDigest};
use super::metrics::{NoopPromptCacheMetrics, PromptCacheMetrics};
use super::policy::{PromptCacheConfig, PromptCacheStats};

/// Insert-time failure mode.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum InsertError {
    /// The feature is disabled (either via config or via store construction).
    #[error("prompt cache is disabled")]
    Disabled,
    /// The token prefix being inserted is shorter than
    /// [`PromptCacheConfig::min_prefix_tokens`].
    #[error("prompt cache: prefix is too short ({got} < {min_required})")]
    PrefixTooShort { got: usize, min_required: usize },
    /// The single entry exceeds the store's configured byte budget on its
    /// own, so no amount of eviction could make room for it.
    #[error(
        "prompt cache: entry size {entry_bytes} exceeds capacity {capacity_bytes} (cannot fit even alone)"
    )]
    OversizedEntry {
        entry_bytes: usize,
        capacity_bytes: usize,
    },
}

/// Composition key (model/lora/template/session) kept alongside the digest
/// so lookups can disambiguate partial prefix collisions.
///
/// The key identifies a *bucket* — a set of entries that share the same
/// model/lora/template/session and can therefore share a KV-cache prefix.
/// Token prefixes distinguish entries *within* a bucket.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BucketKey {
    pub model_id: String,
    pub lora_id: Option<String>,
    pub template_sig: String,
    pub session_key: Option<String>,
}

impl BucketKey {
    /// Build a bucket key from a [`PromptCacheKey`] (drops the token prefix
    /// since it does not participate in bucket identity).
    pub fn from_key(key: &PromptCacheKey<'_>) -> Self {
        Self {
            model_id: key.model_id.to_string(),
            lora_id: key.lora_id.map(str::to_string),
            template_sig: key.template_sig.to_string(),
            session_key: key.session_key.map(str::to_string),
        }
    }
}

/// Internal entry bookkeeping.
struct EntrySlot {
    entry: Arc<CacheEntry>,
    /// Bucket identity for prefix-matching fallback paths. The digest is
    /// recoverable via the HashMap key, so we only keep the bucket key here.
    bucket: BucketKey,
}

struct Inner {
    config: PromptCacheConfig,
    // Primary map: digest -> entry.
    entries: HashMap<PromptCacheKeyDigest, EntrySlot>,
    // Secondary map: bucket -> list of digests, so `lookup_longest_prefix`
    // can efficiently enumerate candidate entries that share the bucket.
    //
    // TODO(#420): radix trie — replace this `Vec<Digest>` bucket index with
    // a prefix-tree per bucket to avoid scanning every entry's token vector
    // on lookup.
    buckets: HashMap<BucketKey, Vec<PromptCacheKeyDigest>>,
    total_bytes: usize,
    inserts: u64,
    rejections_oversized: u64,
    lookups: u64,
    hits: u64,
    evictions_lru: u64,
    evictions_ttl: u64,
}

impl Inner {
    fn new(config: PromptCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            buckets: HashMap::new(),
            total_bytes: 0,
            inserts: 0,
            rejections_oversized: 0,
            lookups: 0,
            hits: 0,
            evictions_lru: 0,
            evictions_ttl: 0,
        }
    }

    fn stats(&self) -> PromptCacheStats {
        PromptCacheStats {
            entries: self.entries.len(),
            bytes: self.total_bytes,
            inserts: self.inserts,
            rejections_oversized: self.rejections_oversized,
            lookups: self.lookups,
            hits: self.hits,
            evictions_lru: self.evictions_lru,
            evictions_ttl: self.evictions_ttl,
        }
    }

    fn remove_entry(
        &mut self,
        digest: &PromptCacheKeyDigest,
    ) -> Option<(BucketKey, Arc<CacheEntry>)> {
        let slot = self.entries.remove(digest)?;
        self.total_bytes = self.total_bytes.saturating_sub(slot.entry.size_bytes);
        if let Some(list) = self.buckets.get_mut(&slot.bucket) {
            list.retain(|d| d != digest);
            if list.is_empty() {
                self.buckets.remove(&slot.bucket);
            }
        }
        Some((slot.bucket, slot.entry))
    }

    /// Sweep every entry that has been idle for longer than `config.ttl`.
    /// Returns `(bytes_freed, evicted_count)`.
    fn sweep_ttl(&mut self, now: Instant) -> (usize, usize) {
        if self.config.ttl.is_zero() || self.entries.is_empty() {
            return (0, 0);
        }
        let ttl = self.config.ttl;
        let stale: Vec<PromptCacheKeyDigest> = self
            .entries
            .iter()
            .filter(|(_, slot)| now.duration_since(slot.entry.last_used()) >= ttl)
            .map(|(d, _)| *d)
            .collect();
        let mut bytes = 0;
        for digest in &stale {
            if let Some((_, entry)) = self.remove_entry(digest) {
                bytes += entry.size_bytes;
            }
        }
        let count = stale.len();
        self.evictions_ttl = self.evictions_ttl.saturating_add(count as u64);
        (bytes, count)
    }

    /// Evict the single oldest entry. Returns the number of bytes freed, or
    /// `0` if the store is empty.
    fn evict_oldest(&mut self) -> usize {
        let oldest = self
            .entries
            .iter()
            .min_by_key(|(_, slot)| slot.entry.last_used())
            .map(|(d, _)| *d);
        match oldest {
            Some(digest) => {
                let freed = self
                    .remove_entry(&digest)
                    .map(|(_, e)| e.size_bytes)
                    .unwrap_or(0);
                if freed > 0 {
                    self.evictions_lru = self.evictions_lru.saturating_add(1);
                }
                freed
            }
            None => 0,
        }
    }

    /// Enforce both caps: max_entries, then capacity_bytes. Returns the
    /// number of bytes freed.
    fn enforce_caps(&mut self, metrics: &dyn PromptCacheMetrics) -> usize {
        let mut freed = 0;
        while self.entries.len() > self.config.max_entries {
            let n = self.evict_oldest();
            if n == 0 {
                break;
            }
            metrics.record_evict_lru(n);
            freed += n;
        }
        while self.total_bytes > self.config.capacity_bytes {
            let n = self.evict_oldest();
            if n == 0 {
                break;
            }
            metrics.record_evict_lru(n);
            freed += n;
        }
        freed
    }
}

/// Shared LRU store for detached KV caches.
///
/// Construct once via [`PromptCacheStore::new`] / [`PromptCacheStore::with_config`]
/// and share via `Arc<PromptCacheStore>`. All methods take `&self`; internal
/// mutation goes through an `RwLock`.
pub struct PromptCacheStore {
    inner: RwLock<Inner>,
    metrics: Arc<dyn PromptCacheMetrics>,
}

impl PromptCacheStore {
    /// Build a store with the default configuration.
    pub fn new() -> Self {
        Self::with_config(PromptCacheConfig::default())
    }

    /// Build a store with a caller-supplied configuration.
    pub fn with_config(config: PromptCacheConfig) -> Self {
        Self {
            inner: RwLock::new(Inner::new(config)),
            metrics: Arc::new(NoopPromptCacheMetrics),
        }
    }

    /// Build a store with a caller-supplied configuration and metrics
    /// implementor. Sub-issue #423 uses this entry point to hand in the
    /// Prometheus / `BatchMetrics` bridge.
    pub fn with_metrics(config: PromptCacheConfig, metrics: Arc<dyn PromptCacheMetrics>) -> Self {
        Self {
            inner: RwLock::new(Inner::new(config)),
            metrics,
        }
    }

    /// Convenience: wrap in an `Arc` for sharing across threads and
    /// subsystems.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Whether the store will accept inserts. When `false`, `insert` short
    /// circuits to [`InsertError::Disabled`] and lookups immediately return
    /// `None`.
    pub fn is_enabled(&self) -> bool {
        self.inner
            .read()
            .map(|g| g.config.is_enabled())
            .unwrap_or(false)
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.inner.read().map(|g| g.entries.len()).unwrap_or(0)
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Current cumulative byte footprint of all entries.
    pub fn bytes(&self) -> usize {
        self.inner.read().map(|g| g.total_bytes).unwrap_or(0)
    }

    /// Capacity in bytes as configured. Does not change at runtime.
    pub fn capacity_bytes(&self) -> usize {
        self.inner
            .read()
            .map(|g| g.config.capacity_bytes)
            .unwrap_or(0)
    }

    /// Maximum entry count as configured.
    pub fn max_entries(&self) -> usize {
        self.inner.read().map(|g| g.config.max_entries).unwrap_or(0)
    }

    /// Snapshot of the store's internal counters. Safe to call concurrently
    /// with inserts/lookups; values are captured under the read lock.
    pub fn stats(&self) -> PromptCacheStats {
        self.inner.read().map(|g| g.stats()).unwrap_or_default()
    }

    /// Insert an entry. Evicts older entries as needed to satisfy the
    /// entry-count and byte-budget caps. Returns [`InsertError`] if the
    /// store is disabled or if the single entry is too large to ever fit.
    pub fn insert(&self, key: &PromptCacheKey<'_>, entry: CacheEntry) -> Result<(), InsertError> {
        let digest = key.digest();
        let entry_bytes = entry.size_bytes;
        let bucket = BucketKey::from_key(key);

        let mut guard = self.inner.write().expect("prompt cache inner lock");

        if !guard.config.is_enabled() {
            return Err(InsertError::Disabled);
        }
        if key.effective_prefix_len() < guard.config.min_prefix_tokens {
            return Err(InsertError::PrefixTooShort {
                got: key.effective_prefix_len(),
                min_required: guard.config.min_prefix_tokens,
            });
        }
        if entry_bytes > guard.config.capacity_bytes {
            guard.rejections_oversized = guard.rejections_oversized.saturating_add(1);
            let metrics = Arc::clone(&self.metrics);
            drop(guard);
            metrics.record_reject_oversized(entry_bytes);
            return Err(InsertError::OversizedEntry {
                entry_bytes,
                capacity_bytes: self
                    .inner
                    .read()
                    .map(|g| g.config.capacity_bytes)
                    .unwrap_or(0),
            });
        }

        // Replace an existing entry under the same digest (idempotent insert
        // semantics for repeated prefill of the same prompt).
        if let Some((_, _)) = guard.remove_entry(&digest) {
            // Treat replacement as an LRU eviction for accounting purposes.
            guard.evictions_lru = guard.evictions_lru.saturating_add(1);
        }

        // Speculatively account for the new bytes, then evict as needed.
        guard.total_bytes = guard.total_bytes.saturating_add(entry_bytes);
        let slot = EntrySlot {
            entry: Arc::new(entry),
            bucket: bucket.clone(),
        };
        guard.entries.insert(digest, slot);
        guard.buckets.entry(bucket).or_default().push(digest);
        guard.inserts = guard.inserts.saturating_add(1);

        let metrics = Arc::clone(&self.metrics);
        metrics.record_insert(entry_bytes);

        // Enforce caps. This may evict freshly inserted entries if they are
        // already beyond capacity, which is intentional — we never exceed
        // the configured budget.
        guard.enforce_caps(metrics.as_ref());
        Ok(())
    }

    /// Find the cache entry in the same bucket as `key` whose tokens form
    /// the longest prefix of `tokens`. Returns the entry and the matched
    /// length. `min_prefix_tokens` still applies: matches shorter than the
    /// configured minimum are ignored.
    ///
    /// TODO(#420): radix trie — this currently scans every digest in the
    /// bucket and compares token vectors element-by-element. A radix trie
    /// per bucket will reduce the cost to `O(prefix_len)`.
    pub fn lookup_longest_prefix(
        &self,
        key: &PromptCacheKey<'_>,
        tokens: &[i32],
    ) -> Option<(Arc<CacheEntry>, usize)> {
        // Fast path: TTL sweep under a read-then-upgrade pattern would need
        // the write lock anyway. Do the sweep under the write lock so we
        // never hand out expired entries.
        {
            let now = Instant::now();
            let mut guard = self.inner.write().expect("prompt cache inner lock");
            if !guard.config.is_enabled() {
                return None;
            }
            let (freed, count) = guard.sweep_ttl(now);
            if let Some(per_entry) = freed.checked_div(count) {
                let metrics = Arc::clone(&self.metrics);
                for _ in 0..count {
                    metrics.record_evict_ttl(per_entry);
                }
            }
        }

        let bucket = BucketKey::from_key(key);
        let guard = self.inner.read().expect("prompt cache inner lock");

        let mut best: Option<(PromptCacheKeyDigest, usize)> = None;
        if let Some(digests) = guard.buckets.get(&bucket) {
            for digest in digests {
                if let Some(slot) = guard.entries.get(digest) {
                    if !slot.entry.has_detached() {
                        continue;
                    }
                    let matched = common_prefix_len(&slot.entry.tokens, tokens);
                    if matched < guard.config.min_prefix_tokens {
                        continue;
                    }
                    match best {
                        Some((_, b)) if b >= matched => {}
                        _ => best = Some((*digest, matched)),
                    }
                }
            }
        }

        drop(guard);

        // Increment lookup counters under the write lock so statistics stay
        // accurate even under concurrent readers. The hot path is the miss
        // case, which only writes a single atomic.
        let (entry, matched_len) = {
            let mut guard = self.inner.write().expect("prompt cache inner lock");
            guard.lookups = guard.lookups.saturating_add(1);
            match best {
                Some((digest, matched)) => {
                    guard.hits = guard.hits.saturating_add(1);
                    let slot = match guard.entries.get(&digest) {
                        Some(s) => s,
                        None => {
                            drop(guard);
                            let metrics = Arc::clone(&self.metrics);
                            metrics.record_lookup(false, 0);
                            return None;
                        }
                    };
                    slot.entry.touch();
                    let entry = Arc::clone(&slot.entry);
                    (Some(entry), matched)
                }
                None => (None, 0),
            }
        };

        let metrics = Arc::clone(&self.metrics);
        match &entry {
            Some(_) => metrics.record_lookup(true, matched_len),
            None => metrics.record_lookup(false, 0),
        }
        entry.map(|e| (e, matched_len))
    }

    /// Force a sweep. Returns the total bytes freed.
    pub fn evict_if_needed(&self) -> usize {
        let mut guard = self.inner.write().expect("prompt cache inner lock");
        if !guard.config.is_enabled() {
            return 0;
        }
        let now = Instant::now();
        let (ttl_freed, ttl_count) = guard.sweep_ttl(now);
        if let Some(per_entry) = ttl_freed.checked_div(ttl_count) {
            let metrics = Arc::clone(&self.metrics);
            for _ in 0..ttl_count {
                metrics.record_evict_ttl(per_entry);
            }
        }
        let metrics = Arc::clone(&self.metrics);
        let cap_freed = guard.enforce_caps(metrics.as_ref());
        ttl_freed + cap_freed
    }

    /// Drop every entry. Primarily for tests and shutdown paths.
    pub fn clear(&self) {
        let mut guard = self.inner.write().expect("prompt cache inner lock");
        guard.entries.clear();
        guard.buckets.clear();
        guard.total_bytes = 0;
    }
}

impl Default for PromptCacheStore {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PromptCacheStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("PromptCacheStore")
            .field("stats", &stats)
            .finish()
    }
}

/// Length of the common prefix of two token slices.
fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
