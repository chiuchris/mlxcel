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

//! Cross-sequence KV cache reuse: trim / detach / adopt primitives.
//!
//! This module extends [`super::KVCache`] and [`super::CachePool`] with the
//! primitives required by the cross-request prompt prefix cache (epic #416,
//! sub-issue #417):
//!
//! * [`KVCache::trim_to`] — shrink the logical cache length to an exact value
//!   while keeping the pre-allocated backing buffer around.
//! * [`KVCache::clone_handle`] — move the underlying `MlxArray` ownership out
//!   into an inert [`DetachedKVCache`] that can outlive the original sequence.
//! * [`CachePool::detach`] — lift a whole [`super::SequenceCacheSet`] off the
//!   active HashMap and return it as an owned [`DetachedCacheSet`] without
//!   freeing the MLX buffers.
//! * [`CachePool::adopt`] — install a previously-detached cache set under a
//!   fresh [`super::SequenceId`] and prime the model's sidecar state for it.
//!
//! Only the **dense** KV cache backend (`SequenceStateBackend::DenseKvCache`)
//! is handled directly by this file. Paged sequences are handled by the
//! parallel API surface in [`super::paged_detach`] (sub-issue #418); this
//! module delegates through the shared `DetachedHandle` namespace so parking
//! remains a single pool-level abstraction.
//!
//! ## Memory accounting
//!
//! While a detached cache set is in-flight — e.g. inside a scheduler that has
//! taken it out of `CachePool` but is about to re-adopt it — callers can
//! [`park`] the set so that the pool's
//! [`CachePool::memory_usage_bytes`] keeps including the bytes. Parking is
//! optional; `detach` + `adopt` work end-to-end without it.
//!
//! ## INT8 preservation
//!
//! Both the INT8 key/value tensors and the per-token FP16 scale tensors are
//! moved through detach/adopt unchanged, so `KVCache::mode == Int8` sequences
//! round-trip losslessly.
//!
//! ## Aliasing with `MLXCEL_ENABLE_DIRECT_PREFILL_CACHE_STORE`
//!
//! The direct-prefill-store fast path in [`super::KVCache::update`] installs
//! the incoming FP16 tensor directly as the cache buffer (with a
//! `contiguous` call) when the cache is empty and the env var is set. Detach
//! simply moves that buffer out via [`UniquePtr::take`]; no aliasing survives
//! because `MlxArray` buffers are functional — every operation produces a
//! fresh array, and the move semantics of `UniquePtr` prevent concurrent
//! access. Adopting that same buffer into a new sequence is therefore safe.
//!
//! [`park`]: CachePool::park_detached

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cxx::UniquePtr;

use crate::ffi;
use crate::ffi::MlxArray;

use super::{CachePool, KVCache, KVCacheMode, SequenceCacheSet, SequenceId, SequenceStateBackend};

// ---------------------------------------------------------------------------
// DetachedKVCache
// ---------------------------------------------------------------------------

/// Inert, model-agnostic snapshot of a single [`KVCache`] that can outlive the
/// sequence which produced it.
///
/// `DetachedKVCache` owns the underlying MLX `UniquePtr<MlxArray>` buffers
/// directly, so detach/adopt never allocates a new tensor or copies data.
/// INT8-mode caches carry their per-token scale tensors alongside the INT8
/// key/value buffers so dequantization behavior is bit-identical after adopt.
pub struct DetachedKVCache {
    pub(super) keys: Option<UniquePtr<MlxArray>>,
    pub(super) values: Option<UniquePtr<MlxArray>>,
    pub(super) offset: i32,
    pub(super) step: i32,
    pub(super) mode: KVCacheMode,
    pub(super) key_scales: Option<UniquePtr<MlxArray>>,
    pub(super) val_scales: Option<UniquePtr<MlxArray>>,
}

impl DetachedKVCache {
    /// Logical length of the stored cache (matches the live
    /// [`KVCache::seq_len`] at detach time).
    pub fn seq_len(&self) -> i32 {
        self.offset
    }

    /// Quantization mode of the detached cache.
    pub fn mode(&self) -> KVCacheMode {
        self.mode
    }

    /// Total byte footprint of the detached tensors (keys + values + INT8
    /// scales when applicable).
    pub fn nbytes(&self) -> usize {
        let k = self.keys.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let v = self.values.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let ks = self.key_scales.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vs = self.val_scales.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        k + v + ks + vs
    }

    /// Whether the detached handle carries no data (all tensors were `None`).
    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }

    /// Read-only access to the detached keys tensor.
    pub fn keys(&self) -> Option<&MlxArray> {
        self.keys.as_deref()
    }

    /// Read-only access to the detached values tensor.
    pub fn values(&self) -> Option<&MlxArray> {
        self.values.as_deref()
    }
}

impl std::fmt::Debug for DetachedKVCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetachedKVCache")
            .field("offset", &self.offset)
            .field("step", &self.step)
            .field("mode", &self.mode)
            .field("has_keys", &self.keys.is_some())
            .field("has_values", &self.values.is_some())
            .field("has_key_scales", &self.key_scales.is_some())
            .field("has_val_scales", &self.val_scales.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// KVCache extensions
// ---------------------------------------------------------------------------

impl KVCache {
    /// Shrink the logical cache length to exactly `new_len`.
    ///
    /// Semantics:
    /// * `new_len == 0` fully rewinds the cache (equivalent to
    ///   `trim(self.offset)`) and drops all backing buffers.
    /// * `0 < new_len < self.offset` keeps the pre-allocated buffer but
    ///   re-slices its visible region to `new_len`. This mirrors
    ///   [`KVCache::trim`] but takes an absolute target instead of a delta.
    /// * `new_len == self.offset` is a no-op.
    /// * `new_len < 0` or `new_len > self.offset` returns `Err`.
    ///
    /// INT8 mode: the per-token scale buffers are trimmed in lock-step so
    /// subsequent `update_and_fetch` dequantization stays consistent.
    ///
    /// Used by: prompt prefix cache reuse (#417), speculative decode rewinds,
    /// server scheduler trim-to-exact-prefix paths.
    pub fn trim_to(&mut self, new_len: i32) -> Result<(), String> {
        if new_len < 0 {
            return Err(format!(
                "KVCache::trim_to: new_len must be non-negative, got {new_len}"
            ));
        }
        if new_len > self.offset {
            return Err(format!(
                "KVCache::trim_to: new_len ({new_len}) exceeds current offset ({})",
                self.offset
            ));
        }

        let delta = self.offset - new_len;
        if delta == 0 {
            return Ok(());
        }

        let trimmed = self.trim(delta);
        debug_assert_eq!(
            trimmed, delta,
            "KVCache::trim returned {trimmed} but trim_to computed delta {delta}"
        );
        Ok(())
    }

    /// Move the underlying MLX buffers out of this cache into a
    /// [`DetachedKVCache`] handle.
    ///
    /// After this call the source `KVCache` is empty (`is_empty() == true`,
    /// `offset == 0`) but retains its quantization mode and step size so it
    /// can be reused for a new sequence. The returned `DetachedKVCache`
    /// carries the original tensors unchanged — including INT8 scale buffers
    /// when `mode == Int8` — so adopt is a zero-copy operation.
    ///
    /// Used by: prompt prefix cache detach/adopt (#417), cross-request reuse
    /// handoff inside `CachePool::detach`.
    pub fn clone_handle(&mut self) -> DetachedKVCache {
        DetachedKVCache {
            keys: self.keys.take(),
            values: self.values.take(),
            offset: std::mem::replace(&mut self.offset, 0),
            step: self.step,
            mode: self.mode,
            key_scales: self.key_scales.take(),
            val_scales: self.val_scales.take(),
        }
    }

    /// Re-install a previously detached cache into this `KVCache` slot.
    ///
    /// This is the inverse of [`KVCache::clone_handle`]. The receiver must be
    /// empty (`is_empty() == true`) to guarantee no live buffer is silently
    /// dropped; callers that need to overwrite a populated cache should
    /// `trim_to(0)` first.
    ///
    /// Used by: `CachePool::adopt` when re-hydrating per-layer caches for a
    /// freshly allocated sequence id.
    pub fn install_detached(&mut self, detached: DetachedKVCache) -> Result<(), String> {
        if !self.is_empty() {
            return Err(
                "KVCache::install_detached: target cache is not empty; trim_to(0) first".into(),
            );
        }
        self.keys = detached.keys;
        self.values = detached.values;
        self.offset = detached.offset;
        self.step = detached.step;
        self.mode = detached.mode;
        self.key_scales = detached.key_scales;
        self.val_scales = detached.val_scales;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DetachedCacheSet
// ---------------------------------------------------------------------------

/// Inert snapshot of a whole sequence's per-layer KV caches.
///
/// Produced by [`CachePool::detach`] and consumed by [`CachePool::adopt`].
/// Only the dense backend is supported; paged sequences produce `None` on
/// detach per sub-issue #418.
pub struct DetachedCacheSet {
    /// Per-layer detached caches, one per model layer.
    pub caches: Vec<DetachedKVCache>,
    /// Logical backend tag (always `DenseKvCache` in this API surface).
    pub backend: SequenceStateBackend,
    /// Prompt length as recorded on the originating [`SequenceCacheSet`].
    pub prompt_len: usize,
    /// Last known decode offset at detach time.
    pub current_offset: i32,
    /// Timestamp of the originating allocation (preserved across handoffs).
    pub created_at: Instant,
    /// Timestamp of the most recent detach.
    pub detached_at: Instant,
    /// Original sequence id this cache set was last installed under (for
    /// logging / observability; the adopt path always assigns a fresh id).
    pub origin_seq_id: SequenceId,
}

impl DetachedCacheSet {
    /// Summed tensor bytes across all layer caches.
    pub fn nbytes(&self) -> usize {
        self.caches.iter().map(|c| c.nbytes()).sum()
    }

    /// Number of layer caches carried by this set.
    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }

    /// Logical token length of the first non-empty layer (or 0).
    ///
    /// All transformer layers share a common prefix length by construction,
    /// so the first layer's `offset` is a faithful summary of the set.
    pub fn seq_len(&self) -> i32 {
        self.caches.first().map(|c| c.offset).unwrap_or(0)
    }
}

impl std::fmt::Debug for DetachedCacheSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetachedCacheSet")
            .field("backend", &self.backend)
            .field("num_layers", &self.num_layers())
            .field("seq_len", &self.seq_len())
            .field("prompt_len", &self.prompt_len)
            .field("current_offset", &self.current_offset)
            .field("origin_seq_id", &self.origin_seq_id)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Parking (in-flight holding)
// ---------------------------------------------------------------------------

/// Opaque handle returned by [`CachePool::park_detached`].
///
/// Parking is an optional escape hatch: a scheduler can hand a
/// [`DetachedCacheSet`] back to the pool for the duration of a cross-request
/// lookup so that [`CachePool::memory_usage_bytes`] keeps accounting for the
/// tensors that the pool logically still holds in-flight. The normal
/// `detach` → store in external cache → `adopt` flow does not require
/// parking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DetachedHandle(u64);

impl DetachedHandle {
    /// Raw numeric representation of this handle, useful for logging and
    /// metric labels.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Construct a handle from a raw id. Provided for cross-module builders
    /// (e.g. the paged detach surface in [`super::paged_detach`]) that mint
    /// handles out of the same `CachePool::next_id` space.
    pub(super) fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for DetachedHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "detached-{}", self.0)
    }
}

/// Internal map of parked detached cache sets, keyed by handle. This map is
/// attached to [`CachePool`] as `detached: DetachedMap` via the
/// `detached` field declared in the parent module. The map stores a
/// [`super::paged_detach::ParkedCache`] enum so dense and paged variants
/// share the same handle namespace.
pub(super) type DetachedMap = HashMap<DetachedHandle, super::paged_detach::ParkedCache>;

// ---------------------------------------------------------------------------
// CachePool extensions
// ---------------------------------------------------------------------------

impl CachePool {
    /// Remove `seq_id` from the active set and return its per-layer caches as
    /// a `DetachedCacheSet` without freeing the MLX buffers.
    ///
    /// Returns `None` if:
    /// * `seq_id` is not currently active, or
    /// * the sequence uses the paged backend (paged detach is #418's
    ///   responsibility — this method deliberately rejects it).
    ///
    /// The caller is responsible for re-homing the detached set, either by
    /// passing it to [`CachePool::adopt`] or by parking it via
    /// [`CachePool::park_detached`]. Dropping the returned set releases the
    /// underlying MLX memory normally.
    ///
    /// Used by: prompt prefix cache store (#418), scheduler request-boundary
    /// handoff.
    pub fn detach(&mut self, seq_id: SequenceId) -> Option<DetachedCacheSet> {
        // Peek first so we can refuse non-dense backends without destructive
        // side effects.
        {
            let sequence = self.active.get(&seq_id)?;
            if sequence.backend != SequenceStateBackend::DenseKvCache {
                return None;
            }
        }

        let mut sequence = self.active.remove(&seq_id)?;
        let detached_caches: Vec<DetachedKVCache> = sequence
            .caches
            .iter_mut()
            .map(|cache| cache.clone_handle())
            .collect();

        Some(DetachedCacheSet {
            caches: detached_caches,
            backend: sequence.backend,
            prompt_len: sequence.prompt_len,
            current_offset: sequence.current_offset,
            created_at: sequence.created_at,
            detached_at: Instant::now(),
            origin_seq_id: sequence.seq_id,
        })
    }

    /// Install a previously-detached cache set under a fresh `SequenceId`.
    ///
    /// Capacity is checked against `max_sequences` before allocation. On
    /// success the model's
    /// [`prepare_sequence_state`](crate::generate::LanguageModel::prepare_sequence_state)
    /// hook is invoked with the new id so any per-model sidecar maps
    /// (mixed-cache models, quantized sidecars, etc.) are initialized
    /// consistently with a freshly allocated sequence.
    ///
    /// Only `DenseKvCache` sets are supported; attempting to adopt a paged
    /// set returns an error and the original set is dropped (its tensors
    /// freed) to avoid leaks. Use [`CachePool::adopt_preserving`] when the
    /// caller wants the set back on failure.
    ///
    /// Used by: prompt prefix cache re-entry (#418), scheduler fast-path
    /// when a new request reuses an existing prefix.
    pub fn adopt(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        detached: DetachedCacheSet,
    ) -> Result<SequenceId, String> {
        self.adopt_preserving(model, detached)
            .map_err(|(err, _)| err)
    }

    /// Like [`CachePool::adopt`] but returns the original [`DetachedCacheSet`]
    /// back to the caller on failure so it can be retried or routed
    /// elsewhere.
    pub fn adopt_preserving(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        detached: DetachedCacheSet,
    ) -> Result<SequenceId, (String, DetachedCacheSet)> {
        if detached.backend != SequenceStateBackend::DenseKvCache {
            return Err((
                format!(
                    "CachePool::adopt: backend {:?} is not supported (paged adopt is tracked in #418)",
                    detached.backend
                ),
                detached,
            ));
        }
        if self.active.len() >= self.max_sequences {
            return Err((
                format!(
                    "CachePool::adopt: max capacity ({}) reached, cannot adopt new sequence",
                    self.max_sequences
                ),
                detached,
            ));
        }

        let id = SequenceId::from_raw(self.next_id.fetch_add(1, Ordering::Relaxed));

        // Reconstruct the live per-layer caches from the detached handles.
        // `KVCache::install_detached` demands an empty target, which
        // `KVCache::new()` trivially satisfies.
        let DetachedCacheSet {
            caches,
            backend,
            prompt_len,
            current_offset,
            created_at,
            detached_at: _,
            origin_seq_id: _,
        } = detached;

        let mut live: Vec<KVCache> = Vec::with_capacity(caches.len());
        for detached_cache in caches {
            let mut cache = KVCache::new_with_mode(detached_cache.mode);
            cache
                .install_detached(detached_cache)
                .expect("freshly constructed KVCache is empty");
            live.push(cache);
        }

        let mut entry = SequenceCacheSet::dense_external(id, live);
        // Preserve the originating metadata across the handoff so scheduler
        // stats and reuse bookkeeping stay coherent.
        entry.backend = backend;
        entry.prompt_len = prompt_len;
        entry.current_offset = current_offset;
        entry.created_at = created_at;
        self.active.insert(id, entry);

        // Hook in the model-side sidecar state for the new id, matching the
        // normal `allocate` -> `prepare_sequence_state` sequencing that the
        // batch scheduler uses today.
        model.prepare_sequence_state(id);

        Ok(id)
    }

    /// Park a detached set inside the pool so its bytes remain visible to
    /// [`CachePool::memory_usage_bytes`].
    ///
    /// Returns an opaque [`DetachedHandle`] that can later be consumed by
    /// [`CachePool::take_parked`] or [`CachePool::adopt_parked`]. Parked
    /// caches do **not** count toward `active_count()` and do not consume
    /// an `allocate()` slot — they only contribute to memory accounting.
    pub fn park_detached(&mut self, detached: DetachedCacheSet) -> DetachedHandle {
        let handle = DetachedHandle(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.detached
            .insert(handle, super::paged_detach::ParkedCache::Dense(detached));
        handle
    }

    /// Retrieve a previously parked dense set, leaving the pool.
    ///
    /// Returns `None` if the handle was never parked, already taken, or
    /// points to a paged set (use [`CachePool::take_parked_paged`] for
    /// paged sets).
    pub fn take_parked(&mut self, handle: DetachedHandle) -> Option<DetachedCacheSet> {
        match self.detached.remove(&handle) {
            Some(super::paged_detach::ParkedCache::Dense(set)) => Some(set),
            Some(other) => {
                // Wrong variant — put it back so the caller can dispatch to
                // the paged-side take_parked.
                self.detached.insert(handle, other);
                None
            }
            None => None,
        }
    }

    /// Read-only peek at a parked dense set (for inspection / metrics).
    ///
    /// Returns `None` if the handle points to a paged set.
    pub fn peek_parked(&self, handle: DetachedHandle) -> Option<&DetachedCacheSet> {
        match self.detached.get(&handle) {
            Some(super::paged_detach::ParkedCache::Dense(set)) => Some(set),
            _ => None,
        }
    }

    /// Convenience: consume a parked handle and re-adopt it in one call.
    pub fn adopt_parked(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        handle: DetachedHandle,
    ) -> Result<SequenceId, String> {
        let detached = self
            .take_parked(handle)
            .ok_or_else(|| format!("CachePool::adopt_parked: unknown handle {handle}"))?;
        self.adopt(model, detached)
    }

    /// Number of currently parked detached sets (dense and paged combined).
    pub fn parked_count(&self) -> usize {
        self.detached.len()
    }

    /// Summed bytes across all parked detached sets (dense and paged).
    pub fn parked_bytes(&self) -> usize {
        self.detached.values().map(|d| d.nbytes()).sum()
    }
}

// Tests live in the companion `detach_tests.rs` so this file stays at a
// comfortable implementation-only size (see `docs/code-guidelines.md`).
#[cfg(test)]
#[path = "detach_tests.rs"]
mod tests;
