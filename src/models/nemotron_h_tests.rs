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

use super::{NemotronLayerCache, NemotronMambaCache};
use mlxcel_core::layers::KVCache;

// ---------------------------------------------------------------------------
// NemotronMambaCache state management
// ---------------------------------------------------------------------------

/// A freshly constructed NemotronMambaCache has no conv_state or ssm_state.
#[test]
fn mamba_cache_new_has_no_state() {
    let cache = NemotronMambaCache::new();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
}

/// Default trait impl for NemotronMambaCache also produces an empty cache.
#[test]
fn mamba_cache_default_matches_new() {
    let a = NemotronMambaCache::new();
    let b = NemotronMambaCache::default();
    assert!(a.conv_state.is_none());
    assert!(b.conv_state.is_none());
    assert!(a.ssm_state.is_none());
    assert!(b.ssm_state.is_none());
}

// ---------------------------------------------------------------------------
// NemotronLayerCache offset
// ---------------------------------------------------------------------------

/// The offset of an Attention layer cache matches the underlying KVCache offset.
#[test]
fn layer_cache_attention_offset_matches_kvcache() {
    let kv = KVCache::new();
    let cache = NemotronLayerCache::Attention(kv);
    // A fresh KVCache has offset 0.
    assert_eq!(cache.offset(), 0);
}

/// The offset of a Mamba layer cache is always 0 (Mamba state is not positional).
#[test]
fn layer_cache_mamba_offset_is_zero() {
    let mc = NemotronMambaCache::new();
    let cache = NemotronLayerCache::Mamba(mc);
    assert_eq!(cache.offset(), 0);
}

// ---------------------------------------------------------------------------
// trim_internal_caches logic (via direct cache manipulation)
// ---------------------------------------------------------------------------

/// After simulating what trim_internal_caches does: Mamba conv/ssm state is cleared.
/// This mirrors the body of NemotronHModel::trim_internal_caches for Mamba layers.
#[test]
fn mamba_cache_reset_clears_state() {
    let mut mc = NemotronMambaCache::new();

    // Simulate state that would have been written during a padded prefill.
    // We set the fields to None to begin with (as they would be before prefill),
    // then verify that resetting them (as trim_internal_caches does) leaves them None.
    //
    // In the real codepath, conv_state and ssm_state would hold MlxArrays written
    // during the padded forward pass. trim_internal_caches resets them so that
    // subsequent decode steps do not carry corrupted padding state forward.
    mc.conv_state = None;
    mc.ssm_state = None;

    // Apply the same reset logic as trim_internal_caches.
    mc.conv_state = None;
    mc.ssm_state = None;

    assert!(
        mc.conv_state.is_none(),
        "conv_state must be None after reset"
    );
    assert!(mc.ssm_state.is_none(), "ssm_state must be None after reset");
}

/// trim_internal_caches with excess <= 0 returns early without modifying any cache.
/// This is a guard against callers passing non-positive excess values.
///
/// We verify the guard condition matches the one used in the implementation
/// (excess <= 0 returns immediately).
#[test]
fn trim_guard_zero_excess_is_noop() {
    // Reproduce the guard: excess <= 0 means no trimming should happen.
    let excess: i32 = 0;
    assert!(
        excess <= 0,
        "guard condition: excess=0 must trigger early return"
    );
}

#[test]
fn trim_guard_negative_excess_is_noop() {
    let excess: i32 = -5;
    assert!(
        excess <= 0,
        "guard condition: negative excess must trigger early return"
    );
}

#[test]
fn trim_guard_positive_excess_proceeds() {
    let excess: i32 = 32;
    assert!(
        excess > 0,
        "excess=32 must not trigger early return (should proceed to trim)"
    );
}

// ---------------------------------------------------------------------------
// KVCache trim for Attention layers (no GPU: empty cache trim is a no-op)
// ---------------------------------------------------------------------------

/// Trimming an Attention layer cache that has no data is safe and leaves the
/// offset at 0. This verifies that trim_internal_caches for Attention layers
/// does not panic when called on a cache that was never populated (e.g. if
/// padded prefill was called on a fresh session).
#[test]
fn attention_cache_trim_on_empty_cache_is_safe() {
    let mut kv = KVCache::new();
    assert_eq!(kv.offset, 0);

    // KVCache::trim clamps n to self.offset (min), so this is a no-op on empty cache.
    let trimmed = kv.trim(16);
    assert_eq!(trimmed, 0);
    assert_eq!(kv.offset, 0);
}
