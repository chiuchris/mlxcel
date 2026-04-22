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

use super::*;

fn digest(byte: u8) -> PromptCacheKeyDigest {
    PromptCacheKeyDigest([byte; 32])
}

fn collect_all(trie: &RadixTrie, tokens: &[i32], min_len: usize) -> Vec<PromptCacheKeyDigest> {
    let mut out = Vec::new();
    if let Some(m) = trie.find_longest_prefix(tokens, min_len) {
        m.for_each_candidate(|d| out.push(d.digest));
    }
    out.sort_by_key(|d| d.0);
    out
}

#[test]
fn insert_and_exact_lookup() {
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3, 4], digest(1));
    let m = t.find_longest_prefix(&[1, 2, 3, 4], 0).expect("hit");
    assert_eq!(m.depth(), 4);
    let mut found: Vec<_> = Vec::new();
    m.for_each_candidate(|d| found.push(d.digest));
    assert_eq!(found, vec![digest(1)]);
}

#[test]
fn stored_longer_than_requested() {
    // Stored: [1,2,3,4,5]. Request: [1,2,3].
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3, 4, 5], digest(1));
    let m = t.find_longest_prefix(&[1, 2, 3], 0).expect("hit");
    assert_eq!(m.depth(), 3);
    let mut found: Vec<_> = Vec::new();
    m.for_each_candidate(|d| found.push(d));
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].digest, digest(1));
    assert_eq!(found[0].token_len, 5);
}

#[test]
fn stored_shorter_than_requested() {
    // Stored: [1,2,3]. Request: [1,2,3,4,5].
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3], digest(1));
    let m = t.find_longest_prefix(&[1, 2, 3, 4, 5], 0).expect("hit");
    assert_eq!(m.depth(), 3);
}

#[test]
fn divergence_past_common_prefix() {
    // Stored: [1,2,3,9]. Request: [1,2,3,5].
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3, 9], digest(1));
    let m = t.find_longest_prefix(&[1, 2, 3, 5], 0).expect("hit");
    assert_eq!(m.depth(), 3);
}

#[test]
fn min_len_threshold_filters_short_matches() {
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3, 4], digest(1));
    // Request diverges after 2 tokens; min_len=3 rejects the match.
    assert!(t.find_longest_prefix(&[1, 2, 9, 9], 3).is_none());
    // Request diverges after 3 tokens; min_len=3 accepts.
    let m = t.find_longest_prefix(&[1, 2, 3, 9], 3).expect("hit");
    assert_eq!(m.depth(), 3);
}

#[test]
fn multiple_entries_under_common_prefix() {
    // Two entries: [1,2,3,10] and [1,2,3,20]. Request [1,2,3,99] matches
    // only up to depth 3 and both entries sit in the subtree.
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3, 10], digest(1));
    t.insert(&[1, 2, 3, 20], digest(2));
    let m = t.find_longest_prefix(&[1, 2, 3, 99], 0).expect("hit");
    assert_eq!(m.depth(), 3);
    let mut found: Vec<_> = Vec::new();
    m.for_each_candidate(|d| found.push(d.digest));
    found.sort_by_key(|d| d.0);
    assert_eq!(found, vec![digest(1), digest(2)]);
}

#[test]
fn entries_at_various_depths() {
    // Entry at depth 3 (stored prefix [1,2,3]) and one at depth 6 on the
    // same path. Request [1,2,3,4,5,6] should match the deeper one.
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3], digest(1));
    t.insert(&[1, 2, 3, 4, 5, 6], digest(2));
    let m = t.find_longest_prefix(&[1, 2, 3, 4, 5, 6], 0).expect("hit");
    assert_eq!(m.depth(), 6);
    let mut found: Vec<_> = Vec::new();
    m.for_each_candidate(|d| found.push(d.digest));
    assert_eq!(found, vec![digest(2)]);

    // Request that diverges at depth 3 returns digest(1) (depth-3 entry)
    // plus digest(2) whose subtree contains ‘1,2,3,...’.
    let m = t.find_longest_prefix(&[1, 2, 3, 99], 0).expect("hit");
    assert_eq!(m.depth(), 3);
    let mut found: Vec<_> = Vec::new();
    m.for_each_candidate(|d| found.push(d.digest));
    found.sort_by_key(|d| d.0);
    assert_eq!(found, vec![digest(1), digest(2)]);
}

#[test]
fn remove_deletes_digest_and_prunes() {
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3], digest(1));
    t.insert(&[1, 2, 4], digest(2));
    assert_eq!(t.len(), 2);
    assert!(t.remove(&[1, 2, 3], digest(1)));
    assert_eq!(t.len(), 1);
    assert!(
        collect_all(&t, &[1, 2, 3], 0)
            .iter()
            .all(|d| *d != digest(1))
    );
    assert!(t.remove(&[1, 2, 4], digest(2)));
    assert_eq!(t.len(), 0);
    assert!(t.find_longest_prefix(&[1, 2, 3], 0).is_none());
}

#[test]
fn remove_absent_digest_is_noop() {
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3], digest(1));
    assert!(!t.remove(&[1, 2, 3], digest(9)));
    assert_eq!(t.len(), 1);
}

#[test]
fn empty_trie_returns_none() {
    let t = RadixTrie::new();
    assert!(t.find_longest_prefix(&[1, 2, 3], 0).is_none());
}

#[test]
fn idempotent_insert_does_not_double_count() {
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3], digest(1));
    t.insert(&[1, 2, 3], digest(1));
    assert_eq!(t.len(), 1);
}

#[test]
fn split_then_insert_adds_branch() {
    // Insert A=[1,2,3,4]. Then B=[1,2,5]. Edge [1,2,3,4] must split
    // into shared [1,2] and two children: [3,4] (A) and [5] (B).
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3, 4], digest(1));
    t.insert(&[1, 2, 5], digest(2));
    assert_eq!(t.len(), 2);
    let m = t.find_longest_prefix(&[1, 2, 3, 4], 0).expect("hit A");
    assert_eq!(m.depth(), 4);
    let m = t.find_longest_prefix(&[1, 2, 5], 0).expect("hit B");
    assert_eq!(m.depth(), 3);
    let m = t.find_longest_prefix(&[1, 2, 7], 0).expect("prefix hit");
    assert_eq!(m.depth(), 2);
}

#[test]
fn deep_insert_scales_linearly_in_prefix_len() {
    // 32k tokens — insert, lookup, remove should each complete without
    // pathological slowdown.
    let tokens: Vec<i32> = (0..32_768).collect();
    let mut t = RadixTrie::new();
    t.insert(&tokens, digest(1));
    let m = t.find_longest_prefix(&tokens, 0).expect("hit");
    assert_eq!(m.depth(), tokens.len());
    assert!(t.remove(&tokens, digest(1)));
    assert_eq!(t.len(), 0);
}
