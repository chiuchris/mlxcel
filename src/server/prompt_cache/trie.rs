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

//! Per-bucket path-compressed radix trie over token-id prefixes.
//!
//! # Why a radix trie?
//!
//! `PromptCacheStore::lookup_longest_prefix` needs to find, among all cached
//! entries that share the same `(model_id, lora_id, template_sig)` bucket, the
//! stored entry whose token prefix forms the longest common prefix with the
//! incoming request's token vector.
//!
//! Two reasonable data structures were considered:
//!
//! 1. **Hash-of-prefixes with sorted prefix lengths.** Precompute hashes of the
//!    request at a few prefix lengths (say, every power of two up to the
//!    request length) and probe a hash map for each. Each probe costs a
//!    full `O(L)` token hash, and a hit still requires an `O(L)` linear
//!    compare to validate against a non-cryptographic collision. Total work is
//!    `O(L log L)` in the number of tokens, plus global rehash on every
//!    insert/delete. The hot path is dominated by hashing, not structure
//!    traversal, but the structure itself has no sharing — every inserted
//!    prefix consumes `O(L)` bytes in its own hash-entry value.
//!
//! 2. **Per-bucket path-compressed radix trie (this module).** Lookup cost is
//!    `O(L)` in the request length, traversing at most one trie node per
//!    divergent token. Insertion shares edges with existing entries that have
//!    a common prefix, which is the common case for cross-request prompt
//!    caching (large shared system prompts, tool preambles, long document
//!    contexts). Deletion is a bounded rewrite — remove a digest at a node
//!    and, if the node becomes empty and has a single child, merge the edge
//!    back. In-memory overhead per distinct-suffix node is one `HashMap<i32,
//!    Box<Node>>` plus the edge `Vec<i32>`.
//!
//! We chose the radix trie because:
//!
//! * It turns the **common case** (many requests sharing a long system
//!   prompt) into cheap pointer chasing rather than redundant hashing. The
//!   hot-path work scales with *divergence depth*, not with the absolute
//!   token count.
//! * Lookup is deterministically `O(L)` for any realistic prompt (≤ 32k
//!   tokens). That bound holds regardless of how many cached entries live in
//!   the same bucket, which is the property the epic requires.
//! * Inserts/deletes are `O(L)` and don't trigger global rebuilds.
//! * It composes cleanly with the two-tier lookup (exact-session + same
//!   model/lora/template fallback): both tiers run against the same trie
//!   and filter by `session_key` during digest selection.
//!
//! The downside — per-node `HashMap<i32, Box<Node>>` — is tolerable because
//! typical branching factors at any one node are very small (most tokens
//! along a shared prompt don't branch), and path compression collapses long
//! linear runs into a single edge label.
//!
//! # Layout
//!
//! Each node stores:
//!
//! * `edge` — the run of tokens on the incoming edge (path compression).
//! * `entries` — digests whose stored prefix ends exactly at this node's
//!   depth.
//! * `subtree_count` — total `entries` across this node and all descendants,
//!   used to short-circuit "empty subtree" cases during lookup.
//! * `children` — hash map keyed by the first token of each child edge.
//!
//! # Invariants
//!
//! * The root node has an empty `edge` and `entries` never contains its own
//!   digest (the root represents the empty prefix; real entries must have
//!   length >= 1 to be stored at all).
//! * Every non-root node has `edge.len() >= 1`.
//! * `children` is keyed by `edge[0]` of the child; this is what makes
//!   lookup deterministic.
//! * A node with empty `entries` and empty `children` is a leaf candidate
//!   for pruning; `remove` unconditionally prunes.
//! * A node with empty `entries` and exactly one child is a candidate for
//!   edge merging on `remove`.

use std::collections::HashMap;

use super::key::PromptCacheKeyDigest;

/// A single digest paired with the stored token length it corresponds to.
///
/// The token length matters for the caller's matched-length bookkeeping: a
/// lookup may surface a digest whose stored prefix is *longer* than the
/// traversal depth (it lives deeper in the trie), in which case the matched
/// length against the incoming request is clamped at the traversal depth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DigestAndLen {
    pub digest: PromptCacheKeyDigest,
    pub token_len: usize,
}

/// Path-compressed radix trie node keyed on `i32` token ids.
#[derive(Debug, Default)]
pub(super) struct TrieNode {
    /// The run of tokens labeling the edge that leads into this node from
    /// its parent. Empty only for the synthetic root.
    edge: Vec<i32>,
    /// Digests whose stored token vector ends exactly at this node's depth.
    /// Plain `Vec` because bucket cardinality is tiny and we care about
    /// cache locality on scan more than on point-contains.
    entries: Vec<DigestAndLen>,
    /// `entries.len()` summed across this node and all descendants. Used as
    /// a cheap "is subtree empty" check.
    subtree_count: usize,
    /// Children keyed by the first token of the child's `edge`.
    children: HashMap<i32, Box<TrieNode>>,
}

/// Root of the per-bucket radix trie.
#[derive(Debug, Default)]
pub(super) struct RadixTrie {
    root: TrieNode,
}

impl RadixTrie {
    /// Test-only constructor kept alongside `Default` for symmetry with
    /// the data-type's constructor family and readable unit tests.
    #[cfg(test)]
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Total number of digests stored across this trie.
    pub(super) fn len(&self) -> usize {
        self.root.subtree_count
    }

    /// Insert `digest` at the position described by `tokens`. Returns the
    /// depth at which the digest was installed (always equal to
    /// `tokens.len()`).
    ///
    /// Safe to call repeatedly with the same `(digest, tokens)`; the digest
    /// is stored at most once per node.
    pub(super) fn insert(&mut self, tokens: &[i32], digest: PromptCacheKeyDigest) {
        let record = DigestAndLen {
            digest,
            token_len: tokens.len(),
        };
        insert_into(&mut self.root, tokens, record);
    }

    /// Remove the given `digest` from the trie. No-op if the digest isn't
    /// present (callers must therefore remember the insertion tokens).
    /// Returns `true` if the digest was found and removed.
    pub(super) fn remove(&mut self, tokens: &[i32], digest: PromptCacheKeyDigest) -> bool {
        remove_from(&mut self.root, tokens, digest, /*is_root=*/ true)
    }

    /// Find the longest token-prefix of `tokens` that reaches a position
    /// whose subtree contains at least one stored digest. Returns the
    /// matched depth and a reference to the subtree root whose entries all
    /// share the matched prefix with `tokens`.
    ///
    /// Returns `None` when no entry in the trie matches a prefix of at
    /// least `min_len` tokens. The returned depth is always `>= min_len`
    /// and always `<= tokens.len()`.
    pub(super) fn find_longest_prefix(
        &self,
        tokens: &[i32],
        min_len: usize,
    ) -> Option<TrieMatch<'_>> {
        let m = walk_longest(&self.root, tokens)?;
        if m.depth < min_len {
            return None;
        }
        Some(m)
    }
}

/// Successful longest-prefix walk result.
///
/// `depth` — number of leading request tokens that successfully matched
/// somewhere in the trie. This is the matched length for any entry under
/// the returned `subtree` (capped at the stored entry's own length).
///
/// `subtree` — the trie node at or beyond the matched depth whose entries
/// all share the matched prefix. When the walk stopped exactly at a node
/// boundary this is that node itself; when it stopped inside an edge
/// (path-compressed), it is the child node the edge points to, because
/// any entry under that child still has the matched prefix as its own
/// prefix.
pub(super) struct TrieMatch<'a> {
    subtree: &'a TrieNode,
    /// When the walk ended inside an edge, this records how many tokens of
    /// the traversed edge have **not** been matched: those belong to the
    /// `subtree` node's own edge (all entries in the subtree still have
    /// those tokens, so they're past the matched depth). When the walk
    /// ended exactly at a node boundary, this is 0.
    consumed_from_subtree_edge: usize,
    depth: usize,
}

impl<'a> TrieMatch<'a> {
    /// Number of request tokens that matched (i.e. the traversal depth).
    pub(super) fn depth(&self) -> usize {
        self.depth
    }

    /// Walk the matched subtree and invoke `visit` with every stored digest.
    /// Traversal order is DFS, but callers must not depend on any particular
    /// ordering.
    pub(super) fn for_each_candidate<F: FnMut(DigestAndLen)>(&self, mut visit: F) {
        // When we stopped mid-edge, `subtree`'s own `entries` are at a depth
        // *greater* than the matched depth (they extend past the match
        // point). They still share the matched prefix with the request, so
        // they count — their match length is clamped at the matched depth.
        // `self.consumed_from_subtree_edge > 0` indicates this mid-edge
        // case; entries on this node are still valid candidates either way,
        // so we traverse the whole subtree rooted here.
        let _ = self.consumed_from_subtree_edge;
        dfs(self.subtree, &mut visit);
    }
}

fn dfs<F: FnMut(DigestAndLen)>(node: &TrieNode, visit: &mut F) {
    for d in &node.entries {
        visit(*d);
    }
    for child in node.children.values() {
        dfs(child, visit);
    }
}

fn walk_longest<'a>(root: &'a TrieNode, tokens: &[i32]) -> Option<TrieMatch<'a>> {
    if root.subtree_count == 0 {
        return None;
    }
    let mut node: &TrieNode = root;
    let mut consumed = 0usize;
    // Walk until we reach a point where no child can extend the match any
    // further OR we've consumed all request tokens.
    loop {
        if consumed >= tokens.len() {
            // Request exhausted exactly at a node boundary.
            break;
        }
        let next_tok = tokens[consumed];
        let child = match node.children.get(&next_tok) {
            Some(c) => c.as_ref(),
            None => break,
        };
        let edge = child.edge.as_slice();
        let remaining = &tokens[consumed..];
        let matchable = edge.len().min(remaining.len());
        let mut matched = 0usize;
        while matched < matchable && edge[matched] == remaining[matched] {
            matched += 1;
        }
        if matched == 0 {
            // Defensive: should be unreachable since the children map is
            // keyed by `edge[0]` and we only descended because `edge[0] ==
            // next_tok`.
            break;
        }
        consumed += matched;
        if matched < edge.len() {
            // Walk stopped mid-edge. The subtree rooted at `child` has
            // every entry sharing the first `consumed` request tokens; the
            // tokens past the match point are shared across the whole
            // subtree (they're part of `child.edge`).
            return Some(TrieMatch {
                subtree: child,
                consumed_from_subtree_edge: matched,
                depth: consumed,
            });
        }
        // Whole child edge consumed — descend.
        node = child;
    }
    if consumed == 0 {
        // We didn't advance past root; treat as a miss.
        return None;
    }
    Some(TrieMatch {
        subtree: node,
        consumed_from_subtree_edge: 0,
        depth: consumed,
    })
}

fn insert_into(node: &mut TrieNode, tokens: &[i32], record: DigestAndLen) {
    if tokens.is_empty() {
        if !node.entries.iter().any(|e| e.digest == record.digest) {
            node.entries.push(record);
            node.subtree_count += 1;
        } else {
            // Replace in place to keep token_len accurate after token
            // updates (not typically expected, but safe).
            for e in &mut node.entries {
                if e.digest == record.digest {
                    *e = record;
                }
            }
        }
        return;
    }

    let first_tok = tokens[0];
    if let Some(child) = node.children.get_mut(&first_tok) {
        let edge_clone = child.edge.clone();
        let common = common_prefix_len(&edge_clone, tokens);
        if common == edge_clone.len() {
            // Full edge consumed: recurse into child.
            insert_into(child, &tokens[common..], record);
            node.subtree_count = recompute_subtree_count(node);
            return;
        }
        // Edge diverges mid-way: split.
        // Create a new intermediate node that keeps the first `common` tokens
        // of the old edge as its own incoming edge. Move the old child under
        // the new intermediate with its edge truncated to what's left.
        let shared: Vec<i32> = edge_clone[..common].to_vec();
        let old_remainder: Vec<i32> = edge_clone[common..].to_vec();

        // Rebuild the existing child with its shorter edge.
        let mut old_child = node.children.remove(&first_tok).expect("child just seen");
        let old_first_after_split = old_remainder[0];
        old_child.edge = old_remainder;

        // New intermediate holds no entries yet.
        let mut intermediate = TrieNode {
            edge: shared,
            entries: Vec::new(),
            subtree_count: 0,
            children: HashMap::new(),
        };
        intermediate
            .children
            .insert(old_first_after_split, old_child);

        if tokens.len() == common {
            // New record ends exactly at the split point.
            intermediate.entries.push(record);
        } else {
            // New record continues past the split; add a fresh child for
            // the remaining tokens.
            let new_remainder: Vec<i32> = tokens[common..].to_vec();
            let new_first = new_remainder[0];
            let new_child = TrieNode {
                edge: new_remainder,
                entries: vec![record],
                subtree_count: 1,
                children: HashMap::new(),
            };
            intermediate.children.insert(new_first, Box::new(new_child));
        }
        intermediate.subtree_count = recompute_subtree_count(&intermediate);
        node.children.insert(first_tok, Box::new(intermediate));
        node.subtree_count = recompute_subtree_count(node);
        return;
    }

    // No child for this first token yet: create a fresh leaf with the full
    // remaining tokens as its edge.
    let edge: Vec<i32> = tokens.to_vec();
    let leaf = TrieNode {
        edge,
        entries: vec![record],
        subtree_count: 1,
        children: HashMap::new(),
    };
    node.children.insert(first_tok, Box::new(leaf));
    node.subtree_count = recompute_subtree_count(node);
}

fn remove_from(
    node: &mut TrieNode,
    tokens: &[i32],
    digest: PromptCacheKeyDigest,
    is_root: bool,
) -> bool {
    if tokens.is_empty() {
        let before = node.entries.len();
        node.entries.retain(|e| e.digest != digest);
        let removed = node.entries.len() != before;
        if removed {
            node.subtree_count = recompute_subtree_count(node);
        }
        return removed;
    }

    let first_tok = tokens[0];
    let mut take_child = false;
    let removed = if let Some(child) = node.children.get_mut(&first_tok) {
        let edge_len = child.edge.len();
        if tokens.len() < edge_len || tokens[..edge_len] != child.edge[..] {
            false
        } else {
            let rest = &tokens[edge_len..];
            let removed = remove_from(child, rest, digest, false);
            if removed && child.entries.is_empty() && child.children.is_empty() {
                // Leaf with no entries — drop it.
                take_child = true;
            }
            removed
        }
    } else {
        false
    };

    if take_child {
        node.children.remove(&first_tok);
    }

    // After removing, compress the chain: if `node` is non-root, has no
    // entries, and has exactly one child, we can merge. But the caller owns
    // the current node, so we can only merge children of this node into
    // their grandchildren from here. We handle this by opportunistically
    // merging each single-child chain reachable from here.
    if !is_root && removed {
        merge_only_child_if_empty(node);
    }

    if removed {
        node.subtree_count = recompute_subtree_count(node);
    }
    removed
}

fn merge_only_child_if_empty(node: &mut TrieNode) {
    if !node.entries.is_empty() || node.children.len() != 1 {
        return;
    }
    // Take the single child.
    let only_key = *node.children.keys().next().expect("single child");
    let mut child = node.children.remove(&only_key).expect("single child");
    // Fuse child edge onto `node.edge`.
    let mut fused = std::mem::take(&mut node.edge);
    fused.extend_from_slice(&child.edge);
    node.edge = fused;
    node.entries = std::mem::take(&mut child.entries);
    node.children = std::mem::take(&mut child.children);
    node.subtree_count = recompute_subtree_count(node);
}

fn recompute_subtree_count(node: &TrieNode) -> usize {
    let mut total = node.entries.len();
    for c in node.children.values() {
        total += c.subtree_count;
    }
    total
}

fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
#[path = "trie_tests.rs"]
mod tests;
