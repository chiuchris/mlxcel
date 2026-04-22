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

//! Cache-key hashing and composition for the prompt prefix cache.
//!
//! A [`PromptCacheKey`] composes the identity dimensions that define a
//! reusable KV-cache bucket:
//!
//! * `model_id`     — which loaded model produced these tensors.
//! * `lora_id`      — LoRA adapter identifier (`None` means base model).
//! * `template_sig` — chat-template digest so two requests with different
//!   `<|im_start|>` / tool prompts never collide. Full template-signature
//!   wiring is sub-issue #422's scope; this type accepts the input
//!   verbatim and hashes it.
//! * `session_key`  — optional caller-supplied tenancy / conversation scope.
//! * `tokens[..prefix_len]` — the actual prefix of token ids that the cache
//!   bucket represents. Only the first `prefix_len` entries participate in
//!   the bucket digest, so partially matching prefixes still map to
//!   distinct buckets.
//!
//! The bucket digest uses BLAKE3 because it is fast, stable across processes
//! and architectures, and has no per-call allocation overhead for typical
//! prompt prefix sizes. The digest is **not** treated as a security primitive;
//! it exists purely to support `HashMap`-style lookup across sequences.

use std::fmt;

/// A 32-byte BLAKE3 digest identifying a prompt-prefix cache bucket.
///
/// Used by [`super::store::PromptCacheStore`] as the primary map key.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PromptCacheKeyDigest(pub(crate) [u8; 32]);

impl PromptCacheKeyDigest {
    /// Raw byte view, useful for logging.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hex representation of the 32-byte digest.
    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    /// Short hex prefix (16 hex chars, 8 bytes) for log lines.
    pub fn short_hex(self) -> String {
        let mut out = String::with_capacity(16);
        for byte in &self.0[..8] {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }
}

impl fmt::Debug for PromptCacheKeyDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PromptCacheKeyDigest({})", self.short_hex())
    }
}

impl fmt::Display for PromptCacheKeyDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Composite identity for a prompt-prefix cache bucket.
///
/// Construct one per logical lookup or insert; the struct intentionally
/// borrows its inputs so callers don't have to clone strings and token
/// vectors on the hot path.
#[derive(Clone, Copy, Debug)]
pub struct PromptCacheKey<'a> {
    /// Loaded model identifier (typically the display `model_id` — whatever
    /// `AppState::display_model_id` returns).
    pub model_id: &'a str,
    /// LoRA adapter identifier; `None` for the base model.
    pub lora_id: Option<&'a str>,
    /// Chat-template signature. Full wiring is #422; for now any stable
    /// digest of the template + tool-schema inputs is acceptable and will
    /// simply slot into this field.
    pub template_sig: &'a str,
    /// Caller-supplied tenancy / conversation scope. `None` means global.
    pub session_key: Option<&'a str>,
    /// Prefix slice of token ids. Only the first `prefix_len` elements are
    /// hashed.
    pub tokens: &'a [i32],
    /// Number of tokens from the start of `tokens` to include in the bucket
    /// digest. Must be `<= tokens.len()`; longer values saturate.
    pub prefix_len: usize,
}

impl<'a> PromptCacheKey<'a> {
    /// Build a cache key where the full `tokens` slice participates in the
    /// bucket digest.
    pub fn new_full(
        model_id: &'a str,
        lora_id: Option<&'a str>,
        template_sig: &'a str,
        session_key: Option<&'a str>,
        tokens: &'a [i32],
    ) -> Self {
        Self {
            model_id,
            lora_id,
            template_sig,
            session_key,
            tokens,
            prefix_len: tokens.len(),
        }
    }

    /// Build a cache key where only the first `prefix_len` tokens are hashed.
    pub fn new_prefix(
        model_id: &'a str,
        lora_id: Option<&'a str>,
        template_sig: &'a str,
        session_key: Option<&'a str>,
        tokens: &'a [i32],
        prefix_len: usize,
    ) -> Self {
        Self {
            model_id,
            lora_id,
            template_sig,
            session_key,
            tokens,
            prefix_len: prefix_len.min(tokens.len()),
        }
    }

    /// Effective number of tokens that participate in the bucket digest.
    pub fn effective_prefix_len(&self) -> usize {
        self.prefix_len.min(self.tokens.len())
    }

    /// Compute the 32-byte BLAKE3 digest that identifies this bucket.
    ///
    /// The input is a length-prefixed, domain-separated concatenation of:
    ///
    /// ```text
    ///     "mlxcel:prompt-cache:v1"
    ///     model_id              (len-prefixed utf-8 bytes)
    ///     lora_id               (len-prefixed utf-8 bytes, empty if None)
    ///     template_sig          (len-prefixed utf-8 bytes)
    ///     session_key           (len-prefixed utf-8 bytes, empty if None)
    ///     prefix_len            (u64 little-endian)
    ///     tokens[..prefix_len]  (each token as i32 little-endian)
    /// ```
    ///
    /// Length prefixes and the fixed `v1` tag keep the digest resistant to
    /// accidental collisions between fields with overlapping bytes, and make
    /// it safe to extend the schema later (new fields bump the `v1` tag).
    pub fn digest(&self) -> PromptCacheKeyDigest {
        let mut hasher = blake3::Hasher::new();

        // Domain separator so this digest cannot be reused from a different
        // hashing context accidentally.
        hasher.update(b"mlxcel:prompt-cache:v1");

        write_field(&mut hasher, self.model_id.as_bytes());
        write_field(
            &mut hasher,
            self.lora_id.map(str::as_bytes).unwrap_or_default(),
        );
        write_field(&mut hasher, self.template_sig.as_bytes());
        write_field(
            &mut hasher,
            self.session_key.map(str::as_bytes).unwrap_or_default(),
        );

        let prefix_len = self.effective_prefix_len();
        hasher.update(&(prefix_len as u64).to_le_bytes());
        for tok in &self.tokens[..prefix_len] {
            hasher.update(&tok.to_le_bytes());
        }

        let mut out = [0u8; 32];
        hasher.finalize_xof().fill(&mut out);
        PromptCacheKeyDigest(out)
    }
}

fn write_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
#[path = "key_tests.rs"]
mod tests;
