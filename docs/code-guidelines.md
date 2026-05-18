# Code Modification Guidelines

## Shared Function Comments

When modifying shared/common functions that multiple models depend on (e.g., attention implementations, normalization, KV cache, activation functions), **always add or update comments** indicating which models use that function.

**Format:**
```rust
// Used by: Llama, Qwen, Gemma2, Gemma3
fn repeat_kv(keys: &MlxArray, values: &MlxArray, n_rep: i32) -> ... {
    ...
}
```

**Why this matters:**
- Prevents regression when fixing one model from breaking others
- Makes it clear which models need retesting after changes
- Helps future developers understand the impact of modifications

**When to update:**
1. When adding a new model that uses an existing shared function
2. When modifying a shared function's behavior
3. When discovering that a function is used by a model not listed

**Key shared components to track:**
- `src/lib/mlxcel-core/src/layers.rs` - KVCache, Attention, Normalization
- `src/lib/mlxcel-core/src/utils.rs` - create_causal_mask, softcap, repeat_kv
- Model-specific attention variants in `src/models/*.rs`
