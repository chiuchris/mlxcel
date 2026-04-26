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

//! TurboQuant KV cache compression (epic #458).
//!
//! This module is the entry point for all TurboQuant / PolarQuant components
//! used by the KV cache compression pipeline. Sub-modules are added
//! incrementally as the sub-issues land:
//!
//! | Sub-module    | Sub-issue | Status   | Description                      |
//! |---------------|-----------|----------|----------------------------------|
//! | `codebook`    | B1 (#472) | ✓ done   | Lloyd-Max centroid generator     |
//! | `quant`       | B2 (#474) | ✓ done   | V-side PolarQuant pipeline       |
//! | (more to come)| B3–B12    | pending  | Quality gates, K-side, paged KV  |
//!
//! # Usage by downstream sub-issues
//!
//! ```rust,ignore
//! use mlxcel_core::cache::turbo::codebook::optimal_codebook;
//!
//! let cb = optimal_codebook(4, 128);
//! ```

pub mod codebook;
pub mod quant;

// Re-export the most commonly used entry points for convenience
pub use codebook::{
    compute_centroids, nearest_centroid_indices, nearest_centroid_indices_with_boundaries,
    optimal_centroids, optimal_codebook, Codebook,
};
pub use quant::{
    dequantize_v_turbo4, generate_signs, quantize_v_turbo4, turbo4_v_rotate, TurboQuantParams,
    BLOCK_SIZE, V_BIT_WIDTH,
};
