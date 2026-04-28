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

#include "sparse_v_sdpa.h"

#include <mlx/fast.h>
#include <mlx/ops.h>

#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlxcel::turbo {

namespace {

// Source for the fused Sparse-V weighted-sum Metal kernel. The string is the
// kernel BODY only; `mlx::core::fast::metal_kernel` wraps it with the
// declaration, buffer arguments, and template-argument substitution.
//
// See `sparse_v_sdpa.metal` for an annotated reference copy.
//
// IMPORTANT: per-thread skipping is the WHOLE point of this kernel. The
// `if (any_alive == 0) continue;` block is what produces the speedup at long
// context — every skipped token saves the codebook gather, norm fetch, sqrt-
// reduction, and (D × RepeatCount) accumulator updates that the graph-level
// `where(alive, V_dq, 0)` path still pays.
//
// Layout contract:
//   weights_shape = [B*Hq, Tq, Tk]   f32
//   norms_shape   = [B*Hkv, Tk]      f16
//   packed_shape  = [B*Hkv, Tk, D/2] u8 (low/high nibble = even/odd dim)
//   threshold     = [1]              f32 (alive cutoff scalar passed as array)
//   codebook      = [16]             f32 (Lloyd-Max centroids, 4-bit)
//   out_shape     = [B*Hq, Tq, D]    f32 (unrotated weighted sum)
//
// The caller applies the inverse Turbo4 rotation (`signs1 * WHT * signs2 *`)
// to the [B, Hq, Tq, D] output, so the per-cache-token rotation cost is
// eliminated.
//
// Threadgroup layout: (Dim, 1, 1). One threadgroup per (B*Hq) slot. Threads
// share `tg_y_hat` (D floats) so a per-token `|y_hat|` reduction can be done
// in lockstep without a host round-trip.
constexpr const char* SPARSE_V_WEIGHTED_SUM_SOURCE = R"(
    auto d = thread_position_in_grid.x;
    auto n = thread_position_in_grid.z;
    auto tg_d = thread_position_in_threadgroup.x;

    // Decode `n` into (batch * Hq) and infer the matching Hkv slot. We pass
    // the layout sizes via the input shapes; `weights_shape` is
    // [B*Hq, Tq, Tk], `norms_shape` is [B*Hkv, Tk]. From these we recover
    // B*Hq and B*Hkv directly without needing extra constants.
    auto bhq = n;                              // 0 .. B*Hq-1
    auto t_count = (uint)weights_shape[2];     // Tk
    auto bhkv_count = (uint)norms_shape[0];    // B*Hkv
    auto bhq_count = (uint)weights_shape[0];   // B*Hq

    // Grouped attention: NRep = Hq / Hkv. To map B*Hq → B*Hkv we need the
    // (b, hq) decomposition. Hq = bhq_count * Hkv / bhq_count = NRep * Hkv,
    // so Hq = (bhq_count / bhkv_count) * NRep — but more usefully, we know
    // Hq = bhq_count / B and Hkv = bhkv_count / B. NRep is given as a
    // template constant. Since bhq_count / NRep == bhkv_count holds for
    // valid inputs, the per-`n` Hkv slot is `n / NRep`.
    uint nrep_u = (uint)NRep;
    uint bhkv = bhq / nrep_u;
    if (bhkv >= bhkv_count) {
        bhkv = 0;  // defensive fallback for inconsistent shapes
    }

    auto wt = weights + bhq * (uint)RepeatCount * t_count; // [Tq, Tk]
    auto nm = norms + bhkv * t_count;                       // [Tk]
    auto pk = packed + bhkv * t_count * (uint)(Dim / 2);    // [Tk, D/2]

    // Thresholds passed as a 1-element array so MLX accepts them as a regular
    // input buffer. (TemplateArg only supports int/bool/Dtype; ScalarArg
    // is reserved for `precompiled_*_kernel` paths, not `metal_kernel`.)
    float thresh = threshold[0];

    // One accumulator per Q token. RepeatCount is a template constant so
    // the array size is known at compile time and lives in registers.
    float acc[RepeatCount];
    for (int r = 0; r < RepeatCount; r++) {
        acc[r] = 0.0f;
    }

    // Per-dim packed-byte coordinates: byte index = d/2; nibble = d%2.
    uint byte_idx = (uint)d >> 1u;
    uint nibble_shift = ((uint)d & 1u) * 4u;

    // Threadgroup-shared buffer used to compute `|y_hat[t]|` per token via a
    // simple parallel reduction. Each thread writes its `code²` into
    // `tg_y_hat[d]`, then a barrier + tree reduction folds the sum into
    // tg_y_hat[0]. Size is `Dim` (template constant, ≤ 1024 for typical
    // head_dim values).
    threadgroup float tg_y_hat[Dim];
    threadgroup float tg_norm[1];

    // Sweep over T tokens. Each token: check if it's alive on at least one
    // Q slot, then look up the centroid for `(t, d)`, cooperatively reduce
    // |y_hat[t]| across the threadgroup, multiply by per-token norm and
    // (1 / |y_hat|), and gate by per-Q attention weight.
    for (uint t = 0; t < t_count; t++) {
        // Aliveness check: scan RepeatCount Q slots and short-circuit if all
        // are below threshold for this (b, h, t). For RepeatCount=1 (decode
        // case) this is a single compare-and-skip — the dominant production
        // path on Apple Silicon at long context. For RepeatCount > 1 the
        // compiler unrolls the inner loop.
        bool any_alive = false;
        for (int r = 0; r < RepeatCount; r++) {
            if (wt[r * t_count + t] > thresh) {
                any_alive = true;
                break;
            }
        }
        if (!any_alive) {
            // Per-thread skip — the speed gate of this kernel. We MUST also
            // synchronize the threadgroup memory we are about to read in the
            // next live iteration, but the next iteration starts with each
            // thread writing `tg_y_hat[d]` and then a barrier, so there is
            // no stale-read hazard from skipping the unpack/gather steps.
            continue;
        }

        // Unpack nibble for this (t, d). Byte offset = byte_idx, nibble
        // selected by nibble_shift (0 for even d, 4 for odd d). Out-of-Dim
        // threads (d >= Dim) write a benign 0 contribution — the `d < Dim`
        // bounds gate at the bottom prevents stray output writes.
        float code = 0.0f;
        if (d < (uint)Dim) {
            uint pb = (uint)pk[t * (uint)(Dim / 2) + byte_idx];
            uint idx = (pb >> nibble_shift) & 0x0Fu;
            code = codebook[idx];
        }

        // Cooperative |y_hat[t]| reduction across threadgroup. Each thread
        // contributes its `code²`. Tree reduction folds the per-d squares
        // into `tg_y_hat[0]`. Threadgroup size is exactly `Dim` so the
        // reduction unrolls cleanly (Dim is a template constant).
        tg_y_hat[tg_d] = code * code;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction with stride halving. Every iteration halves the
        // active thread count; threads above `stride` are idle.
        for (uint stride = (uint)Dim >> 1; stride > 0; stride >>= 1) {
            if (tg_d < stride) {
                tg_y_hat[tg_d] += tg_y_hat[tg_d + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Thread 0 computes the per-token rescale factor:
        //   rescale = (1 / max(|y_hat|, eps)) * v_norms[t]
        // and broadcasts it via `tg_norm[0]`. eps matches the graph path's
        // 1e-10 guard (see `dequantize_from_packed`).
        if (tg_d == 0) {
            float yh_sumsq = tg_y_hat[0];
            float yh_norm = sqrt(yh_sumsq);
            float yh_safe = max(yh_norm, 1e-10f);
            float vn = (float)nm[t];
            tg_norm[0] = vn / yh_safe;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scaled = code * tg_norm[0];

        // Accumulate into each Q slot's running sum, gated by its weight.
        // Per-r aliveness check: even if any_alive=true above, only
        // contribute the weight-gated value. This keeps the result
        // deterministic regardless of the any_alive short-circuit.
        for (int r = 0; r < RepeatCount; r++) {
            float w = wt[r * t_count + t];
            if (w > thresh) {
                acc[r] += w * scaled;
            }
        }
    }

    // Write D-aligned output. `out_shape` is [B*Hq, Tq, D].
    if (d < (uint)Dim) {
        for (int r = 0; r < RepeatCount; r++) {
            out[(bhq * (uint)RepeatCount + (uint)r) * (uint)Dim + d] = acc[r];
        }
    }
)";

// Thread-safe lazy-initialised holder for the JIT-compiled kernel.
//
// The server runs each request on its own `tokio::task::spawn_blocking`
// worker, so concurrent first-use of the kernel is reachable. The earlier
// implementation mutated `kernel` and `initialized` without synchronisation,
// which is C++ data-race UB. We use `std::call_once` here because:
// - the initializer can throw (MLX device lookup failure on a misconfigured
//   build), and `call_once` re-runs the initializer on the next call after a
//   throw, matching the documented MLX behaviour;
// - the post-init read of `kernel` is publication-safe since `call_once` is a
//   release/acquire fence.
struct SparseVKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_sparse_v_weighted_sum",
                {"weights", "norms", "packed", "threshold", "codebook"},
                {"out"},
                std::string(SPARSE_V_WEIGHTED_SUM_SOURCE));
        });
        return *kernel;
    }
};

inline SparseVKernelHolder& get_sparse_v_kernel() {
    static SparseVKernelHolder holder;
    return holder;
}

} // namespace

mlx::core::array sparse_v_weighted_sum(
    const mlx::core::array& attn_weights,
    const mlx::core::array& v_packed,
    const mlx::core::array& v_norms,
    const mlx::core::array& codebook,
    int dim,
    int n_rep,
    float threshold) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    auto& w_shape = attn_weights.shape();

    int bhq = w_shape[0];
    int tq = w_shape[1];

    auto& kernel = get_sparse_v_kernel().get();

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"Dim", dim},
        {"RepeatCount", tq},
        {"NRep", n_rep},
    };

    // Pack threshold into a 1-element f32 array. metal_kernel only takes
    // arrays for inputs (ScalarArg is reserved for the precompiled path).
    // Use a 1-D shape so the kernel side gets `constant float* threshold`
    // (size < 8 → constant address space) and `threshold[0]` works.
    auto threshold_arr = mlx::core::full(
        mlx::core::Shape{1}, threshold, mlx::core::float32);

    std::vector<mlx::core::array> inputs = {
        attn_weights,   // [B*Hq, Tq, Tk]    f32
        v_norms,        // [B*Hkv, Tk]       f16
        v_packed,       // [B*Hkv, Tk, D/2]  u8
        threshold_arr,  // [1]               f32
        codebook,       // [16]              f32
    };
    std::vector<Shape> output_shapes = {Shape{bhq, tq, dim}};
    std::vector<Dtype> output_dtypes = {mlx::core::float32};

    // Threadgroup: (Dim, 1, 1). Grid: (Dim, 1, B*Hq). Apple Silicon caps
    // threadgroup width at 1024 — for D > 1024 (rare) the caller must split
    // along D, but typical head_dim is 64/128/256 so we ignore that path.
    int tg_x = dim;
    if (tg_x > 1024) {
        tg_x = 1024;
    }
    auto results = kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(dim, 1, bhq),    // grid
        std::make_tuple(tg_x, 1, 1),     // threadgroup
        template_args,
        std::nullopt,
        false,
        {});

    return results[0];
}

} // namespace mlxcel::turbo
