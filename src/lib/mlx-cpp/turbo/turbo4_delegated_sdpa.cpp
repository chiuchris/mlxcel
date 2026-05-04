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

#include "turbo4_delegated_sdpa.h"

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

// Source for the Turbo4Delegated cold-V weighted-sum Metal kernel. The string
// is the kernel BODY only; `mlx::core::fast::metal_kernel` wraps it with the
// declaration, buffer arguments, and template-argument substitution.
//
// Algorithmic contract (matches `sparse_v_sdpa.cpp` for the cold path):
//   for each token t in [0, T_cold):
//     if all attn_weights[b, q, t] ≤ threshold: skip token (per-thread)
//     code  = codebook[unpack_nibble(packed[b, t, d])]
//     scaled = code * rescale[b, t]
//     for each q in 0..Tq:
//       out[b, q, d] += attn_weights[b, q, t] * scaled
//
// The kernel returns `out_cold_pre[b, q, d]` (unrotated). The host caller
// applies `signs1 · WHT(signs2 · out_cold_pre)` to produce the rotated cold
// contribution and adds the hot V matmul contribution before returning the
// final attention output.
//
// Why this is identical to `SPARSE_V_WEIGHTED_SUM_SOURCE`:
//
// The Turbo4Delegated cache stores cold V tokens with the same Turbo4 packing
// as Turbo4Asym (4-bit indices into a 16-centroid Lloyd-Max codebook plus a
// per-token fp16 rescale; see `cache::turbo::quant`). The fused weighted-sum
// kernel only operates on cold tokens; hot tokens are read directly from
// `hot_v` via a host-graph matmul. So the per-coordinate kernel body is the
// same — only the inputs are restricted to the cold range and the host-side
// orchestration is different.
//
// We deliberately keep this as a separate source string (rather than calling
// into the sparse-V launcher) for two reasons:
// 1. **Independent kernel name** — the JIT cache key is the function name. A
//    separate name avoids accidental cross-issue cache collisions when
//    `MLX_DEBUG_KERNELS=1` dumps kernel sources.
// 2. **Independent re-validation surface** — the CLAUDE.md MLX upgrade
//    checklist treats each kernel launcher as its own re-validation unit.
//    Two launchers, two test entries, no shared mutable state.
//
// If you change one kernel body, you must consider whether the other needs
// the matching change. As of issue #528 the bodies are bit-identical.
constexpr const char* TURBO4_DELEGATED_COLD_WEIGHTED_SUM_SOURCE = R"(
    auto d = thread_position_in_grid.x;
    auto n = thread_position_in_grid.z;

    // Decode `n` into (batch * Hq) and infer the matching Hkv slot. We pass
    // the layout sizes via the input shapes; `weights_shape` is
    // [B*Hq, Tq, T_cold], `rescale_shape` is [B*Hkv, T_cold]. From these we
    // recover B*Hq and B*Hkv directly without needing extra constants.
    auto bhq = n;                              // 0 .. B*Hq-1
    auto t_count = (uint)weights_shape[2];     // T_cold
    auto bhkv_count = (uint)rescale_shape[0];  // B*Hkv

    // Grouped attention: NRep = Hq / Hkv. NRep is given as a template
    // constant. Since bhq_count / NRep == bhkv_count holds for valid
    // inputs, the per-`n` Hkv slot is `n / NRep`.
    uint nrep_u = (uint)NRep;
    uint bhkv = bhq / nrep_u;
    if (bhkv >= bhkv_count) {
        bhkv = 0;  // defensive fallback for inconsistent shapes
    }

    auto wt = weights + bhq * (uint)RepeatCount * t_count;  // [Tq, T_cold]
    auto rs = rescale + bhkv * t_count;                      // [T_cold] f16
    auto pk = packed + bhkv * t_count * (uint)(Dim / 2);     // [T_cold, D/2]

    // Threshold passed as a 1-element array so MLX accepts it as a regular
    // input buffer. (TemplateArg only supports int/bool/Dtype; ScalarArg is
    // reserved for `precompiled_*_kernel` paths, not `metal_kernel`.)
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

    // Sweep over T_cold tokens. Each token: check aliveness, look up the
    // centroid for `(t, d)`, multiply by the precomputed per-token rescale,
    // and gate by per-Q attention weight.
    //
    // No threadgroup memory and no barriers in the inner loop — the same
    // discipline issue #520 introduced for the Sparse-V kernel applies here.
    for (uint t = 0; t < t_count; t++) {
        bool any_alive = false;
        for (int r = 0; r < RepeatCount; r++) {
            if (wt[r * t_count + t] > thresh) {
                any_alive = true;
                break;
            }
        }
        if (!any_alive) {
            continue;
        }

        // Unpack nibble for this (t, d). Byte offset = byte_idx, nibble
        // selected by nibble_shift (0 for even d, 4 for odd d). Out-of-Dim
        // threads (d >= Dim) compute a benign 0 contribution; the `d < Dim`
        // bounds gate at the bottom prevents stray output writes.
        float code = 0.0f;
        if (d < (uint)Dim) {
            uint pb = (uint)pk[t * (uint)(Dim / 2) + byte_idx];
            uint idx = (pb >> nibble_shift) & 0x0Fu;
            code = codebook[idx];
        }

        // Per-token rescale `norm[t] / max(|y_hat[t]|, 1e-10)` was computed
        // on the host at quantize time (issue #520). Each thread loads the
        // same fp16 scalar; Apple's L1 / per-threadgroup cache coalesces the
        // broadcast so the bandwidth cost is effectively one read per
        // threadgroup per token.
        float rescale_t = (float)rs[t];
        float scaled = code * rescale_t;

        // Accumulate into each Q slot's running sum, gated by its weight.
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

// Thread-safe lazy-initialised holder for the JIT-compiled kernel. Same
// pattern as `SparseVKernelHolder` — the server runs each request on its own
// `tokio::task::spawn_blocking` worker, so concurrent first-use is reachable.
struct Turbo4DelegatedKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_turbo4_delegated_cold_weighted_sum",
                // Input names — must match the launcher's `inputs` vector
                // order in `turbo4_delegated_cold_weighted_sum`.
                {"weights", "rescale", "packed", "threshold", "codebook"},
                {"out"},
                std::string(TURBO4_DELEGATED_COLD_WEIGHTED_SUM_SOURCE));
        });
        return *kernel;
    }
};

inline Turbo4DelegatedKernelHolder& get_turbo4_delegated_kernel() {
    static Turbo4DelegatedKernelHolder holder;
    return holder;
}

} // namespace

mlx::core::array turbo4_delegated_cold_weighted_sum(
    const mlx::core::array& attn_weights_cold,
    const mlx::core::array& v_packed_cold,
    const mlx::core::array& v_rescale_cold,
    const mlx::core::array& codebook,
    int dim,
    int n_rep,
    float threshold) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    auto& w_shape = attn_weights_cold.shape();

    int bhq = w_shape[0];
    int tq = w_shape[1];

    auto& kernel = get_turbo4_delegated_kernel().get();

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"Dim", dim},
        {"RepeatCount", tq},
        {"NRep", n_rep},
    };

    auto threshold_arr = mlx::core::full(
        mlx::core::Shape{1}, threshold, mlx::core::float32);

    // Input order MUST match the buffer-name vector in `metal_kernel(...)`:
    // {"weights", "rescale", "packed", "threshold", "codebook"}.
    std::vector<mlx::core::array> inputs = {
        attn_weights_cold,  // [B*Hq, Tq, T_cold]    f32
        v_rescale_cold,     // [B*Hkv, T_cold]       f16
        v_packed_cold,      // [B*Hkv, T_cold, D/2]  u8
        threshold_arr,      // [1]                   f32
        codebook,           // [16]                  f32
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
