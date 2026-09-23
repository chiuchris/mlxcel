cmake_minimum_required(VERSION 3.16)

project(mlx-cpp LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)

# ----------------------------- Fetch MLX C++ -----------------------------

include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(MLX_BUILD_TESTS OFF)
set(MLX_BUILD_EXAMPLES OFF)
set(MLX_BUILD_BENCHMARKS OFF)
set(MLX_BUILD_PYTHON_BINDINGS OFF)

function(mlx_apply_source_overlays mlx_source_dir)
  # Metal patches: JIT type promotion fix + kernel utils (SIMD group matrix).
  #
  # The SDPA / steel-attention overlays (scaled_dot_product_attention.cpp,
  # scaled_dot_product_attention.metal, steel_attention.metal,
  # steel_attention_nax.metal) were dropped when the MLX pin moved to
  # v0.31.2 — that release already carries the NAX fixes we had been
  # carrying locally (`#3361 int16 overflow in SDPA NAX mask indexing`,
  # `#3422 Speed up NAX split-K`, `#3419 Segmented mm nax kernel`). Removing
  # the patches keeps our build binary-equivalent to the upstream Python
  # wheel and simplifies future upstream syncs.
  # The pinned Metal sources still need address-space compatibility overlays
  # below for the Metal 4.1 SDK. They qualify thread-local tile methods without
  # changing kernel operations or compile-time iteration.
  set(_metal_patch_files
      "mlx/backend/metal/compiled.cpp"
      "mlx/backend/metal/kernels/utils.h")
  foreach(_patch_file ${_metal_patch_files})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}")
      configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}"
        "${mlx_source_dir}/${_patch_file}"
        COPYONLY)
    endif()
  endforeach()
  set(_nax_header "${mlx_source_dir}/mlx/backend/metal/kernels/steel/gemm/nax.h")
  if(NOT EXISTS "${_nax_header}")
    message(FATAL_ERROR "Pinned MLX NAX header not found: ${_nax_header}")
  endif()
  file(READ "${_nax_header}" _nax_source)
  set(_nax_load_original [=[
  template <typename U, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }
]=])
  set(_nax_load_previous [=[
  template <typename U, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::load(
            frag_at(i, j),
            src,
            Int<str_x>{},
            Int<str_y>{},
            i * kFragRows,
            j * kFragCols);
      }
    }
  }
]=])
  set(_nax_load_pointer [=[
  template <typename U, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    auto* tile = this;
    const_for_loop<0, kTileRows, 1>([tile](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([tile, idx_row](auto idx_col) {
        NAXFrag_t::load(
            tile->template frag_at<idx_row.value, idx_col.value>(),
            src,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }
]=])
  set(_nax_overlay_changed FALSE)
  foreach(_old_load IN ITEMS "${_nax_load_previous}" "${_nax_load_pointer}")
    string(FIND "${_nax_source}" "${_old_load}" _old_load_index)
    if(NOT _old_load_index EQUAL -1)
      string(REPLACE "${_old_load}" "${_nax_load_original}"
             _nax_source "${_nax_source}")
      set(_nax_overlay_changed TRUE)
    endif()
  endforeach()

  string(FIND "${_nax_source}" "struct NAXTile {" _nax_start)
  if(_nax_start EQUAL -1)
    message(FATAL_ERROR "Pinned MLX NAXTile declaration not found")
  endif()
  string(SUBSTRING "${_nax_source}" ${_nax_start} -1 _nax_remainder)
  string(FIND "${_nax_remainder}" "\n};" _nax_end)
  if(_nax_end EQUAL -1)
    message(FATAL_ERROR "Pinned MLX NAXTile declaration is unterminated")
  endif()
  math(EXPR _nax_block_length "${_nax_end} + 3")
  string(SUBSTRING "${_nax_remainder}" 0 ${_nax_block_length} _nax_block)
  string(SUBSTRING "${_nax_remainder}" ${_nax_block_length} -1 _nax_tail)
  string(SUBSTRING "${_nax_source}" 0 ${_nax_start} _nax_prefix)

  set(_nax_signature_pairs
      "clear() {|clear() thread {"
      "frag_at(const short i, const short j) {|frag_at(const short i, const short j) thread {"
      "const short j) const {|const short j) const thread {"
      "frag_at() const {|frag_at() const thread {"
      "frag_at() {|frag_at() thread {"
      "metal::bool_constant<transpose>) const {|metal::bool_constant<transpose>) const thread {"
      "metal::bool_constant<transpose>) {|metal::bool_constant<transpose>) thread {"
      "elems() const {|elems() const thread {"
      "elems() {|elems() thread {"
      "kRowsPerThread>& vals) const {|kRowsPerThread>& vals) const thread {"
      "kRowsPerThread>& vals) {|kRowsPerThread>& vals) thread {"
      "load(const threadgroup U* src) {|load(const threadgroup U* src) thread {"
      "store(threadgroup U* dst) const {|store(threadgroup U* dst) const thread {"
      "load(const device U* src, const int ld) {|load(const device U* src, const int ld) thread {"
      "store(device U* dst, const int ld) const {|store(device U* dst, const int ld) const thread {"
      "load_rows(const device U* src, const int ld, const short n_rows) {|load_rows(const device U* src, const int ld, const short n_rows) thread {"
      "load_safe(const device U* src, const int ld, const short2 src_tile_dims) {|load_safe(const device U* src, const int ld, const short2 src_tile_dims) thread {"
      "const short n_rows)\n      const {|const short n_rows)\n      const thread {"
      "store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {|store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const thread {"
      "const short2 stop) const {|const short2 stop) const thread {")
  foreach(_signature_pair IN LISTS _nax_signature_pairs)
    string(REPLACE "|" ";" _signature_parts "${_signature_pair}")
    list(GET _signature_parts 0 _old_signature)
    list(GET _signature_parts 1 _new_signature)
    string(FIND "${_nax_block}" "${_old_signature}" _old_signature_index)
    if(NOT _old_signature_index EQUAL -1)
      string(REPLACE "${_old_signature}" "${_new_signature}"
             _nax_block "${_nax_block}")
      set(_nax_overlay_changed TRUE)
    else()
      string(FIND "${_nax_block}" "${_new_signature}" _new_signature_index)
      if(_new_signature_index EQUAL -1)
        message(FATAL_ERROR "Pinned MLX NAXTile signature did not match overlay: ${_old_signature}")
      endif()
    endif()
  endforeach()

  if(_nax_overlay_changed)
    set(_nax_source "${_nax_prefix}${_nax_block}${_nax_tail}")
    file(WRITE "${_nax_header}" "${_nax_source}")
    message(STATUS "Applied Metal NAX thread-address-space qualifiers")
  endif()

  set(_attn_nax_header "${mlx_source_dir}/mlx/backend/metal/kernels/steel/attn/nax.h")
  if(NOT EXISTS "${_attn_nax_header}")
    message(FATAL_ERROR "Pinned MLX attention NAX header not found")
  endif()
  file(READ "${_attn_nax_header}" _attn_nax_source)
  string(FIND "${_attn_nax_source}" "struct NAXTile {" _attn_nax_start)
  if(_attn_nax_start EQUAL -1)
    message(FATAL_ERROR "Pinned MLX attention NAXTile declaration not found")
  endif()
  string(SUBSTRING "${_attn_nax_source}" ${_attn_nax_start} -1 _attn_nax_remainder)
  string(FIND "${_attn_nax_remainder}" "\n};" _attn_nax_end)
  if(_attn_nax_end EQUAL -1)
    message(FATAL_ERROR "Pinned MLX attention NAXTile declaration is unterminated")
  endif()
  math(EXPR _attn_nax_block_length "${_attn_nax_end} + 3")
  string(SUBSTRING "${_attn_nax_remainder}" 0 ${_attn_nax_block_length} _attn_nax_block)
  string(SUBSTRING "${_attn_nax_remainder}" ${_attn_nax_block_length} -1 _attn_nax_tail)
  string(SUBSTRING "${_attn_nax_source}" 0 ${_attn_nax_start} _attn_nax_prefix)
  set(_attn_nax_signature_pairs
      "clear() {|clear() thread {"
      "const short j) {|const short j) thread {"
      "const short j) const {|const short j) const thread {"
      "frag_at() const {|frag_at() const thread {"
      "frag_at() {|frag_at() thread {"
      "metal::bool_constant<transpose>) const {|metal::bool_constant<transpose>) const thread {"
      "metal::bool_constant<transpose>) {|metal::bool_constant<transpose>) thread {"
      "bool transpose>\n  METAL_FUNC constexpr const thread frag_type& frag_at() const {|bool transpose>\n  METAL_FUNC constexpr const thread frag_type& frag_at() const thread {"
      "bool transpose>\n  METAL_FUNC constexpr thread frag_type& frag_at() {|bool transpose>\n  METAL_FUNC constexpr thread frag_type& frag_at() thread {"
      "elems() const {|elems() const thread {"
      "elems() {|elems() thread {"
      "kRowsPerThread>& vals) const {|kRowsPerThread>& vals) const thread {"
      "kRowsPerThread>& vals) {|kRowsPerThread>& vals) thread {"
      "load(const threadgroup U* src) {|load(const threadgroup U* src) thread {"
      "store(threadgroup U* dst) const {|store(threadgroup U* dst) const thread {"
      "load(const device U* src, const int ld) {|load(const device U* src, const int ld) thread {"
      "store(device U* dst, const int ld) const {|store(device U* dst, const int ld) const thread {"
      "const short n_rows)\n      { {|const short n_rows)\n      thread {"
      "const short n_rows)\n      const {|const short n_rows)\n      const thread {"
      "const short n_rows) {|const short n_rows) thread {"
      "const short2 src_tile_dims) {|const short2 src_tile_dims) thread {"
      "const short2 dst_tile_dims) const {|const short2 dst_tile_dims) const thread {"
      "const short2 stop) const {|const short2 stop) const thread {")
  set(_attn_nax_changed FALSE)
  foreach(_signature_pair IN LISTS _attn_nax_signature_pairs)
    string(REPLACE "|" ";" _signature_parts "${_signature_pair}")
    list(GET _signature_parts 0 _old_signature)
    list(GET _signature_parts 1 _new_signature)
    string(FIND "${_attn_nax_block}" "${_old_signature}" _old_signature_index)
    if(NOT _old_signature_index EQUAL -1)
      string(REPLACE "${_old_signature}" "${_new_signature}"
             _attn_nax_block "${_attn_nax_block}")
      set(_attn_nax_changed TRUE)
    endif()
  endforeach()
  set(_attn_nax_loop_marker "  template <typename U, int str_x, int str_y>")
  string(FIND "${_attn_nax_block}" "${_attn_nax_loop_marker}" _attn_nax_loop_start)
  if(_attn_nax_loop_start EQUAL -1)
    message(FATAL_ERROR "Pinned MLX attention NAXTile loop methods not found")
  endif()
  string(SUBSTRING "${_attn_nax_block}" 0 ${_attn_nax_loop_start} _attn_nax_methods_prefix)
  string(REPLACE "tile->frag_at(" "frag_at(" _attn_nax_methods_prefix "${_attn_nax_methods_prefix}")
  string(SUBSTRING "${_attn_nax_block}" ${_attn_nax_loop_start} -1 _attn_nax_loop_methods)
  set(_attn_nax_outer_loop [=[    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {]=])
  set(_attn_nax_outer_loop_patched [=[    auto* tile = this;
    const_for_loop<0, kTileRows, 1>([=](auto idx_row) {]=])
  string(FIND "${_attn_nax_loop_methods}" "${_attn_nax_outer_loop}" _attn_nax_outer_loop_index)
  if(NOT _attn_nax_outer_loop_index EQUAL -1)
    string(REPLACE "${_attn_nax_outer_loop}" "${_attn_nax_outer_loop_patched}"
           _attn_nax_loop_methods "${_attn_nax_loop_methods}")
    set(_attn_nax_inner_loop [=[      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {]=])
    set(_attn_nax_inner_loop_patched [=[      const_for_loop<0, kTileCols, 1>([=](auto idx_col) {]=])
    string(REPLACE "${_attn_nax_inner_loop}" "${_attn_nax_inner_loop_patched}"
           _attn_nax_loop_methods "${_attn_nax_loop_methods}")
    string(REPLACE "frag_at<" "tile->template frag_at<" _attn_nax_loop_methods "${_attn_nax_loop_methods}")
    string(REPLACE "frag_at(" "tile->frag_at(" _attn_nax_loop_methods "${_attn_nax_loop_methods}")
    set(_attn_nax_changed TRUE)
  else()
    set(_attn_nax_outer_loop_previous [=[    const_for_loop<0, kTileRows, 1>([tile](auto idx_row) {]=])
    set(_attn_nax_outer_loop_previous_patched [=[    const_for_loop<0, kTileRows, 1>([=](auto idx_row) {]=])
    string(REPLACE "${_attn_nax_outer_loop_previous}" "${_attn_nax_outer_loop_previous_patched}"
           _attn_nax_loop_methods "${_attn_nax_loop_methods}")
    set(_attn_nax_inner_loop_previous [=[      const_for_loop<0, kTileCols, 1>([tile, idx_row](auto idx_col) {]=])
    set(_attn_nax_inner_loop_previous_patched [=[      const_for_loop<0, kTileCols, 1>([=](auto idx_col) {]=])
    string(REPLACE "${_attn_nax_inner_loop_previous}" "${_attn_nax_inner_loop_previous_patched}"
           _attn_nax_loop_methods "${_attn_nax_loop_methods}")
    set(_attn_nax_changed TRUE)
  endif()
  set(_attn_nax_block "${_attn_nax_methods_prefix}${_attn_nax_loop_methods}")
  if(_attn_nax_changed)
    set(_attn_nax_source "${_attn_nax_prefix}${_attn_nax_block}${_attn_nax_tail}")
    file(WRITE "${_attn_nax_header}" "${_attn_nax_source}")
    message(STATUS "Applied Metal attention NAX thread-address-space qualifiers")
  endif()

  set(_attn_mma_header "${mlx_source_dir}/mlx/backend/metal/kernels/steel/attn/mma.h")
  if(NOT EXISTS "${_attn_mma_header}")
    message(FATAL_ERROR "Pinned MLX attention MMA header not found")
  endif()
  file(READ "${_attn_mma_header}" _attn_mma_source)
  string(FIND "${_attn_mma_source}" "struct MMATile {" _attn_mma_start)
  if(_attn_mma_start EQUAL -1)
    message(FATAL_ERROR "Pinned MLX attention MMATile declaration not found")
  endif()
  string(SUBSTRING "${_attn_mma_source}" ${_attn_mma_start} -1 _attn_mma_remainder)
  string(FIND "${_attn_mma_remainder}" "\n};" _attn_mma_end)
  if(_attn_mma_end EQUAL -1)
    message(FATAL_ERROR "Pinned MLX attention MMATile declaration is unterminated")
  endif()
  math(EXPR _attn_mma_block_length "${_attn_mma_end} + 3")
  string(SUBSTRING "${_attn_mma_remainder}" 0 ${_attn_mma_block_length} _attn_mma_block)
  string(SUBSTRING "${_attn_mma_remainder}" ${_attn_mma_block_length} -1 _attn_mma_tail)
  string(SUBSTRING "${_attn_mma_source}" 0 ${_attn_mma_start} _attn_mma_prefix)
  set(_attn_mma_original_block "${_attn_mma_block}")
  string(REPLACE "clear() {" "clear() thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "const short j) {" "const short j) thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "const short j) const {" "const short j) const thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "mat_at(const short i, const short j) {" "mat_at(const short i, const short j) thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "elems() {" "elems() thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "elems() const {" "elems() const thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "kRowsPerThread]) const {" "kRowsPerThread]) const thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "kRowsPerThread]) {" "kRowsPerThread]) thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "const threadgroup U* src) {" "const threadgroup U* src) thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "threadgroup U* dst) const {" "threadgroup U* dst) const thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "const device U* src, const int ld) {" "const device U* src, const int ld) thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "device U* dst, const int ld) const {" "device U* dst, const int ld) const thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "const short2 src_tile_dims) {" "const short2 src_tile_dims) thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "const short2 dst_tile_dims) const {" "const short2 dst_tile_dims) const thread {" _attn_mma_block "${_attn_mma_block}")
  string(REPLACE "short2 stop) const {" "short2 stop) const thread {" _attn_mma_block "${_attn_mma_block}")
  if(NOT _attn_mma_block STREQUAL _attn_mma_original_block)
    set(_attn_mma_changed TRUE)
  else()
    set(_attn_mma_changed FALSE)
  endif()
  if(_attn_mma_changed)
    set(_attn_mma_source "${_attn_mma_prefix}${_attn_mma_block}${_attn_mma_tail}")
    file(WRITE "${_attn_mma_header}" "${_attn_mma_source}")
    message(STATUS "Applied Metal attention MMATile thread-address-space qualifiers")
  endif()

  set(_gemm_mma_header "${mlx_source_dir}/mlx/backend/metal/kernels/steel/gemm/mma.h")
  if(NOT EXISTS "${_gemm_mma_header}")
    message(FATAL_ERROR "Pinned MLX GEMM MMA header not found")
  endif()
  file(READ "${_gemm_mma_header}" _gemm_mma_source)
  string(FIND "${_gemm_mma_source}" "struct MMATile {" _gemm_mma_start)
  if(_gemm_mma_start EQUAL -1)
    message(FATAL_ERROR "Pinned MLX GEMM MMATile declaration not found")
  endif()
  string(SUBSTRING "${_gemm_mma_source}" ${_gemm_mma_start} -1 _gemm_mma_remainder)
  string(FIND "${_gemm_mma_remainder}" "\n};" _gemm_mma_end)
  if(_gemm_mma_end EQUAL -1)
    message(FATAL_ERROR "Pinned MLX GEMM MMATile declaration is unterminated")
  endif()
  math(EXPR _gemm_mma_block_length "${_gemm_mma_end} + 3")
  string(SUBSTRING "${_gemm_mma_remainder}" 0 ${_gemm_mma_block_length} _gemm_mma_block)
  string(SUBSTRING "${_gemm_mma_remainder}" ${_gemm_mma_block_length} -1 _gemm_mma_tail)
  string(SUBSTRING "${_gemm_mma_source}" 0 ${_gemm_mma_start} _gemm_mma_prefix)
  set(_gemm_mma_original_block "${_gemm_mma_block}")
  string(REPLACE "clear() {" "clear() thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "const short j) {" "const short j) thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "const short j) const {" "const short j) const thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "mat_at(const short i, const short j) {" "mat_at(const short i, const short j) thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "elems() {" "elems() thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "elems() const {" "elems() const thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "kRowsPerThread]) const {" "kRowsPerThread]) const thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "kRowsPerThread]) {" "kRowsPerThread]) thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "const threadgroup U* src) {" "const threadgroup U* src) thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "threadgroup U* dst) const {" "threadgroup U* dst) const thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "const device U* src, const int ld) {" "const device U* src, const int ld) thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "device U* dst, const int ld) const {" "device U* dst, const int ld) const thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "const short2 src_tile_dims) {" "const short2 src_tile_dims) thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "const short2 dst_tile_dims) const {" "const short2 dst_tile_dims) const thread {" _gemm_mma_block "${_gemm_mma_block}")
  string(REPLACE "short2 stop) const {" "short2 stop) const thread {" _gemm_mma_block "${_gemm_mma_block}")
  if(NOT _gemm_mma_block STREQUAL _gemm_mma_original_block)
    set(_gemm_mma_source "${_gemm_mma_prefix}${_gemm_mma_block}${_gemm_mma_tail}")
    file(WRITE "${_gemm_mma_header}" "${_gemm_mma_source}")
    message(STATUS "Applied Metal GEMM MMATile thread-address-space qualifiers")
  endif()

  file(READ "${_gemm_mma_header}" _gemm_block_source)
  string(FIND "${_gemm_block_source}" "struct BlockMMA {" _gemm_block_start)
  if(_gemm_block_start EQUAL -1)
    message(FATAL_ERROR "Pinned MLX GEMM BlockMMA declaration not found")
  endif()
  string(SUBSTRING "${_gemm_block_source}" ${_gemm_block_start} -1 _gemm_block)
  set(_gemm_block_tail "")
  string(SUBSTRING "${_gemm_block_source}" 0 ${_gemm_block_start} _gemm_block_prefix)
  set(_gemm_block_original "${_gemm_block}")
  string(REPLACE "simd_lane_id [[thread_index_in_simdgroup]]) {" "simd_lane_id [[thread_index_in_simdgroup]]) thread {" _gemm_block "${_gemm_block}")
  string(REPLACE "const threadgroup T* Bs) {" "const threadgroup T* Bs) thread {" _gemm_block "${_gemm_block}")
  string(REPLACE "const threadgroup complex64_t* Bs) {" "const threadgroup complex64_t* Bs) thread {" _gemm_block "${_gemm_block}")
  string(REPLACE "store_result(device U* D, const int ldd) {" "store_result(device U* D, const int ldd) thread {" _gemm_block "${_gemm_block}")
  string(REPLACE "short2 start, short2 stop) {" "short2 start, short2 stop) thread {" _gemm_block "${_gemm_block}")
  string(REPLACE "short2 dst_tile_dims) {" "short2 dst_tile_dims) thread {" _gemm_block "${_gemm_block}")
  string(REPLACE "epilogue_op) {" "epilogue_op) thread {" _gemm_block "${_gemm_block}")
  string(REPLACE "thread const Epilogue& epilogue_op) const {" "thread const Epilogue& epilogue_op) const thread {" _gemm_block "${_gemm_block}")
  if(NOT _gemm_block STREQUAL _gemm_block_original)
    set(_gemm_block_source "${_gemm_block_prefix}${_gemm_block}${_gemm_block_tail}")
    file(WRITE "${_gemm_mma_header}" "${_gemm_block_source}")
    message(STATUS "Applied Metal GEMM BlockMMA thread-address-space qualifiers")
  endif()

  set(_integral_constant_header
      "${mlx_source_dir}/mlx/backend/metal/kernels/steel/utils/integral_constant.h")
  if(NOT EXISTS "${_integral_constant_header}")
    message(FATAL_ERROR "Pinned MLX integral_constant header not found")
  endif()
  file(READ "${_integral_constant_header}" _integral_constant_source)
  set(_integral_constant_original [=[
    constexpr auto res = tv __op__ uv;                      \
    return integral_constant<decltype(res), res>{};         \
]=])
  set(_integral_constant_patched [=[
    constexpr auto res = tv __op__ uv;                      \
    using res_t = metal::remove_addrspace_t<decltype(res)>; \
    return integral_constant<res_t, res>{};                 \
]=])
  string(FIND "${_integral_constant_source}" "${_integral_constant_original}"
         _integral_constant_original_index)
  if(NOT _integral_constant_original_index EQUAL -1)
    string(REPLACE "${_integral_constant_original}" "${_integral_constant_patched}"
           _integral_constant_source "${_integral_constant_source}")
    file(WRITE "${_integral_constant_header}" "${_integral_constant_source}")
    message(STATUS "Applied Metal integral-constant address-space fix")
  else()
    string(FIND "${_integral_constant_source}" "${_integral_constant_patched}"
           _integral_constant_patched_index)
    if(_integral_constant_patched_index EQUAL -1)
      message(FATAL_ERROR "Pinned MLX integral-constant operator did not match overlay")
    endif()
  endif()

  set(_nax_mpp_original [=[
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        decltype(ct_a),
        decltype(ct_b),
        CType>();
]=])
  set(_nax_mpp_patched [=[
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();
]=])
  foreach(_nax_mpp_header IN ITEMS "${_nax_header}" "${_attn_nax_header}")
    file(READ "${_nax_mpp_header}" _nax_mpp_source)
    string(FIND "${_nax_mpp_source}" "${_nax_mpp_original}" _nax_mpp_original_index)
    if(NOT _nax_mpp_original_index EQUAL -1)
      string(REPLACE "${_nax_mpp_original}" "${_nax_mpp_patched}"
             _nax_mpp_source "${_nax_mpp_source}")
      file(WRITE "${_nax_mpp_header}" "${_nax_mpp_source}")
      message(STATUS "Applied Metal NAX cooperative-tensor type fix to ${_nax_mpp_header}")
    else()
      string(FIND "${_nax_mpp_source}" "${_nax_mpp_patched}" _nax_mpp_patched_index)
      if(_nax_mpp_patched_index EQUAL -1)
        message(FATAL_ERROR "Pinned MLX NAX MPP call did not match overlay: ${_nax_mpp_header}")
      endif()
    endif()
  endforeach()

  # CUDA patches: custom grouped GEMM, quantized kernels, reduce ops, binary ops.
  # These are in patches/mlx/backend/cuda/ and only compiled for CUDA targets,
  # so they are safe to apply unconditionally (harmless on Metal-only builds).
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx/backend/cuda")
    file(GLOB_RECURSE _cuda_patch_files
         RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/patches"
         "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx/backend/cuda/*")
    foreach(_patch_file ${_cuda_patch_files})
      configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}"
        "${mlx_source_dir}/${_patch_file}"
        COPYONLY)
    endforeach()
    message(STATUS "Applied CUDA backend patches (${_cuda_patch_files})")
  endif()

  # CUDA-only patches: bf16 type promotion and norm computation changes.
  # These change dtype promotion rules (bf16+fp32→bf16) to avoid copy_v kernel
  # overhead on CUDA, but break Metal precision for bf16 vision weights
  # (e.g., SigLIP position embeddings). Apply ONLY for CUDA builds.
  if(MLX_BUILD_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches-cuda")
    file(GLOB _cuda_only_patches
         "${CMAKE_CURRENT_SOURCE_DIR}/patches-cuda/*.cpp")
    foreach(_patch_file ${_cuda_only_patches})
      get_filename_component(_fname "${_patch_file}" NAME)
      configure_file(
        "${_patch_file}"
        "${mlx_source_dir}/mlx/${_fname}"
        COPYONLY)
    endforeach()
    message(STATUS "Applied CUDA-only dtype/precision patches")
  endif()
endfunction()

set(_mlx_cached_source_dir "${CMAKE_BINARY_DIR}/_deps/mlx-src")
set(_mlx_cached_binary_dir "${CMAKE_BINARY_DIR}/_deps/mlx-build")

# An explicit source directory is useful for offline rebuilds when a previous
# Cargo target directory still has a verified MLX checkout.
if(DEFINED ENV{MLXCEL_MLX_SOURCE_DIR} AND
   EXISTS "$ENV{MLXCEL_MLX_SOURCE_DIR}/CMakeLists.txt")
  set(mlx_SOURCE_DIR "$ENV{MLXCEL_MLX_SOURCE_DIR}")
  get_filename_component(_mlx_override_deps_dir "${mlx_SOURCE_DIR}" DIRECTORY)
  set(mlx_BINARY_DIR "${_mlx_cached_binary_dir}")
  if(EXISTS "${_mlx_override_deps_dir}/metal_cpp-src/CMakeLists.txt")
    set(FETCHCONTENT_BASE_DIR "${_mlx_override_deps_dir}")
  endif()
  if(DEFINED ENV{MLXCEL_METAL_CPP_SOURCE_DIR} AND
     EXISTS "$ENV{MLXCEL_METAL_CPP_SOURCE_DIR}/Metal/Metal.hpp")
    set(FETCHCONTENT_SOURCE_DIR_METAL_CPP
        "$ENV{MLXCEL_METAL_CPP_SOURCE_DIR}")
    message(STATUS "Using Metal C++ source override from $ENV{MLXCEL_METAL_CPP_SOURCE_DIR}")
  endif()
  message(STATUS "Using MLX source override from ${mlx_SOURCE_DIR}")
# Reuse an already-populated source tree when present so offline rebuilds can
# proceed without re-running FetchContent. This is especially useful when Cargo
# changes the build hash but the developer already has a cached MLX checkout.
# NOTE: Stale-cache validation is handled by build.rs (purge_stale_mlx_cache)
# which runs *before* CMake, ensuring _deps/ is either valid or absent, and
# build.rs re-checks the fetched checkout's HEAD afterwards so a tree seeded
# out of band cannot slip through this reuse branch either.
elseif(EXISTS "${_mlx_cached_source_dir}/CMakeLists.txt")
  set(mlx_SOURCE_DIR "${_mlx_cached_source_dir}")
  set(mlx_BINARY_DIR "${_mlx_cached_binary_dir}")
  message(STATUS "Using cached MLX source from ${mlx_SOURCE_DIR}")
else()
  FetchContent_Declare(
    mlx
    GIT_REPOSITORY "https://github.com/chiuchris/mlx.git"
    GIT_TAG b7c3dd6d27f45b5365b08a840310187dc503f1db)

  # Use FetchContent_Populate + add_subdirectory so we can apply source
  # overlays before the MLX build system processes the files.
  FetchContent_GetProperties(mlx)
  if(NOT mlx_POPULATED)
    FetchContent_Populate(mlx)
  endif()
endif()

mlx_apply_source_overlays("${mlx_SOURCE_DIR}")

# Sanity-check: detect deprecated MLX Device API usage in patch files that
# would cause build failures after an upstream MLX version bump.
set(_deprecated_patterns
    "d\\.get_command_encoder("
    "d\\.new_command_buffer(")
foreach(_patch_file ${_metal_patch_files_global})
  # _metal_patch_files is function-local; re-glob here
endforeach()
file(GLOB_RECURSE _all_patches
     "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx/backend/metal/*.cpp")
foreach(_pf ${_all_patches})
  file(READ "${_pf}" _pf_content)
  foreach(_pat ${_deprecated_patterns})
    string(FIND "${_pf_content}" "${_pat}" _found)
    if(NOT _found EQUAL -1)
      message(WARNING "Patch file ${_pf} uses deprecated API '${_pat}' — "
                      "update to match upstream MLX at GIT_TAG commit.")
    endif()
  endforeach()
endforeach()

add_subdirectory(${mlx_SOURCE_DIR} ${mlx_BINARY_DIR})

# ----------------------------- CUDA CCCL ---------------------------------
# Ensure MLX_CCCL_DIR is defined for CUDA JIT kernel compilation at runtime.
# MLX only sets this for test builds, but statically linked binaries need it too.
# Since upstream e9463bb the macro is consumed by cccl_dir() in dirs.cpp, which
# is compiled in the dedicated mlx_dirs OBJECT target (upstream isolates the
# dynamic defines there to avoid invalidating the compile cache), so define it
# on mlx_dirs when that target exists and fall back to mlx otherwise.
if(MLX_BUILD_CUDA)
  set(_cccl_include "${CMAKE_BINARY_DIR}/_deps/cccl-src/include")
  if(EXISTS "${_cccl_include}")
    if(TARGET mlx_dirs)
      target_compile_definitions(mlx_dirs
                                 PRIVATE MLX_CCCL_DIR="${_cccl_include}")
    else()
      target_compile_definitions(mlx
                                 PRIVATE MLX_CCCL_DIR="${_cccl_include}")
    endif()
  endif()

  # CUTLASS/CuTe headers for the runtime-JIT'd quantized kernels (MLX #3706/
  # #3576 moved qmm_sm80/sm90/naive and gather_gemm to NVRTC JIT compilation,
  # and those kernels include <cute/...> / <cutlass/...> at runtime). Upstream
  # only installs the headers next to installed binaries; statically linked
  # mlxcel binaries need a compiled-in fallback for dev builds. Consumed by
  # our jit_module.cpp overlay, which compiles into the mlx target.
  set(_cutlass_include "${CMAKE_BINARY_DIR}/_deps/cutlass-src/include")
  if(EXISTS "${_cutlass_include}")
    target_compile_definitions(mlx
                               PRIVATE MLX_CUTLASS_DIR="${_cutlass_include}")
  endif()
endif()
