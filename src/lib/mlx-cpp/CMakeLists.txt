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
  # Metal patches: JIT type promotion fix + kernel utils (SIMD group matrix)
  set(_metal_patch_files
      "mlx/backend/metal/compiled.cpp"
      "mlx/backend/metal/scaled_dot_product_attention.cpp"
      "mlx/backend/metal/kernels/scaled_dot_product_attention.metal"
      "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal"
      "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.metal"
      "mlx/backend/metal/kernels/utils.h")
  foreach(_patch_file ${_metal_patch_files})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}")
      configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}"
        "${mlx_source_dir}/${_patch_file}"
        COPYONLY)
    endif()
  endforeach()
  message(STATUS "Applied Metal kernel patches")

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
set(_mlx_expected_git_tag "6a9a121d09546fec051038535b444bc797fac045")

# Reuse an already-populated source tree when present so offline rebuilds can
# proceed without re-running FetchContent. This is especially useful when Cargo
# changes the build hash but the developer already has a cached MLX checkout.
# Verify the cached commit matches our pinned tag to avoid stale CI caches.
set(_mlx_cache_valid FALSE)
if(EXISTS "${_mlx_cached_source_dir}/CMakeLists.txt")
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${_mlx_cached_source_dir}"
    OUTPUT_VARIABLE _mlx_cached_commit
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _git_result
    ERROR_QUIET)
  if(_git_result EQUAL 0 AND _mlx_cached_commit STREQUAL _mlx_expected_git_tag)
    set(_mlx_cache_valid TRUE)
  else()
    message(STATUS "Cached MLX source at ${_mlx_cached_commit} does not match "
                   "expected ${_mlx_expected_git_tag} — re-fetching")
    file(REMOVE_RECURSE
         "${_mlx_cached_source_dir}"
         "${_mlx_cached_binary_dir}"
         "${CMAKE_BINARY_DIR}/_deps/mlx-subbuild")
  endif()
endif()

if(_mlx_cache_valid)
  set(mlx_SOURCE_DIR "${_mlx_cached_source_dir}")
  set(mlx_BINARY_DIR "${_mlx_cached_binary_dir}")
  message(STATUS "Using cached MLX source from ${mlx_SOURCE_DIR}")
else()
  FetchContent_Declare(
    mlx
    GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
    GIT_TAG ${_mlx_expected_git_tag})

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
if(MLX_BUILD_CUDA)
  set(_cccl_include "${CMAKE_BINARY_DIR}/_deps/cccl-src/include")
  if(EXISTS "${_cccl_include}")
    target_compile_definitions(mlx
                               PRIVATE MLX_CCCL_DIR="${_cccl_include}")
  endif()
endif()
