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
  # Apply shared patches (Metal JIT fixes, kernel utils) to all builds
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches")
    file(GLOB_RECURSE _patch_files
         RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/patches"
         "${CMAKE_CURRENT_SOURCE_DIR}/patches/*")
    foreach(_patch_file ${_patch_files})
      configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}"
        "${mlx_source_dir}/${_patch_file}"
        COPYONLY)
    endforeach()
    message(STATUS "Applied shared patches to MLX source")
  endif()

  # Apply CUDA-only patches (bf16 type promotion, normalization precision)
  # These change dtype promotion rules (bf16+fp32→bf16) and norm computation
  # to avoid copy_v kernel overhead on CUDA, but break Metal precision for
  # bf16 vision weights (e.g., SigLIP position embeddings read as zeros).
  if(MLX_BUILD_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches-cuda")
    file(GLOB _cuda_patch_files
         "${CMAKE_CURRENT_SOURCE_DIR}/patches-cuda/*.cpp")
    foreach(_patch_file ${_cuda_patch_files})
      get_filename_component(_fname "${_patch_file}" NAME)
      configure_file(
        "${_patch_file}"
        "${mlx_source_dir}/mlx/${_fname}"
        COPYONLY)
    endforeach()
    message(STATUS "Applied CUDA-only patches to MLX source")
  endif()
endfunction()

set(_mlx_cached_source_dir "${CMAKE_BINARY_DIR}/_deps/mlx-src")
set(_mlx_cached_binary_dir "${CMAKE_BINARY_DIR}/_deps/mlx-build")

# Reuse an already-populated source tree when present so offline rebuilds can
# proceed without re-running FetchContent. This is especially useful when Cargo
# changes the build hash but the developer already has a cached MLX checkout.
if(EXISTS "${_mlx_cached_source_dir}/CMakeLists.txt")
  set(mlx_SOURCE_DIR "${_mlx_cached_source_dir}")
  set(mlx_BINARY_DIR "${_mlx_cached_binary_dir}")
  message(STATUS "Using cached MLX source from ${mlx_SOURCE_DIR}")
else()
  FetchContent_Declare(
    mlx
    GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
    GIT_TAG v0.31.1)

  # Use FetchContent_Populate + add_subdirectory so we can apply source
  # overlays before the MLX build system processes the files.
  FetchContent_GetProperties(mlx)
  if(NOT mlx_POPULATED)
    FetchContent_Populate(mlx)
  endif()
endif()

mlx_apply_source_overlays("${mlx_SOURCE_DIR}")
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
