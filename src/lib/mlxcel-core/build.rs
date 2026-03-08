// Build script for mlx-cxx
// This builds the MLX C++ library and the cxx bridge

use cmake::Config;
use std::{env, path::PathBuf};

fn main() {
    let _out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Build MLX using cmake
    let mlx_dst = build_mlx();
    let mlx_include = mlx_dst.join("build/include");
    let mlx_lib = mlx_dst.join("build/lib");

    // Build the cxx bridge with optimization flags
    let mut bridge = cxx_build::bridge("src/lib.rs");
    bridge
        .file("cpp/mlx_cxx_bridge.cpp")
        .include(&mlx_include)
        .include("cpp")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-declarations");

    // Add optimization flags for release builds
    #[cfg(not(debug_assertions))]
    {
        bridge
            .flag_if_supported("-O3")
            .flag_if_supported("-DNDEBUG")
            .flag_if_supported("-flto")
            .flag_if_supported("-ffast-math")
            .flag_if_supported("-march=native")
            .flag_if_supported("-Wno-deprecated-copy");
    }

    bridge.compile("mlx_cxx_bridge");

    // Link against MLX
    println!("cargo:rustc-link-search=native={}", mlx_lib.display());
    println!("cargo:rustc-link-lib=static=mlx");

    // Platform-specific system libraries
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=dylib=objc");
        println!("cargo:rustc-link-lib=framework=Foundation");

        // Link clang compiler-rt for ___isPlatformVersionAtLeast
        // (required by MLX C++ @available() runtime checks)
        if let Ok(output) = std::process::Command::new("clang")
            .arg("--print-runtime-dir")
            .output()
        {
            let runtime_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !runtime_dir.is_empty() {
                println!("cargo:rustc-link-search=native={runtime_dir}");
                println!("cargo:rustc-link-lib=static=clang_rt.osx");
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++");
        // CPU backend needs BLAS/LAPACK on Linux
        println!("cargo:rustc-link-lib=dylib=openblas");
        println!("cargo:rustc-link-lib=dylib=lapack");
    }

    // Backend-specific linking
    #[cfg(target_os = "macos")]
    {
        // Metal and Accelerate are always linked on macOS
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    #[cfg(feature = "cuda")]
    {
        link_cuda();
    }

    // Rerun if bridge files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_bridge.h");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_bridge.cpp");
    println!("cargo:rerun-if-changed=../mlx-cpp/CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=MLX_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=MLXCEL_BUILD_METAL");
    println!("cargo:rerun-if-env-changed=MLXCEL_BUILD_ACCELERATE");
}

fn build_mlx() -> PathBuf {
    let mut config = Config::new("../mlx-cpp");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    // Platform features
    // On macOS: Metal and Accelerate are always available and enabled by default.
    // Feature flags can still override (e.g. for CPU-only testing).
    // On Linux: CPU-only by default, CUDA opt-in via feature flag.
    config.define("MLX_BUILD_CUDA", "OFF");

    #[cfg(target_os = "macos")]
    {
        let build_metal = cmake_bool_from_env("MLXCEL_BUILD_METAL").unwrap_or("ON");
        let build_accelerate = cmake_bool_from_env("MLXCEL_BUILD_ACCELERATE").unwrap_or("ON");

        // Default to Metal + Accelerate on macOS, but allow CPU-only rebuilds
        // for environments where Metal device enumeration is unavailable.
        config.define("MLX_BUILD_METAL", build_metal);
        config.define("MLX_BUILD_ACCELERATE", build_accelerate);
    }

    #[cfg(not(target_os = "macos"))]
    {
        config.define("MLX_BUILD_METAL", "OFF");
        config.define("MLX_BUILD_ACCELERATE", "OFF");
    }

    #[cfg(feature = "cuda")]
    {
        config.define("MLX_BUILD_CUDA", "ON");

        // Help CMake find CUDA toolkit
        if let Ok(cuda_path) = env::var("CUDA_HOME") {
            config.define("CMAKE_CUDA_COMPILER", format!("{cuda_path}/bin/nvcc"));
        } else if PathBuf::from("/usr/local/cuda/bin/nvcc").exists() {
            config.define("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc");
        }

        // Set CUDA architecture - use env var or auto-detect via nvidia-smi
        let cuda_arch = env::var("MLX_CUDA_ARCHITECTURES")
            .unwrap_or_else(|_| detect_cuda_arch().unwrap_or_else(|| "90".to_string()));
        config.define("MLX_CUDA_ARCHITECTURES", &cuda_arch);
    }

    config.build()
}

fn cmake_bool_from_env(name: &str) -> Option<&'static str> {
    let value = env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" => Some("ON"),
        "0" | "off" | "false" | "no" => Some("OFF"),
        _ => panic!(
            "Invalid {name} value {:?}. Expected one of: 1/0, on/off, true/false, yes/no.",
            value
        ),
    }
}

#[cfg(feature = "cuda")]
fn detect_cuda_arch() -> Option<String> {
    use std::process::Command;
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    let caps = String::from_utf8_lossy(&output.stdout);
    // Parse "X.Y" compute capabilities, convert to SM number (e.g. "9.0" -> "90")
    let archs: Vec<String> = caps
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let parts: Vec<&str> = line.split('.').collect();
            if parts.len() == 2 {
                Some(format!("{}{}", parts[0], parts[1]))
            } else {
                None
            }
        })
        .collect();
    if archs.is_empty() {
        None
    } else {
        // Deduplicate
        let mut unique = archs;
        unique.sort();
        unique.dedup();
        Some(unique.join(";"))
    }
}

#[cfg(feature = "cuda")]
fn link_cuda() {
    // Find CUDA lib directory
    let cuda_lib = if let Ok(cuda_home) = env::var("CUDA_HOME") {
        PathBuf::from(cuda_home).join("lib64")
    } else if PathBuf::from("/usr/local/cuda/lib64").exists() {
        PathBuf::from("/usr/local/cuda/lib64")
    } else {
        panic!("Cannot find CUDA library directory. Set CUDA_HOME environment variable.");
    };

    println!("cargo:rustc-link-search=native={}", cuda_lib.display());

    // CUDA runtime and math libraries
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");

    // CUDA driver API (cuLaunchKernel, cuModuleLoad, etc.)
    println!("cargo:rustc-link-lib=dylib=cuda");

    // cuDNN
    println!("cargo:rustc-link-lib=dylib=cudnn");

    // NVRTC for runtime compilation (JIT kernels)
    println!("cargo:rustc-link-lib=dylib=nvrtc");

    // CUDA stubs directory (for driver API on systems without GPU driver in lib path)
    let cuda_stubs = cuda_lib.join("stubs");
    if cuda_stubs.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_stubs.display());
    }
}
