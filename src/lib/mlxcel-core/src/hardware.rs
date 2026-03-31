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

//! Runtime Apple Silicon generation detection.
//!
//! Detects chip generation, GPU core count, memory bandwidth, and Neural
//! Accelerator availability (M5+) at runtime via `sysctlbyname`. Results are
//! cached in a `OnceLock` so detection runs exactly once per process.

use std::sync::OnceLock;

// ── Public types ──────────────────────────────────────────────────────────────

/// Apple Silicon chip generation detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AppleSiliconGen {
    M1,
    M2,
    M3,
    M4,
    M5,
    Unknown,
}

impl AppleSiliconGen {
    /// Returns true for chips that have a Neural Accelerator (M5+).
    #[inline]
    pub fn has_neural_accelerator(self) -> bool {
        matches!(self, AppleSiliconGen::M5)
    }

    /// Returns the expected Metal GPU family version (3 for M1–M4, 4 for M5+).
    #[inline]
    pub fn metal_version(self) -> u32 {
        match self {
            AppleSiliconGen::M5 => 4,
            _ => 3,
        }
    }
}

impl std::fmt::Display for AppleSiliconGen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppleSiliconGen::M1 => write!(f, "M1"),
            AppleSiliconGen::M2 => write!(f, "M2"),
            AppleSiliconGen::M3 => write!(f, "M3"),
            AppleSiliconGen::M4 => write!(f, "M4"),
            AppleSiliconGen::M5 => write!(f, "M5"),
            AppleSiliconGen::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Hardware capabilities detected at runtime.
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Apple Silicon chip generation.
    pub silicon_gen: AppleSiliconGen,
    /// Number of GPU cores (performance cluster logical CPUs as a proxy).
    pub gpu_core_count: u32,
    /// True for M5+ chips which have a dedicated Neural Accelerator.
    pub has_neural_accelerator: bool,
    /// Metal GPU family version (3 for M1–M4, 4 for M5+).
    pub metal_version: u32,
    /// True when running on macOS 26.2+ (required to use the Neural Accelerator).
    pub macos_supports_na: bool,
    /// Approximate memory bandwidth in GB/s (estimated from chip generation).
    pub memory_bandwidth_gbps: f64,
    /// Unified memory size in GB.
    pub unified_memory_gb: u32,
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self {
            silicon_gen: AppleSiliconGen::Unknown,
            gpu_core_count: 0,
            has_neural_accelerator: false,
            metal_version: 3,
            macos_supports_na: false,
            memory_bandwidth_gbps: 0.0,
            unified_memory_gb: 0,
        }
    }
}

// ── Process-level singleton ───────────────────────────────────────────────────

static HARDWARE_CAPABILITIES: OnceLock<HardwareCapabilities> = OnceLock::new();

/// Return a reference to the cached `HardwareCapabilities`.
///
/// Detection runs at most once per process; subsequent calls return the cached
/// result immediately.
#[inline]
pub fn get_hardware() -> &'static HardwareCapabilities {
    HARDWARE_CAPABILITIES.get_or_init(detect_hardware)
}

// ── Detection ─────────────────────────────────────────────────────────────────

/// Detect hardware capabilities by querying the OS at runtime.
///
/// On non-macOS platforms this always returns [`HardwareCapabilities::default`].
pub fn detect_hardware() -> HardwareCapabilities {
    #[cfg(target_os = "macos")]
    {
        detect_hardware_macos()
    }

    #[cfg(not(target_os = "macos"))]
    {
        HardwareCapabilities::default()
    }
}

// ── macOS implementation ──────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod platform {
    use std::os::raw::{c_char, c_int, c_void};

    // sysctlbyname(3) is in <sys/sysctl.h> — link against libSystem automatically.
    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// Read a NUL-terminated string sysctl value.
    pub fn sysctl_string(name: &str) -> Option<String> {
        let c_name = std::ffi::CString::new(name).ok()?;
        // First call: query required buffer length.
        let mut len: usize = 0;
        // SAFETY: `c_name` is a valid NUL-terminated C string. We pass null for
        // `oldp` with a valid `&mut len` to query the required buffer size.
        // `newp` is null and `newlen` is 0 (read-only).
        let rc = unsafe {
            sysctlbyname(
                c_name.as_ptr(),
                std::ptr::null_mut(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 || len == 0 {
            return None;
        }
        // Second call: fill buffer.
        let mut buf = vec![0u8; len];
        // SAFETY: `buf` is a valid, writable buffer of `len` bytes. `len` is
        // updated by the kernel to reflect the actual bytes written (at most
        // the original buffer size).
        let rc = unsafe {
            sysctlbyname(
                c_name.as_ptr(),
                buf.as_mut_ptr() as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            return None;
        }
        // Truncate to the actual length returned by the kernel (may be shorter
        // than the original allocation if the value shrank between calls).
        buf.truncate(len);
        // Trim trailing NUL bytes.
        while buf.last() == Some(&0) {
            buf.pop();
        }
        String::from_utf8(buf).ok()
    }

    /// Read a `u64` sysctl value.
    pub fn sysctl_u64(name: &str) -> Option<u64> {
        let c_name = std::ffi::CString::new(name).ok()?;
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: `c_name` is a valid NUL-terminated C string. `value` is a
        // properly aligned `u64` with `len` set to its exact size. The kernel
        // writes at most `len` bytes into `value`.
        let rc = unsafe {
            sysctlbyname(
                c_name.as_ptr(),
                &mut value as *mut u64 as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 || len != std::mem::size_of::<u64>() {
            return None;
        }
        Some(value)
    }

    /// Read a `u32` sysctl value.
    pub fn sysctl_u32(name: &str) -> Option<u32> {
        let c_name = std::ffi::CString::new(name).ok()?;
        let mut value: u32 = 0;
        let mut len = std::mem::size_of::<u32>();
        // SAFETY: `c_name` is a valid NUL-terminated C string. `value` is a
        // properly aligned `u32` with `len` set to its exact size. The kernel
        // writes at most `len` bytes into `value`.
        let rc = unsafe {
            sysctlbyname(
                c_name.as_ptr(),
                &mut value as *mut u32 as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 || len != std::mem::size_of::<u32>() {
            return None;
        }
        Some(value)
    }
}

#[cfg(target_os = "macos")]
fn detect_hardware_macos() -> HardwareCapabilities {
    use platform::{sysctl_string, sysctl_u32, sysctl_u64};

    // ── Chip generation ───────────────────────────────────────────────────────
    let brand = sysctl_string("machdep.cpu.brand_string").unwrap_or_default();
    let silicon_gen = parse_silicon_gen(&brand);

    // ── GPU core count ────────────────────────────────────────────────────────
    // `hw.perflevel0.logicalcpu` is the performance-cluster CPU count, which
    // closely correlates with GPU core count on Apple Silicon.
    let gpu_core_count = sysctl_u32("hw.perflevel0.logicalcpu").unwrap_or(0);

    // ── Unified memory ────────────────────────────────────────────────────────
    let mem_bytes = sysctl_u64("hw.memsize").unwrap_or(0);
    let unified_memory_gb = (mem_bytes / (1024 * 1024 * 1024)) as u32;

    // ── macOS version ─────────────────────────────────────────────────────────
    let macos_version_str = sysctl_string("kern.osproductversion").unwrap_or_default();
    let (macos_major, macos_minor) = parse_macos_version(&macos_version_str);
    // Neural Accelerator API requires macOS 26.2+
    let macos_supports_na = (macos_major > 26) || (macos_major == 26 && macos_minor >= 2);

    // ── Derived fields ────────────────────────────────────────────────────────
    let has_neural_accelerator = silicon_gen.has_neural_accelerator();
    let metal_version = silicon_gen.metal_version();
    let memory_bandwidth_gbps = estimate_bandwidth(silicon_gen, unified_memory_gb);

    HardwareCapabilities {
        silicon_gen,
        gpu_core_count,
        has_neural_accelerator,
        metal_version,
        macos_supports_na,
        memory_bandwidth_gbps,
        unified_memory_gb,
    }
}

/// Parse Apple Silicon generation from the CPU brand string.
///
/// Examples:
/// - `"Apple M1"` → `M1`
/// - `"Apple M4 Max"` → `M4`
/// - `"Apple M5 Pro"` → `M5`
fn parse_silicon_gen(brand: &str) -> AppleSiliconGen {
    // Look for "Apple M<n>" pattern anywhere in the string.
    if let Some(pos) = brand.find("Apple M") {
        let rest = &brand[pos + "Apple M".len()..];
        // Extract the full generation number (handles multi-digit like M10+).
        let gen_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        match gen_str.as_str() {
            "1" => AppleSiliconGen::M1,
            "2" => AppleSiliconGen::M2,
            "3" => AppleSiliconGen::M3,
            "4" => AppleSiliconGen::M4,
            "5" => AppleSiliconGen::M5,
            _ => AppleSiliconGen::Unknown,
        }
    } else {
        AppleSiliconGen::Unknown
    }
}

/// Parse `"major.minor[.patch]"` version strings.
///
/// Returns `(major, minor)`. Returns `(0, 0)` on parse failure.
fn parse_macos_version(version: &str) -> (u32, u32) {
    let mut parts = version.split('.');
    let major = parts
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let minor = parts
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    (major, minor)
}

/// Return approximate memory bandwidth in GB/s based on chip generation and
/// memory configuration.  Values are midpoints from Apple's published specs
/// for the base chip variant at the given memory size.
fn estimate_bandwidth(gen: AppleSiliconGen, memory_gb: u32) -> f64 {
    match gen {
        AppleSiliconGen::M1 => {
            if memory_gb > 16 {
                400.0 // M1 Max / Ultra
            } else {
                68.25 // M1 base / Pro
            }
        }
        AppleSiliconGen::M2 => {
            if memory_gb > 24 {
                800.0 // M2 Ultra
            } else if memory_gb > 16 {
                400.0 // M2 Max
            } else {
                100.0 // M2 base / Pro
            }
        }
        AppleSiliconGen::M3 => {
            if memory_gb > 36 {
                800.0 // M3 Ultra
            } else if memory_gb > 18 {
                400.0 // M3 Max
            } else {
                100.0 // M3 base / Pro
            }
        }
        AppleSiliconGen::M4 => {
            if memory_gb > 64 {
                800.0 // M4 Ultra
            } else if memory_gb > 32 {
                546.0 // M4 Max
            } else {
                120.0 // M4 base / Pro
            }
        }
        AppleSiliconGen::M5 => {
            // Estimated; update once Apple publishes official specs.
            if memory_gb > 64 {
                1000.0
            } else {
                150.0
            }
        }
        AppleSiliconGen::Unknown => 0.0,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_brand_strings() {
        assert_eq!(parse_silicon_gen("Apple M1"), AppleSiliconGen::M1);
        assert_eq!(parse_silicon_gen("Apple M2 Pro"), AppleSiliconGen::M2);
        assert_eq!(parse_silicon_gen("Apple M3 Max"), AppleSiliconGen::M3);
        assert_eq!(parse_silicon_gen("Apple M4"), AppleSiliconGen::M4);
        assert_eq!(parse_silicon_gen("Apple M5 Pro"), AppleSiliconGen::M5);
        assert_eq!(parse_silicon_gen("Intel Core i9"), AppleSiliconGen::Unknown);
        assert_eq!(parse_silicon_gen(""), AppleSiliconGen::Unknown);
        // Multi-digit generation should not mis-parse as single digit.
        assert_eq!(parse_silicon_gen("Apple M10 Pro"), AppleSiliconGen::Unknown);
    }

    #[test]
    fn parse_macos_versions() {
        assert_eq!(parse_macos_version("14.5"), (14, 5));
        assert_eq!(parse_macos_version("26.2.0"), (26, 2));
        assert_eq!(parse_macos_version("26.2"), (26, 2));
        assert_eq!(parse_macos_version("27.0"), (27, 0));
        assert_eq!(parse_macos_version(""), (0, 0));
    }

    #[test]
    fn neural_accelerator_flag() {
        assert!(!AppleSiliconGen::M4.has_neural_accelerator());
        assert!(AppleSiliconGen::M5.has_neural_accelerator());
        assert!(!AppleSiliconGen::Unknown.has_neural_accelerator());
    }

    #[test]
    fn metal_version_by_gen() {
        assert_eq!(AppleSiliconGen::M1.metal_version(), 3);
        assert_eq!(AppleSiliconGen::M4.metal_version(), 3);
        assert_eq!(AppleSiliconGen::M5.metal_version(), 4);
    }

    #[test]
    fn macos_na_threshold() {
        // Boundary conditions for "macOS 26.2+" check.
        assert!(!{ let (maj, min) = parse_macos_version("26.1"); (maj > 26) || (maj == 26 && min >= 2) });
        assert!({ let (maj, min) = parse_macos_version("26.2"); (maj > 26) || (maj == 26 && min >= 2) });
        assert!({ let (maj, min) = parse_macos_version("27.0"); (maj > 26) || (maj == 26 && min >= 2) });
    }

    #[test]
    fn detect_hardware_does_not_panic() {
        // Just verify detection runs without panicking on the current machine.
        let caps = detect_hardware();
        // We cannot assert specific values (depends on the test runner's hardware),
        // but the enum must be one of the valid variants.
        let _ = format!("{}", caps.silicon_gen);
        let _ = caps.has_neural_accelerator;
    }

    #[test]
    fn get_hardware_returns_same_instance() {
        let a = get_hardware();
        let b = get_hardware();
        // Pointer equality — same cached allocation.
        assert!(std::ptr::eq(a, b));
    }
}
