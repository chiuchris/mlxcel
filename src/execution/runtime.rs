//! Runtime/device selection helpers shared by all inference entry points.
//!
//! CLI generation and the HTTP server both rely on the same environment-based
//! device resolution so CPU overrides and GPU wired-limit behavior stay
//! consistent regardless of how inference is entered.

use std::fmt;

const RUNTIME_DEVICE_ENV: &str = "MLXCEL_DEVICE";
const WIRED_LIMIT_ENV: &str = "MLXCEL_WIRED_LIMIT";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeDevice {
    Cpu,
    Gpu,
}

impl RuntimeDevice {
    const fn uses_gpu(self) -> bool {
        matches!(self, Self::Gpu)
    }
}

impl fmt::Display for RuntimeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu => write!(f, "gpu"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeSetup {
    pub device: RuntimeDevice,
    pub wired_limit_bytes: Option<usize>,
    pub invalid_device_override: Option<String>,
}

pub fn initialize_runtime() -> RuntimeSetup {
    let (requested_device, invalid_device_override) =
        resolve_runtime_device(std::env::var(RUNTIME_DEVICE_ENV).ok().as_deref());

    if requested_device == RuntimeDevice::Cpu {
        mlxcel_core::set_default_device(false);
    }

    let device = if mlxcel_core::is_gpu_available() {
        RuntimeDevice::Gpu
    } else {
        RuntimeDevice::Cpu
    };

    let wired_limit_bytes = if device.uses_gpu() {
        resolve_wired_limit()
    } else {
        None
    };

    RuntimeSetup {
        device,
        wired_limit_bytes,
        invalid_device_override,
    }
}

fn resolve_runtime_device(value: Option<&str>) -> (RuntimeDevice, Option<String>) {
    match value {
        Some(raw) => match parse_runtime_device(raw) {
            Some(device) => (device, None),
            None => (RuntimeDevice::Gpu, Some(raw.to_owned())),
        },
        None => (RuntimeDevice::Gpu, None),
    }
}

/// Resolve wired memory limit from MLXCEL_WIRED_LIMIT env var.
///
/// - Not set or "0": no wired limit (default, matches Python mlx-lm behavior)
/// - "max": set to gpu_max_memory_size (previous default behavior)
/// - Number (bytes) or "NGB"/"NMB": explicit limit
fn resolve_wired_limit() -> Option<usize> {
    let raw = std::env::var(WIRED_LIMIT_ENV).ok();
    let limit = match raw.as_deref() {
        None | Some("") | Some("0") => return None,
        Some("max") => mlxcel_core::gpu_max_memory_size(),
        Some(s) => parse_memory_size(s).unwrap_or(0),
    };
    if limit > 0 {
        mlxcel_core::set_wired_limit(limit);
        Some(limit)
    } else {
        None
    }
}

/// Parse a memory size string: plain bytes, "NGB", or "NMB".
fn parse_memory_size(s: &str) -> Option<usize> {
    let s = s.trim().to_ascii_uppercase();
    if let Some(n) = s.strip_suffix("GB") {
        n.trim()
            .parse::<f64>()
            .ok()
            .map(|v| (v * 1024.0 * 1024.0 * 1024.0) as usize)
    } else if let Some(n) = s.strip_suffix("MB") {
        n.trim()
            .parse::<f64>()
            .ok()
            .map(|v| (v * 1024.0 * 1024.0) as usize)
    } else {
        s.parse::<usize>().ok()
    }
}

fn parse_runtime_device(value: &str) -> Option<RuntimeDevice> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cpu" => Some(RuntimeDevice::Cpu),
        "gpu" | "metal" => Some(RuntimeDevice::Gpu),
        _ => None,
    }
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
