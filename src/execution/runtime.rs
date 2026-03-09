//! Runtime/device selection helpers shared by all inference entry points.
//!
//! CLI generation and the HTTP server both rely on the same environment-based
//! device resolution so CPU overrides and GPU wired-limit behavior stay
//! consistent regardless of how inference is entered.

use std::fmt;

const RUNTIME_DEVICE_ENV: &str = "MLXCEL_DEVICE";

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
        let max_memory = mlxcel_core::gpu_max_memory_size();
        mlxcel_core::set_wired_limit(max_memory);
        Some(max_memory)
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
