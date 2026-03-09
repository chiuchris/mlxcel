use super::{RuntimeDevice, parse_memory_size, parse_runtime_device, resolve_runtime_device};

#[test]
fn parse_runtime_device_accepts_cpu() {
    assert_eq!(parse_runtime_device("cpu"), Some(RuntimeDevice::Cpu));
}

#[test]
fn parse_runtime_device_accepts_gpu_aliases() {
    assert_eq!(parse_runtime_device("gpu"), Some(RuntimeDevice::Gpu));
    assert_eq!(parse_runtime_device("Metal"), Some(RuntimeDevice::Gpu));
}

#[test]
fn parse_runtime_device_rejects_unknown_values() {
    assert_eq!(parse_runtime_device("tpu"), None);
}

#[test]
fn resolve_runtime_device_defaults_to_gpu() {
    assert_eq!(resolve_runtime_device(None), (RuntimeDevice::Gpu, None));
}

#[test]
fn resolve_runtime_device_preserves_invalid_override() {
    assert_eq!(
        resolve_runtime_device(Some("mps")),
        (RuntimeDevice::Gpu, Some("mps".to_string()))
    );
}

#[test]
fn parse_memory_size_gb() {
    assert_eq!(parse_memory_size("64GB"), Some(64 * 1024 * 1024 * 1024));
    assert_eq!(parse_memory_size("128gb"), Some(128 * 1024 * 1024 * 1024));
}

#[test]
fn parse_memory_size_mb() {
    assert_eq!(parse_memory_size("512MB"), Some(512 * 1024 * 1024));
}

#[test]
fn parse_memory_size_bytes() {
    assert_eq!(parse_memory_size("1073741824"), Some(1073741824));
}

#[test]
fn parse_memory_size_fractional_gb() {
    // 1.5 GB
    assert_eq!(
        parse_memory_size("1.5GB"),
        Some((1.5 * 1024.0 * 1024.0 * 1024.0) as usize)
    );
}

#[test]
fn parse_memory_size_invalid() {
    assert_eq!(parse_memory_size("abc"), None);
}
