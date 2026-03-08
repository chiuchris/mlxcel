use super::{RuntimeDevice, parse_runtime_device, resolve_runtime_device};

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
