use std::path::PathBuf;

pub fn repo_model_dir(name: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let primary = manifest_dir.join("models").join(name);
    if primary.exists() {
        return primary;
    }

    let shared_checkout = manifest_dir
        .parent()
        .map(|parent| parent.join("mlxcel-internal").join("models").join(name))
        .unwrap_or(primary.clone());
    if shared_checkout.exists() {
        return shared_checkout;
    }

    primary
}

#[allow(dead_code)]
pub fn extract_generated_body(stdout: &str) -> Option<&str> {
    let start = stdout.rfind("Generating...\n")?;
    let start = start + "Generating...\n".len();
    let rest = &stdout[start..];
    let end = rest.find("\n\n[")?;
    Some(&rest[..end])
}
