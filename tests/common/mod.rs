use std::path::PathBuf;

#[allow(dead_code)]
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
pub fn repo_binary_path(name: &str) -> PathBuf {
    let env_key = format!("CARGO_BIN_EXE_{name}");
    if let Some(path) = std::env::var_os(&env_key) {
        return PathBuf::from(path);
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    manifest_dir.join("target").join(profile).join(name)
}

#[allow(dead_code)]
pub fn extract_generated_body(stdout: &str) -> Option<&str> {
    let start = stdout.rfind("Generating...\n")?;
    let start = start + "Generating...\n".len();
    let rest = &stdout[start..];
    let end = rest.find("\n\n[")?;
    Some(&rest[..end])
}
