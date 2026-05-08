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

//! Unit tests for the HuggingFace snapshot downloader (issue #457).

use super::*;
use std::path::PathBuf;
// Env-var-sensitive tests must serialize through the crate-wide `ENV_LOCK`
// (issue #573); without it, two test threads can call `setenv`/`getenv`
// concurrently — undefined behavior under Rust 2024's unsafe env contract.
use crate::test_support::env_lock::env_lock;

fn args(repo_id: &str) -> DownloadArgs {
    DownloadArgs {
        repo_id: repo_id.to_string(),
        local_dir: None,
        revision: None,
        token: None,
        force: false,
    }
}

#[test]
fn repo_basename_strips_owner() {
    assert_eq!(
        repo_basename("mlx-community/Qwen3-4B-4bit"),
        "Qwen3-4B-4bit"
    );
    assert_eq!(
        repo_basename("meta-llama/Llama-3.1-8B-Instruct"),
        "Llama-3.1-8B-Instruct"
    );
    assert_eq!(repo_basename("gpt2"), "gpt2");
}

#[test]
fn default_local_dir_is_models_basename() {
    let opts = DownloadOptions::from_args(&args("mlx-community/Qwen3-4B-4bit"));
    assert_eq!(
        opts.resolve_local_dir(),
        PathBuf::from("models").join("Qwen3-4B-4bit")
    );
}

#[test]
fn explicit_local_dir_is_respected() {
    let mut a = args("mlx-community/Qwen3-4B-4bit");
    a.local_dir = Some(PathBuf::from("/tmp/custom-dir"));
    let opts = DownloadOptions::from_args(&a);
    assert_eq!(opts.resolve_local_dir(), PathBuf::from("/tmp/custom-dir"));
}

#[test]
fn from_args_carries_all_fields() {
    let a = DownloadArgs {
        repo_id: "owner/repo".to_string(),
        local_dir: Some(PathBuf::from("/tmp/x")),
        revision: Some("v1".to_string()),
        token: Some("hf_xxx".to_string()),
        force: true,
    };
    let opts = DownloadOptions::from_args(&a);
    assert_eq!(opts.repo_id, "owner/repo");
    assert_eq!(opts.local_dir, Some(PathBuf::from("/tmp/x")));
    assert_eq!(opts.revision.as_deref(), Some("v1"));
    assert_eq!(opts.token.as_deref(), Some("hf_xxx"));
    assert!(opts.force);
}

#[test]
fn allow_list_includes_safetensors_and_index() {
    assert!(is_wanted_file("model.safetensors"));
    assert!(is_wanted_file("model-00001-of-00002.safetensors"));
    assert!(is_wanted_file("model.safetensors.index.json"));
}

#[test]
fn allow_list_includes_configs_and_tokenizer_files() {
    assert!(is_wanted_file("config.json"));
    assert!(is_wanted_file("generation_config.json"));
    assert!(is_wanted_file("tokenizer_config.json"));
    assert!(is_wanted_file("tokenizer.json"));
    assert!(is_wanted_file("tokenizer.model"));
    assert!(is_wanted_file("special_tokens_map.json"));
    assert!(is_wanted_file("preprocessor_config.json"));
    assert!(is_wanted_file("processor_config.json"));
    assert!(is_wanted_file("chat_template.json"));
    assert!(is_wanted_file("merges.txt"));
    assert!(is_wanted_file("vocab.json"));
    assert!(is_wanted_file("vocab.txt"));
}

#[test]
fn allow_list_includes_tiktoken() {
    assert!(is_wanted_file("tokenizer.tiktoken"));
    assert!(is_wanted_file("o200k_base.tiktoken"));
}

#[test]
fn deny_list_excludes_pytorch_and_other_formats() {
    assert!(!is_wanted_file("pytorch_model.bin"));
    assert!(!is_wanted_file("pytorch_model-00001-of-00002.bin"));
    assert!(!is_wanted_file("pytorch_model.bin.index.json"));
    assert!(!is_wanted_file("model.gguf"));
    assert!(!is_wanted_file("model.pt"));
    assert!(!is_wanted_file("model.h5"));
    assert!(!is_wanted_file("model.msgpack"));
    assert!(!is_wanted_file("model.onnx"));
}

#[test]
fn deny_list_excludes_docs_and_metadata() {
    assert!(!is_wanted_file("README.md"));
    assert!(!is_wanted_file("readme.md"));
    assert!(!is_wanted_file("LICENSE"));
    assert!(!is_wanted_file("license.txt"));
    assert!(!is_wanted_file("NOTICE"));
    assert!(!is_wanted_file("USAGE.md"));
    assert!(!is_wanted_file(".gitattributes"));
    assert!(!is_wanted_file(".huggingface/cache.lock"));
}

#[test]
fn deny_list_excludes_media() {
    assert!(!is_wanted_file("preview.png"));
    assert!(!is_wanted_file("logo.jpg"));
    assert!(!is_wanted_file("demo.mp4"));
    assert!(!is_wanted_file("sample.wav"));
}

#[test]
fn deny_list_excludes_source_code() {
    assert!(!is_wanted_file("modeling_custom.py"));
    assert!(!is_wanted_file("notebook.ipynb"));
    assert!(!is_wanted_file("config.yaml"));
    assert!(!is_wanted_file("pyproject.toml"));
}

#[test]
fn allow_list_handles_subdirectories() {
    // VLM repos sometimes ship vision configs under a subfolder.
    assert!(is_wanted_file("vision/preprocessor_config.json"));
    assert!(is_wanted_file("vision/model.safetensors"));
    assert!(!is_wanted_file("docs/USAGE.md"));
}

// ---------------------------------------------------------------------------
// Path-traversal regression tests for security finding C1 on PR #486.
//
// A malicious HuggingFace repo (or MitM) can return a sibling whose
// `rfilename` is `/etc/cron.d/evil` or `../../etc/passwd.json`. Before the
// fix the basename-only allow-list happily accepted those values and
// `PathBuf::join` silently anchored absolute paths outside `--local-dir`.
// These tests pin the rejection at the filter layer; the canonicalized
// `starts_with` guard inside `download_repo` is the defense-in-depth backup.
// ---------------------------------------------------------------------------

#[test]
fn deny_list_rejects_absolute_paths() {
    assert!(!is_wanted_file("/etc/passwd"));
    assert!(!is_wanted_file("/tmp/evil.json"));
    assert!(!is_wanted_file("/var/lib/foo.safetensors"));
}

#[test]
fn deny_list_rejects_parent_traversal() {
    assert!(!is_wanted_file("../etc/passwd.json"));
    assert!(!is_wanted_file("subdir/../../etc/passwd.json"));
    assert!(!is_wanted_file("./model.safetensors"));
    assert!(!is_wanted_file("foo/../bar.json"));
}

#[test]
fn deny_list_rejects_backslash_separators() {
    assert!(!is_wanted_file("..\\..\\evil.json"));
    assert!(!is_wanted_file("subdir\\evil.safetensors"));
}

#[test]
fn deny_list_rejects_empty_or_dot_components() {
    assert!(!is_wanted_file(""));
    assert!(!is_wanted_file("./"));
    assert!(!is_wanted_file("foo//bar.json"));
}

#[test]
fn allow_list_still_accepts_normal_subdir_paths() {
    // sentinel: ensure we didn't break legitimate nested files
    assert!(is_wanted_file("config.json"));
    assert!(is_wanted_file("subdir/tokenizer.json"));
    assert!(is_wanted_file("model-00001-of-00002.safetensors"));
}

/// Live PoC: simulate a malicious `info.siblings` payload and assert that
/// every adversarial `rfilename` is rejected by `is_wanted_file` before any
/// IO can occur. This mirrors the exploit shape the security checker
/// empirically verified on PR #486.
#[test]
fn malicious_siblings_payload_is_filtered_out() {
    let malicious_siblings = vec![
        // Absolute paths that would bypass `local_dir.join(...)` entirely.
        "/etc/passwd",
        "/etc/cron.d/evil",
        "/tmp/evil.json",
        "/home/user/.ssh/authorized_keys",
        // Parent-traversal payloads.
        "../etc/passwd.json",
        "../../../home/user/.ssh/authorized_keys",
        "../../etc/passwd.json",
        "subdir/../../etc/passwd.json",
        // Windows-style separator smuggling.
        "..\\..\\evil.json",
    ];
    for rfilename in &malicious_siblings {
        assert!(
            !is_wanted_file(rfilename),
            "is_wanted_file({rfilename:?}) must return false, got true"
        );
    }

    // And confirm a benign payload still survives the same filter pass.
    let benign = vec![
        "config.json",
        "model.safetensors",
        "vision/preprocessor_config.json",
    ];
    for rfilename in &benign {
        assert!(
            is_wanted_file(rfilename),
            "is_wanted_file({rfilename:?}) must return true, got false"
        );
    }
}

#[test]
fn token_resolution_explicit_wins() {
    let _env_guard = env_lock();
    let prev_hf = std::env::var("HF_TOKEN").ok();
    let prev_alt = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("HF_TOKEN", "from-env");
    }
    let resolved = resolve_token(Some("explicit"));
    assert_eq!(resolved.as_deref(), Some("explicit"));
    restore_env("HF_TOKEN", prev_hf);
    restore_env("HUGGING_FACE_HUB_TOKEN", prev_alt);
}

#[test]
fn token_resolution_hf_token_env() {
    let _env_guard = env_lock();
    let prev_hf = std::env::var("HF_TOKEN").ok();
    let prev_alt = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("HF_TOKEN", "tok-hf");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
    }
    let resolved = resolve_token(None);
    assert_eq!(resolved.as_deref(), Some("tok-hf"));
    restore_env("HF_TOKEN", prev_hf);
    restore_env("HUGGING_FACE_HUB_TOKEN", prev_alt);
}

#[test]
fn token_resolution_falls_back_to_alt_env() {
    let _env_guard = env_lock();
    let prev_hf = std::env::var("HF_TOKEN").ok();
    let prev_alt = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::remove_var("HF_TOKEN");
        std::env::set_var("HUGGING_FACE_HUB_TOKEN", "tok-alt");
    }
    let resolved = resolve_token(None);
    assert_eq!(resolved.as_deref(), Some("tok-alt"));
    restore_env("HF_TOKEN", prev_hf);
    restore_env("HUGGING_FACE_HUB_TOKEN", prev_alt);
}

#[test]
fn token_resolution_anonymous_when_unset() {
    let _env_guard = env_lock();
    let prev_hf = std::env::var("HF_TOKEN").ok();
    let prev_alt = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
    }
    assert_eq!(resolve_token(None), None);
    restore_env("HF_TOKEN", prev_hf);
    restore_env("HUGGING_FACE_HUB_TOKEN", prev_alt);
}

#[test]
fn token_resolution_treats_empty_as_anonymous() {
    let _env_guard = env_lock();
    let prev_hf = std::env::var("HF_TOKEN").ok();
    let prev_alt = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("HF_TOKEN", "");
        std::env::set_var("HUGGING_FACE_HUB_TOKEN", "  ");
    }
    assert_eq!(resolve_token(Some("  ")), None);
    assert_eq!(resolve_token(None), None);
    restore_env("HF_TOKEN", prev_hf);
    restore_env("HUGGING_FACE_HUB_TOKEN", prev_alt);
}

fn restore_env(key: &str, prev: Option<String>) {
    unsafe {
        match prev {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}

#[test]
fn snapshot_complete_requires_config_json() {
    let dir = tempfile::tempdir().unwrap();
    let wanted = vec!["config.json".to_string(), "model.safetensors".to_string()];
    // Missing both files.
    assert!(!snapshot_complete(dir.path(), &wanted));

    // config.json present but other file missing.
    std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
    assert!(!snapshot_complete(dir.path(), &wanted));

    // All files present and non-empty.
    std::fs::write(dir.path().join("model.safetensors"), b"shard").unwrap();
    assert!(snapshot_complete(dir.path(), &wanted));
}

#[test]
fn snapshot_complete_rejects_zero_byte_files() {
    let dir = tempfile::tempdir().unwrap();
    let wanted = vec!["config.json".to_string(), "model.safetensors".to_string()];
    std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"").unwrap();
    assert!(!snapshot_complete(dir.path(), &wanted));
}

// Integration test that hits the real Hugging Face Hub. Marked `#[ignore]`
// per the issue acceptance criteria so CI does not depend on network access.
#[test]
#[ignore]
fn live_download_smoke_test() {
    // Use a tiny repo so the test stays cheap when explicitly requested with
    // `cargo test -- --ignored`.
    let dir = tempfile::tempdir().unwrap();
    let opts = DownloadOptions {
        repo_id: "hf-internal-testing/tiny-random-gpt2".to_string(),
        local_dir: Some(dir.path().to_path_buf()),
        revision: None,
        token: None,
        force: true,
    };
    download_repo(opts).expect("live download should succeed");
    assert!(dir.path().join("config.json").exists());
}
