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

//! Help-text consistency invariant for the TurboQuant KV-cache flag group.
//!
//! All three CLI surfaces — `mlxcel generate`, `mlxcel serve`, and
//! `mlxcel-server` — flatten the same `TurboKvCacheArgs` clap group, so the
//! `--help` output for the four shared flags MUST be identical across them.
//! These tests run each binary's `--help`, extract the
//! `KV Cache (TurboQuant) Options` block, and assert that every required
//! flag, mode value, and alias appears in all three. Inside the same block
//! they also forbid closed-repo issue/epic numbers from leaking back into
//! operator-facing help text. The forbidden-substring check is intentionally
//! scoped to the block — pre-existing references on unrelated server-side
//! flags are out of scope for this invariant.
//!
//! When a future flag or mode is added to the shared group, all three
//! binaries fail this test together — no drift is possible.

use std::process::Command;

mod common;
use common::repo_binary_path;

const HEADING: &str = "KV Cache (TurboQuant) Options";

/// Long-form flag names that MUST appear under the heading on every binary.
const EXPECTED_FLAGS: &[&str] = &[
    "--cache-type-k",
    "--cache-type-v",
    "--kv-cache-mode",
    "--turbo-boundary-v",
];

/// Mode tokens that MUST appear in the help block on every binary. The aliases
/// are part of the contract so a user reading the help on any binary can
/// discover the alternate spellings without reading the source.
const EXPECTED_MODES: &[&str] = &[
    "fp16",
    "int8",
    "fp16+turbo4",
    "fp16+turbo3",
    "turbo4",
    "turbo4-delegated",
    "turbo4-asym",
    "turbo3-asym",
    "turbo4-sym",
];

/// Substrings that MUST NOT appear within the TurboQuant KV-cache help block.
/// Closed-repo issue/epic numbers leak internal tracking IDs into the public
/// help text and have no value for end users. The check is scoped to the
/// block introduced by this PR (the four shared TurboQuant flags); pre-existing
/// references on unrelated server flags are out of scope for this invariant.
const FORBIDDEN_SUBSTRINGS: &[&str] = &["issue #", "epic #", "B-step #", "Issue #", "Epic #"];

/// Env vars that any of the binaries' clap definitions consult via
/// `#[arg(env = "...")]` AND that could materialize as `[env: NAME=value]`
/// in the rendered help block. Cleared before spawn so the test is
/// deterministic regardless of the host shell's environment (CI runners or
/// developer shells with `LLAMA_ARG_*` set otherwise leak values into help
/// output and could in theory trip the forbidden-substring check).
///
/// Only the env vars consumed by `TurboKvCacheArgs` itself need to be
/// cleared — the broader llama-server compatibility env vars are scoped to
/// other flag groups whose help isn't asserted here.
const ENV_TO_CLEAR_FOR_HELP: &[&str] = &["LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V"];

/// Run `--help` on a binary and return the resulting stdout. Panics with a
/// descriptive message when the binary fails to execute.
fn help_output(bin_name: &str, args: &[&str]) -> String {
    let path = repo_binary_path(bin_name);
    let mut cmd = Command::new(&path);
    cmd.args(args);
    for key in ENV_TO_CLEAR_FOR_HELP {
        cmd.env_remove(key);
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {} from {:?}: {e}", bin_name, path));
    assert!(
        output.status.success(),
        "{} {:?} exited with status {:?}: stderr=\n{}",
        bin_name,
        args,
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Slice the help output to the `KV Cache (TurboQuant) Options` block:
/// from the heading line to the next help heading or end of input.
///
/// clap renders section headings on their own line ending in `:` followed
/// by an indented list of flag entries. The next heading is the next
/// non-indented line that ends with `:` (e.g. `Options:`, `Generation
/// Options:`). For the invariant tests we only need to ensure the block
/// contains the relevant flags, so an over-inclusive cut is fine — better
/// to fail loud than to silently miss content.
fn extract_kv_cache_block(help: &str) -> &str {
    let start = help
        .find(HEADING)
        .unwrap_or_else(|| panic!("help heading {HEADING:?} not found in:\n{help}"));

    // Walk lines after the heading and find the byte offset of the next
    // section heading. clap headings are non-indented (do not start with a
    // space) and end with `:`.
    let after_heading_offset = help[start..]
        .find('\n')
        .map(|i| start + i + 1)
        .unwrap_or(help.len());

    let mut cursor = after_heading_offset;
    let mut end = help.len();
    for line in help[after_heading_offset..].split_inclusive('\n') {
        let trimmed = line.trim_end_matches('\n');
        let is_section_heading = trimmed.ends_with(':')
            && !trimmed.starts_with(' ')
            && !trimmed.contains("--")
            && trimmed.chars().next().is_some_and(|c| c.is_uppercase())
            && trimmed.len() < 80;
        if is_section_heading {
            end = cursor;
            break;
        }
        cursor += line.len();
    }
    &help[start..end]
}

/// Assert every required flag, mode, and absence of forbidden substrings
/// for one binary's help block.
fn assert_invariants(label: &str, help: &str) {
    let block = extract_kv_cache_block(help);

    for flag in EXPECTED_FLAGS {
        assert!(
            block.contains(flag),
            "{label}: KV Cache help block is missing flag {flag:?}.\n\
             Block was:\n{block}"
        );
    }

    for mode in EXPECTED_MODES {
        assert!(
            block.contains(mode),
            "{label}: KV Cache help block is missing mode token {mode:?}.\n\
             Block was:\n{block}"
        );
    }

    for forbidden in FORBIDDEN_SUBSTRINGS {
        if let Some(idx) = block.find(forbidden) {
            // Multi-byte chars (em-dashes etc.) would panic on raw byte
            // slicing, so use char-boundary-safe windowing for the error
            // context.
            let window_start = idx.saturating_sub(40);
            let window_end = (idx + forbidden.len() + 40).min(block.len());
            let safe_start = (0..=window_start)
                .rev()
                .find(|&i| block.is_char_boundary(i))
                .unwrap_or(0);
            let safe_end = (window_end..=block.len())
                .find(|&i| block.is_char_boundary(i))
                .unwrap_or(block.len());
            panic!(
                "{label}: KV Cache help block contains forbidden closed-repo \
                 reference {forbidden:?}. Move the reference into a non-doc \
                 `//` comment in src/cli/turbo_args.rs or remove it.\n\
                 Match context: {:?}",
                &block[safe_start..safe_end]
            );
        }
    }
}

#[test]
fn mlxcel_generate_help_lists_all_turbo_flags_and_modes() {
    let help = help_output("mlxcel", &["generate", "--help"]);
    assert_invariants("mlxcel generate", &help);
}

#[test]
fn mlxcel_serve_help_lists_all_turbo_flags_and_modes() {
    let help = help_output("mlxcel", &["serve", "--help"]);
    assert_invariants("mlxcel serve", &help);
}

#[test]
fn mlxcel_server_help_lists_all_turbo_flags_and_modes() {
    let help = help_output("mlxcel-server", &["--help"]);
    assert_invariants("mlxcel-server", &help);
}

/// Cross-binary equivalence: the four shared flags should appear with the
/// same names and same value-name in every binary's help block. We do NOT
/// require byte-identical blocks because clap interleaves binary-specific
/// flags around the heading boundary on `mlxcel-server` (no-subcommand
/// invocation). Instead, we assert each flag's "long-form + value-name"
/// pair appears identically.
#[test]
fn turbo_flag_signatures_match_across_binaries() {
    let generate_help = help_output("mlxcel", &["generate", "--help"]);
    let serve_help = help_output("mlxcel", &["serve", "--help"]);
    let server_help = help_output("mlxcel-server", &["--help"]);

    let signatures = [
        "--cache-type-k <TYPE>",
        "--cache-type-v <TYPE>",
        "--kv-cache-mode <MODE>",
        "--turbo-boundary-v <COUNT>",
    ];
    for sig in signatures {
        assert!(
            generate_help.contains(sig),
            "mlxcel generate is missing flag signature {sig:?}"
        );
        assert!(
            serve_help.contains(sig),
            "mlxcel serve is missing flag signature {sig:?}"
        );
        assert!(
            server_help.contains(sig),
            "mlxcel-server is missing flag signature {sig:?}"
        );
    }
}
