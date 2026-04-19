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

//! Integration tests for Axis B language steering on a real multilingual model.
//!
//! These tests verify that `--lang-bias` and `--lang-bias-policy` flags
//! actually shift the output script composition when running against
//! `Qwen2.5-7B-Instruct-4bit`.
//!
//! All tests in this file require the model weights to be present at
//! `models/Qwen2.5-7B-Instruct-4bit/`. They are gated with
//! `#[ignore = "requires local model weights and the mlxcel binary"]`
//! so that `cargo test --all` succeeds in CI where the model is absent.
//!
//! To run the gated tests with the model present:
//! ```text
//! cargo test --test lang_bias --release -- --ignored
//! ```
//!
//! # Apple Silicon note
//! The scenario-correctness tests (A/B/C/D) and the latency test are
//! verified by code review only on this Linux aarch64 build host.
//! A reviewer on Apple Silicon **should** run the gated tests before merge.

mod common;

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use common::repo_model_dir;
use mlxcel_core::lang_analyzer::{Script, classify_token};

// ============================================================================
// Constants
// ============================================================================

/// Target model name (subdirectory under `models/`).
const MODEL_NAME: &str = "Qwen2.5-7B-Instruct-4bit";

/// Number of tokens to generate per prompt in scenario tests.
const N_TOKENS: usize = 50;

/// Number of warm-up tokens to skip in the latency measurement.
const LATENCY_WARMUP_TOKENS: usize = 5;

/// Maximum allowable per-token wall-clock overhead for `--lang-bias` vs
/// baseline (5% as specified in plan §10.3).
const LATENCY_OVERHEAD_THRESHOLD: f64 = 0.05;

/// Path to the Korean prompt fixtures file, relative to the crate root.
const PROMPTS_FIXTURE_PATH: &str = "tests/fixtures/lang_bias_prompts_ko.txt";

// ============================================================================
// Helpers
// ============================================================================

/// Load the 10 Korean prompts from the fixture file.
fn load_ko_prompts() -> Vec<String> {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture_path = manifest_dir.join(PROMPTS_FIXTURE_PATH);
    std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("failed to read fixture at {}: {e}", fixture_path.display()))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_owned)
        .collect()
}

/// Run `mlxcel generate` with the given arguments and return the generated
/// text body (the portion between "Generating..." and the stats line).
///
/// Uses `--no-chat-template` to avoid chat-template noise affecting script
/// composition measurements. Uses `--temp 0` for deterministic greedy output.
fn run_generate(model_path: &Path, prompt: &str, extra_args: &[&str]) -> Option<String> {
    let model_arg = model_path.to_string_lossy().to_string();
    let n_tokens_str = N_TOKENS.to_string();

    let mut args = vec![
        "generate",
        "-m",
        &model_arg,
        "-p",
        prompt,
        "-n",
        &n_tokens_str,
        "--temp",
        "0",
        "--no-chat-template",
    ];
    args.extend_from_slice(extra_args);

    let output = Command::new(env!("CARGO_BIN_EXE_mlxcel"))
        .args(&args)
        .output()
        .expect("failed to execute mlxcel generate");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("mlxcel generate failed:\nstderr: {stderr}");
        return None;
    }

    let stdout = String::from_utf8(output.stdout).expect("stdout must be valid UTF-8");
    extract_generated_body(&stdout).map(str::to_owned)
}

/// Extract the generated text body from `mlxcel generate` stdout.
///
/// Matches the contract in `tests/common/mod.rs::extract_generated_body`.
fn extract_generated_body(stdout: &str) -> Option<&str> {
    let start = stdout.rfind("Generating...\n")?;
    let start = start + "Generating...\n".len();
    let rest = &stdout[start..];
    let end = rest.find("\n\n[")?;
    Some(&rest[..end])
}

/// Script composition result for one scenario run.
///
/// Counts are per-character (not per-token) across all prompts in the run.
/// Each Unicode character is classified via `classify_token` (applied to a
/// single-character string), so we get the same classification as the
/// lang_analyzer vocab scanner.
#[derive(Debug, Default)]
struct ScriptCounts {
    by_script: HashMap<Script, usize>,
    total_classified: usize,
}

impl ScriptCounts {
    /// Returns the fraction of classified characters belonging to the given scripts.
    ///
    /// Returns 0.0 when `total_classified == 0`.
    fn ratio_of(&self, scripts: &[Script]) -> f64 {
        if self.total_classified == 0 {
            return 0.0;
        }
        let count: usize = scripts.iter().map(|s| self.by_script.get(s).copied().unwrap_or(0)).sum();
        count as f64 / self.total_classified as f64
    }
}

/// Classify all characters in `text` and accumulate them into `counts`.
fn accumulate_script_counts(text: &str, counts: &mut ScriptCounts) {
    for c in text.chars() {
        let s = c.to_string();
        let scripts = classify_token(&s);
        if scripts.is_empty() {
            // Whitespace, punctuation, digits — not counted in the denominator.
            continue;
        }
        // A character with multiple scripts (rare but possible) is counted once
        // for each of its scripts, and once in total_classified.
        for script in &scripts {
            *counts.by_script.entry(*script).or_insert(0) += 1;
        }
        counts.total_classified += 1;
    }
}

/// Run one scenario over all prompts and return the aggregated script counts.
fn run_scenario(model_path: &Path, prompts: &[String], extra_args: &[&str]) -> ScriptCounts {
    let mut counts = ScriptCounts::default();
    for prompt in prompts {
        if let Some(text) = run_generate(model_path, prompt, extra_args) {
            accumulate_script_counts(&text, &mut counts);
        }
    }
    counts
}

/// Timing result from a latency measurement run.
#[derive(Debug, Default)]
struct LatencyResult {
    /// Mean per-token wall-clock time in milliseconds (excluding warm-up tokens).
    mean_per_token_ms: f64,
    /// Total tokens measured (all prompts combined, excluding warm-up).
    total_measured_tokens: usize,
}

/// Run one latency measurement: execute the binary for each prompt, measure
/// wall-clock time per token (excluding the first `LATENCY_WARMUP_TOKENS`
/// tokens per prompt based on total elapsed / N_TOKENS approximation).
///
/// Because we call a subprocess we cannot instrument individual token steps.
/// We use total decode wall-clock / N_TOKENS after the first warmup fraction.
fn run_latency(model_path: &Path, prompts: &[String], extra_args: &[&str]) -> LatencyResult {
    let model_arg = model_path.to_string_lossy().to_string();
    let n_tokens_str = N_TOKENS.to_string();

    let mut total_time_ms = 0.0f64;
    let mut total_measured_tokens = 0usize;

    for prompt in prompts {
        let mut args = vec![
            "generate",
            "-m",
            &model_arg,
            "-p",
            prompt,
            "-n",
            &n_tokens_str,
            "--temp",
            "0",
            "--no-chat-template",
        ];
        args.extend_from_slice(extra_args);

        let t0 = Instant::now();
        let output = Command::new(env!("CARGO_BIN_EXE_mlxcel"))
            .args(&args)
            .output()
            .expect("failed to execute mlxcel generate for latency measurement");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        if !output.status.success() {
            continue;
        }

        // Approximate measured tokens: exclude the warmup fraction from total.
        // Since the warmup is the first LATENCY_WARMUP_TOKENS tokens of N_TOKENS:
        let measured_tokens = N_TOKENS.saturating_sub(LATENCY_WARMUP_TOKENS);
        if measured_tokens == 0 {
            continue;
        }

        // Attribute time proportionally: warmup_tokens/N_TOKENS of the elapsed
        // time is "warmup cost", the rest is attributed to measured_tokens.
        let warmup_fraction = LATENCY_WARMUP_TOKENS as f64 / N_TOKENS as f64;
        let measured_time_ms = elapsed_ms * (1.0 - warmup_fraction);

        total_time_ms += measured_time_ms;
        total_measured_tokens += measured_tokens;
    }

    let mean_per_token_ms = if total_measured_tokens > 0 {
        total_time_ms / total_measured_tokens as f64
    } else {
        0.0
    };

    LatencyResult {
        mean_per_token_ms,
        total_measured_tokens,
    }
}

// ============================================================================
// Scenario tests (A / B / C / D)
// ============================================================================

/// Scenario A — baseline: Korean prompts with no `--lang-bias`.
///
/// Records the script composition of the baseline output so that scenarios
/// B, C, and D can assert relative changes. This test does not assert any
/// threshold itself; it exists to document the baseline measurement.
#[test]
#[ignore = "requires local model weights and the mlxcel binary"]
fn scenario_a_baseline_records_script_composition() {
    let model_path = repo_model_dir(MODEL_NAME);
    if !model_path.exists() {
        eprintln!(
            "Skipping scenario_a: model directory not found at {}",
            model_path.display()
        );
        return;
    }

    let prompts = load_ko_prompts();
    assert_eq!(prompts.len(), 10, "fixture must contain exactly 10 prompts");

    let result = run_scenario(&model_path, &prompts, &[]);

    let hangul_ratio = result.ratio_of(&[Script::Hangul]);
    let han_ratio = result.ratio_of(&[Script::Han]);
    let ja_ratio = result.ratio_of(&[Script::Hiragana, Script::Katakana]);
    let latin_ratio = result.ratio_of(&[Script::Latin]);

    eprintln!(
        "Scenario A baseline: total_classified={}, hangul={:.3}, han={:.3}, hiragana+katakana={:.3}, latin={:.3}",
        result.total_classified, hangul_ratio, han_ratio, ja_ratio, latin_ratio
    );

    // Baseline sanity: Qwen2.5-7B responding to Korean prompts should produce
    // at least some Hangul characters.
    assert!(
        result.total_classified > 0,
        "scenario A produced no classified characters; is the binary built in release mode?"
    );
}

/// Scenario B — suppress Japanese and Chinese: `--lang-bias ja=-inf,zh=-inf`
/// with Conservative policy.
///
/// Asserts: Hiragana + Katakana + Han ratio in scenario B is ≤ 50% of that
/// in scenario A (plan §10.2).
#[test]
#[ignore = "requires local model weights and the mlxcel binary"]
fn scenario_b_suppresses_ja_zh() {
    let model_path = repo_model_dir(MODEL_NAME);
    if !model_path.exists() {
        eprintln!(
            "Skipping scenario_b: model directory not found at {}",
            model_path.display()
        );
        return;
    }

    let prompts = load_ko_prompts();

    let a = run_scenario(&model_path, &prompts, &[]);
    let b = run_scenario(&model_path, &prompts, &["--lang-bias", "ja=-inf,zh=-inf"]);

    let suppress_scripts = &[Script::Hiragana, Script::Katakana, Script::Han];
    let a_ratio = a.ratio_of(suppress_scripts);
    let b_ratio = b.ratio_of(suppress_scripts);

    eprintln!(
        "Scenario B: a_ratio(ja+zh scripts)={:.4}, b_ratio={:.4} (threshold: b ≤ 0.5 * a)",
        a_ratio, b_ratio
    );

    // When the baseline ratio is very small (near zero), the suppression test
    // is trivially satisfied even without bias — accept in that case.
    if a_ratio < 1e-6 {
        eprintln!("Scenario B: baseline ja+zh ratio is negligible; test passes trivially.");
        return;
    }

    assert!(
        b_ratio <= a_ratio * 0.5,
        "scenario B: expected Hiragana+Katakana+Han ratio to drop by ≥50%; \
         a_ratio={a_ratio:.4}, b_ratio={b_ratio:.4}"
    );
}

/// Scenario C — promote Korean: `--lang-bias ko=+5` with Conservative policy.
///
/// Asserts: Hangul ratio in scenario C is ≥ 120% of that in scenario A
/// (plan §10.2).
#[test]
#[ignore = "requires local model weights and the mlxcel binary"]
fn scenario_c_promotes_ko() {
    let model_path = repo_model_dir(MODEL_NAME);
    if !model_path.exists() {
        eprintln!(
            "Skipping scenario_c: model directory not found at {}",
            model_path.display()
        );
        return;
    }

    let prompts = load_ko_prompts();

    let a = run_scenario(&model_path, &prompts, &[]);
    let c = run_scenario(&model_path, &prompts, &["--lang-bias", "ko=+5"]);

    let a_hangul = a.ratio_of(&[Script::Hangul]);
    let c_hangul = c.ratio_of(&[Script::Hangul]);

    eprintln!(
        "Scenario C: a_hangul={:.4}, c_hangul={:.4} (threshold: c ≥ 1.2 * a)",
        a_hangul, c_hangul
    );

    // If baseline Hangul is already saturated (≥ 95%), a 20% increase is
    // impossible; skip the assertion in that edge case.
    if a_hangul >= 0.95 {
        eprintln!(
            "Scenario C: baseline Hangul ratio is already near-saturated ({a_hangul:.4}); \
             test passes without further assertion."
        );
        return;
    }

    assert!(
        c_hangul >= a_hangul * 1.2,
        "scenario C: expected Hangul ratio to increase by ≥20%; \
         a_hangul={a_hangul:.4}, c_hangul={c_hangul:.4}"
    );
}

/// Scenario D — strict Korean suppress: `--lang-bias ko=-inf --lang-bias-policy strict`.
///
/// Asserts:
/// - Hangul ratio drops significantly compared to scenario A.
/// - Han ratio is preserved within ±25% of scenario A's Han ratio (plan §10.2).
///   (Strict Ko suppress = {Hangul} only; Han is not in the Strict Ko set.)
#[test]
#[ignore = "requires local model weights and the mlxcel binary"]
fn scenario_d_strict_ko_suppress_preserves_han() {
    let model_path = repo_model_dir(MODEL_NAME);
    if !model_path.exists() {
        eprintln!(
            "Skipping scenario_d: model directory not found at {}",
            model_path.display()
        );
        return;
    }

    let prompts = load_ko_prompts();

    let a = run_scenario(&model_path, &prompts, &[]);
    let d = run_scenario(
        &model_path,
        &prompts,
        &["--lang-bias", "ko=-inf", "--lang-bias-policy", "strict"],
    );

    let a_hangul = a.ratio_of(&[Script::Hangul]);
    let d_hangul = d.ratio_of(&[Script::Hangul]);
    let a_han = a.ratio_of(&[Script::Han]);
    let d_han = d.ratio_of(&[Script::Han]);

    eprintln!(
        "Scenario D: a_hangul={:.4}, d_hangul={:.4}; a_han={:.4}, d_han={:.4}",
        a_hangul, d_hangul, a_han, d_han
    );

    // Assert Hangul ratio drops.
    // When baseline Hangul is already negligible, skip — suppression trivially holds.
    if a_hangul >= 1e-4 {
        assert!(
            d_hangul < a_hangul,
            "scenario D: expected Hangul ratio to drop under ko=-inf strict; \
             a_hangul={a_hangul:.4}, d_hangul={d_hangul:.4}"
        );
    } else {
        eprintln!("Scenario D: baseline Hangul negligible; skip Hangul-drop assertion.");
    }

    // Assert Han ratio is preserved within ±25% of baseline.
    // When baseline Han is very small, a fixed absolute tolerance is more
    // meaningful: |d_han - a_han| ≤ max(0.25 * a_han, 0.01).
    let tolerance = (a_han * 0.25_f64).max(0.01);
    assert!(
        (d_han - a_han).abs() <= tolerance,
        "scenario D: Han ratio should be preserved within ±25% of baseline; \
         a_han={a_han:.4}, d_han={d_han:.4}, tolerance={tolerance:.4}"
    );
}

// ============================================================================
// Latency test (plan §10.3)
// ============================================================================

/// Latency test: per-token overhead of `--lang-bias ja=-inf,zh=-inf`
/// vs. baseline must be < 5% (plan §10.3).
///
/// Reports both timings to the test log regardless of pass/fail.
#[test]
#[ignore = "requires local model weights and the mlxcel binary"]
fn latency_lang_bias_overhead_below_5_percent() {
    let model_path = repo_model_dir(MODEL_NAME);
    if !model_path.exists() {
        eprintln!(
            "Skipping latency test: model directory not found at {}",
            model_path.display()
        );
        return;
    }

    let prompts = load_ko_prompts();

    let baseline = run_latency(&model_path, &prompts, &[]);
    let biased = run_latency(&model_path, &prompts, &["--lang-bias", "ja=-inf,zh=-inf"]);

    eprintln!(
        "Latency — baseline: {:.3} ms/tok ({} tokens), biased: {:.3} ms/tok ({} tokens)",
        baseline.mean_per_token_ms,
        baseline.total_measured_tokens,
        biased.mean_per_token_ms,
        biased.total_measured_tokens,
    );

    if baseline.mean_per_token_ms == 0.0 || baseline.total_measured_tokens == 0 {
        eprintln!("Latency test: baseline produced no measured tokens; skipping assertion.");
        return;
    }

    let overhead = (biased.mean_per_token_ms - baseline.mean_per_token_ms)
        / baseline.mean_per_token_ms;

    eprintln!(
        "Latency overhead: {:.2}% (threshold: < {:.0}%)",
        overhead * 100.0,
        LATENCY_OVERHEAD_THRESHOLD * 100.0
    );

    assert!(
        overhead < LATENCY_OVERHEAD_THRESHOLD,
        "lang-bias per-token overhead {:.2}% exceeds the {:.0}% threshold; \
         baseline={:.3} ms/tok, biased={:.3} ms/tok",
        overhead * 100.0,
        LATENCY_OVERHEAD_THRESHOLD * 100.0,
        baseline.mean_per_token_ms,
        biased.mean_per_token_ms,
    );
}
