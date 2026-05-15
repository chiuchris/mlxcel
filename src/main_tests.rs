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

use clap::Parser;

use super::{Cli, Commands};

#[test]
fn generate_command_parses_tensor_parallel_flags() {
    let cli = Cli::try_parse_from([
        "mlxcel",
        "generate",
        "-m",
        "models/foo",
        "-p",
        "hello",
        "--tp-size",
        "2",
        "--tp-moe-mode",
        "within_expert",
        "--tp-embedding-mode",
        "vocab_parallel",
        "--tp-lm-head-mode",
        "replicated",
    ])
    .unwrap();

    let Commands::Generate(args) = cli.command else {
        panic!("expected generate command");
    };

    assert_eq!(args.tensor_parallel.tp_size, 2);
    assert_eq!(args.tensor_parallel.tp_moe_mode, "within_expert");
    assert_eq!(args.tensor_parallel.tp_embedding_mode, "vocab_parallel");
    assert_eq!(args.tensor_parallel.tp_lm_head_mode, "replicated");
}

// Issue #371 (A4): CLI argument-parsing tests for the `--surgery <FILE>`
// flag on the `generate` and `serve` subcommands. These tests only cover
// the clap surface — they do not invoke the surgery pipeline or touch
// any model weights. The end-to-end behavior is exercised by the
// integration test in `tests/surgery_cli.rs`.

#[cfg(feature = "surgery")]
#[test]
fn generate_command_accepts_surgery_flag_with_path() {
    let cli = Cli::try_parse_from([
        "mlxcel",
        "generate",
        "-m",
        "models/foo",
        "-p",
        "hello",
        "--surgery",
        "config/surgery.yaml",
    ])
    .expect("clap must accept --surgery <path>");

    let Commands::Generate(args) = cli.command else {
        panic!("expected generate command");
    };

    assert_eq!(
        args.surgery,
        Some(std::path::PathBuf::from("config/surgery.yaml")),
        "--surgery must round-trip through clap as PathBuf"
    );
}

#[cfg(feature = "surgery")]
#[test]
fn generate_command_surgery_flag_defaults_to_none() {
    // Baseline path: omitting the flag yields `None`, which keeps the
    // load path bit-exact with pre-#371 main (acceptance criterion (e)).
    let cli = Cli::try_parse_from(["mlxcel", "generate", "-m", "models/foo", "-p", "hello"])
        .expect("clap must accept generate without --surgery");

    let Commands::Generate(args) = cli.command else {
        panic!("expected generate command");
    };

    assert!(
        args.surgery.is_none(),
        "absent --surgery flag must resolve to None"
    );
}

#[cfg(feature = "surgery")]
#[test]
fn serve_command_accepts_surgery_flag_with_path() {
    let cli = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "-m",
        "models/foo",
        "--surgery",
        "config/surgery.yaml",
    ])
    .expect("clap must accept --surgery on serve");

    let Commands::Serve(args) = cli.command else {
        panic!("expected serve command");
    };

    assert_eq!(
        args.surgery,
        Some(std::path::PathBuf::from("config/surgery.yaml")),
        "serve --surgery must round-trip through clap as PathBuf"
    );
}

#[cfg(feature = "surgery")]
#[test]
fn serve_command_surgery_flag_defaults_to_none() {
    let cli = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo"])
        .expect("clap must accept serve without --surgery");

    let Commands::Serve(args) = cli.command else {
        panic!("expected serve command");
    };

    assert!(
        args.surgery.is_none(),
        "absent --surgery flag on serve must resolve to None"
    );
}
