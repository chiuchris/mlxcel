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
