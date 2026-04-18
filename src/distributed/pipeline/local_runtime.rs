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

//! Shared helpers for single-machine in-process pipeline execution.
//!
//! Used by: CLI pipeline generate path, server pipeline runtime

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};

use crate::distributed::pipeline::{
    ChannelConfig, DeviceSpec, InProcessStageWorkerLoop, LoadedStageExecutor, ModelProfile,
    PipelineConfig, StageAssignment, StageExecutor, auto_partition, build_manual_assignments,
    parse_manual_partition,
};
use crate::models::sanitize_config_json;

fn equal_stage_model_profile(num_layers: usize) -> ModelProfile {
    ModelProfile {
        num_layers,
        layer_param_bytes: 1024,
        embedding_param_bytes: 1024,
        lm_head_param_bytes: 1024,
    }
}

fn equal_capacity_devices(num_stages: usize, model: &ModelProfile) -> Vec<DeviceSpec> {
    let per_stage_layers = model.num_layers.div_ceil(num_stages) as u64;
    let per_stage_budget = model
        .embedding_param_bytes
        .saturating_add(model.lm_head_param_bytes)
        .saturating_add(model.layer_param_bytes.saturating_mul(per_stage_layers + 2));
    (0..num_stages)
        .map(|stage_index| DeviceSpec {
            device_id: format!("local-stage-{stage_index}"),
            available_memory_bytes: per_stage_budget,
            compute_units: 1,
        })
        .collect()
}

pub fn resolve_in_process_pipeline_num_layers(model_dir: &Path) -> Result<usize> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config_str = sanitize_config_json(&config_str);
    let config: serde_json::Value = serde_json::from_str(&config_str)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    let num_layers = config
        .get("num_hidden_layers")
        .and_then(|value| value.as_u64())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|text| text.get("num_hidden_layers"))
                .and_then(|value| value.as_u64())
        })
        .ok_or_else(|| {
            anyhow!(
                "config {} is missing an integer num_hidden_layers field (top-level or text_config.num_hidden_layers)",
                config_path.display()
            )
        })?;

    // DeepSeek V3 stores a trailing multi-token-prediction (MTP) layer at
    // `model.layers.{num_hidden_layers - 1}` that is **not** part of the
    // normal transformer stack. The Rust model implementation skips it, so
    // the effective depth seen by the pipeline partitioner must also skip
    // it, otherwise the last stage's auto-assigned range extends past the
    // real decoder blocks and fails to load.
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let effective_layers = if model_type == "deepseek_v3" {
        (num_layers as usize).saturating_sub(1).max(1)
    } else {
        num_layers as usize
    };
    Ok(effective_layers)
}

pub fn resolve_in_process_stage_assignments(
    num_layers: usize,
    num_stages: Option<usize>,
    manual_spec: Option<&str>,
) -> Result<Vec<StageAssignment>> {
    let profile = equal_stage_model_profile(num_layers);
    if let Some(spec) = manual_spec {
        ensure!(
            !spec.trim().is_empty(),
            "--pp-layers must not be empty when provided"
        );
        let ranges = parse_manual_partition(spec, num_layers)?;
        if let Some(expected) = num_stages
            && expected > 1
        {
            ensure!(
                expected == ranges.len(),
                "--pp-size ({expected}) does not match manual partition stage count ({})",
                ranges.len()
            );
        }
        let devices = equal_capacity_devices(ranges.len(), &profile);
        return build_manual_assignments(&ranges, &profile, &devices);
    }

    let num_stages = num_stages.ok_or_else(|| {
        anyhow!("in-process pipeline execution requires either --pp-layers or --pp-size >= 2")
    })?;
    ensure!(
        num_stages >= 2,
        "in-process pipeline execution requires at least 2 stages"
    );
    let devices = equal_capacity_devices(num_stages, &profile);
    auto_partition(&profile, &devices)
}

pub fn load_in_process_stage_worker(
    model_dir: &Path,
    assignments: &[StageAssignment],
    micro_batch_size: usize,
) -> Result<InProcessStageWorkerLoop> {
    load_in_process_stage_worker_with_adapter(model_dir, assignments, micro_batch_size, None)
}

/// Worktree-aware variant that also plumbs a single-adapter LoRA path into
/// every stage's loader. Passing `None` reproduces the base-only path.
///
/// Used by: CLI pipeline generate, server pipeline runtime, tests
pub fn load_in_process_stage_worker_with_adapter(
    model_dir: &Path,
    assignments: &[StageAssignment],
    micro_batch_size: usize,
    adapter_path: Option<&Path>,
) -> Result<InProcessStageWorkerLoop> {
    ensure!(
        assignments.len() >= 2,
        "in-process pipeline execution requires at least 2 stages"
    );
    let executors: Vec<Box<dyn StageExecutor>> = assignments
        .iter()
        .map(|assignment| {
            LoadedStageExecutor::load_with_adapter(model_dir, assignment, adapter_path)
                .map(|executor| Box::new(executor) as Box<dyn StageExecutor>)
        })
        .collect::<Result<_>>()
        .with_context(|| {
            format!(
                "in-process pipeline execution is not available for model {}",
                model_dir.display()
            )
        })?;

    InProcessStageWorkerLoop::new(
        PipelineConfig::new(assignments.len() as u32, micro_batch_size)?,
        executors,
        ChannelConfig::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Minimal temp-dir helper: avoids pulling `tempfile` into the main
    /// crate's dependency tree for a two-test unit suite. The path is kept
    /// unique per-process and removed on drop.
    struct ScratchDir {
        path: std::path::PathBuf,
    }

    impl ScratchDir {
        fn new(tag: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let pid = std::process::id();
            let path = std::env::temp_dir().join(format!("mlxcel-pp-{tag}-{pid}-{nanos}"));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write_config(model_dir: &Path, content: &str) {
        let mut file = fs::File::create(model_dir.join("config.json")).unwrap();
        file.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn num_layers_returns_raw_count_for_llama_family() {
        let dir = ScratchDir::new("llama-count");
        write_config(
            &dir.path,
            r#"{"model_type":"llama","num_hidden_layers":16,"hidden_size":128}"#,
        );
        let layers = resolve_in_process_pipeline_num_layers(&dir.path).unwrap();
        assert_eq!(layers, 16);
    }

    #[test]
    fn num_layers_strips_mtp_layer_for_deepseek_v3() {
        // DeepSeek V3's MTP trailer is not part of the transformer stack,
        // so the pipeline partitioner must see the number of decoder blocks,
        // not the raw config count.
        let dir = ScratchDir::new("deepseek_v3-count");
        write_config(
            &dir.path,
            r#"{"model_type":"deepseek_v3","num_hidden_layers":61,"hidden_size":128}"#,
        );
        let layers = resolve_in_process_pipeline_num_layers(&dir.path).unwrap();
        assert_eq!(layers, 60);
    }
}
