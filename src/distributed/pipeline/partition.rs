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

//! Core layer partitioning algorithm for pipeline parallelism.
//!
//! Given a [`ModelProfile`] (layer count, per-layer parameter cost, embedding
//! and lm_head sizes) and a set of [`DeviceSpec`]s (available memory, compute
//! units), this module produces a vector of [`StageAssignment`]s that map
//! contiguous layer ranges to devices.
//!
//! The auto-partitioner distributes layers proportionally to each device's
//! available memory after reserving space for the embedding (first stage) and
//! lm_head (last stage). Manual partitions are parsed from the `--pp-layers`
//! CLI flag and validated for correctness.
//!
//! Used by: server startup, model loading pipeline

use std::ops::Range;

use anyhow::{Result, bail, ensure};

/// Describes the memory footprint of a model for partitioning purposes.
///
/// All sizes are in bytes. The partitioner uses these to estimate per-stage
/// memory requirements and balance the assignment across devices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelProfile {
    /// Total number of transformer/SSM layers in the model.
    pub num_layers: usize,
    /// Estimated parameter memory per layer (bytes). For simplicity, all
    /// layers are assumed equal; MoE models with variable expert counts
    /// should use the average or worst-case layer size.
    pub layer_param_bytes: u64,
    /// Memory consumed by the embedding table (bytes). Assigned to the
    /// first pipeline stage.
    pub embedding_param_bytes: u64,
    /// Memory consumed by the lm_head / output projection (bytes). Assigned
    /// to the last pipeline stage.
    pub lm_head_param_bytes: u64,
}

impl ModelProfile {
    /// Total model parameter memory (embedding + all layers + lm_head).
    pub fn total_param_bytes(&self) -> u64 {
        self.embedding_param_bytes
            .saturating_add((self.num_layers as u64).saturating_mul(self.layer_param_bytes))
            .saturating_add(self.lm_head_param_bytes)
    }
}

/// Describes the capabilities and constraints of a single device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceSpec {
    /// Unique device identifier (matches node ID in cluster config).
    pub device_id: String,
    /// Available memory in bytes for model parameters on this device.
    pub available_memory_bytes: u64,
    /// Number of compute units (GPU cores, neural engine cores, etc.).
    /// Used as a secondary signal for balancing; 0 means unknown.
    pub compute_units: u32,
}

/// A single pipeline stage assignment produced by the partitioner.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StageAssignment {
    /// Index of this stage (0-based, ascending).
    pub stage_index: usize,
    /// Device this stage is assigned to.
    pub device_id: String,
    /// Contiguous range of layer indices assigned to this stage.
    /// Uses half-open range: `start..end` where `end` is exclusive.
    pub layer_range: Range<usize>,
    /// Whether this stage hosts the embedding table.
    pub has_embedding: bool,
    /// Whether this stage hosts the lm_head / output projection.
    pub has_lm_head: bool,
    /// Estimated total memory consumption for this stage (bytes).
    pub estimated_memory_bytes: u64,
}

/// Partition configuration: auto or manual.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PartitionConfig {
    /// Automatically distribute layers proportionally to device memory.
    #[default]
    Auto,
    /// Manually specified layer ranges (one per device, in device order).
    Manual(Vec<Range<usize>>),
}

/// Parse a manual partition specification string into layer ranges.
///
/// Format: comma-separated `start-end` pairs (inclusive on both ends),
/// e.g., `"0-15,16-31"` for a 32-layer model split across 2 devices.
///
/// Returns one `Range<usize>` per stage, using half-open ranges internally.
pub fn parse_manual_partition(spec: &str, num_layers: usize) -> Result<Vec<Range<usize>>> {
    let spec = spec.trim();
    ensure!(!spec.is_empty(), "partition spec must not be empty");

    let mut ranges = Vec::new();

    for (i, part) in spec.split(',').enumerate() {
        let part = part.trim();
        let dash_pos = part.find('-').ok_or_else(|| {
            anyhow::anyhow!("invalid range at position {i}: '{part}' (expected start-end)")
        })?;

        let start_str = &part[..dash_pos];
        let end_str = &part[dash_pos + 1..];

        let start: usize = start_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid start index in range '{part}'"))?;
        let end_inclusive: usize = end_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid end index in range '{part}'"))?;

        ensure!(
            start <= end_inclusive,
            "invalid range '{part}': start ({start}) > end ({end_inclusive})"
        );
        ensure!(
            end_inclusive < num_layers,
            "layer index {end_inclusive} exceeds model layer count ({num_layers})"
        );

        ranges.push(start..end_inclusive + 1);
    }

    Ok(ranges)
}

/// Validate a set of stage assignments for correctness.
///
/// Checks:
/// - All layers `0..num_layers` are covered exactly once (no gaps, no overlaps)
/// - Ranges are contiguous and sorted
/// - Embedding is on the first stage, lm_head on the last stage
/// - No stage exceeds its device memory (when `devices` is provided)
pub fn validate_partition(assignments: &[StageAssignment], num_layers: usize) -> Result<()> {
    ensure!(
        !assignments.is_empty(),
        "partition must contain at least one stage"
    );

    // Check stage indices are sequential 0..N.
    for (i, a) in assignments.iter().enumerate() {
        ensure!(
            a.stage_index == i,
            "stage index mismatch: expected {i}, got {}",
            a.stage_index
        );
    }

    // Check layer coverage: ranges must tile 0..num_layers exactly.
    let mut expected_start = 0;
    for a in assignments {
        ensure!(
            a.layer_range.start == expected_start,
            "gap or overlap at layer {}: stage {} starts at {} but expected {}",
            expected_start,
            a.stage_index,
            a.layer_range.start,
            expected_start
        );
        ensure!(
            a.layer_range.end > a.layer_range.start,
            "empty range on stage {}",
            a.stage_index
        );
        expected_start = a.layer_range.end;
    }
    ensure!(
        expected_start == num_layers,
        "partition covers layers 0..{expected_start} but model has {num_layers} layers"
    );

    // Check embedding placement.
    ensure!(
        assignments[0].has_embedding,
        "first stage must host the embedding table"
    );
    for a in &assignments[1..] {
        ensure!(
            !a.has_embedding,
            "only the first stage should host the embedding table (stage {} has it)",
            a.stage_index
        );
    }

    // Check lm_head placement.
    let last = assignments.last().unwrap();
    ensure!(last.has_lm_head, "last stage must host the lm_head");
    for a in &assignments[..assignments.len() - 1] {
        ensure!(
            !a.has_lm_head,
            "only the last stage should host lm_head (stage {} has it)",
            a.stage_index
        );
    }

    Ok(())
}

/// Validate that each stage fits within its device's memory.
///
/// Separated from [`validate_partition`] so callers can validate structural
/// correctness without requiring device specs.
pub fn validate_memory_fit(assignments: &[StageAssignment], devices: &[DeviceSpec]) -> Result<()> {
    ensure!(
        assignments.len() == devices.len(),
        "partition has {} stages but {} devices were provided",
        assignments.len(),
        devices.len()
    );

    for (a, d) in assignments.iter().zip(devices.iter()) {
        ensure!(
            a.estimated_memory_bytes <= d.available_memory_bytes,
            "stage {} on device '{}' requires {} bytes but only {} bytes available",
            a.stage_index,
            d.device_id,
            a.estimated_memory_bytes,
            d.available_memory_bytes
        );
    }

    Ok(())
}

/// Automatically partition model layers across devices proportionally to
/// each device's available memory.
///
/// Algorithm:
/// 1. Reserve embedding memory on the first device and lm_head on the last.
/// 2. Compute effective memory per device (after reservations).
/// 3. Distribute layers proportionally to effective memory.
/// 4. Ensure every device gets at least one layer.
/// 5. Build `StageAssignment` with memory estimates.
///
/// Returns an error if the model cannot fit in the combined device memory
/// or if there are more devices than layers.
pub fn auto_partition(
    model: &ModelProfile,
    devices: &[DeviceSpec],
) -> Result<Vec<StageAssignment>> {
    let n_devices = devices.len();
    let n_layers = model.num_layers;

    ensure!(n_devices > 0, "at least one device is required");
    ensure!(n_layers > 0, "model must have at least one layer");
    ensure!(
        n_devices <= n_layers,
        "more devices ({n_devices}) than layers ({n_layers}); \
         reduce the number of pipeline stages"
    );

    // Step 1: Compute effective memory per device after embedding/lm_head reservations.
    let mut effective_memory: Vec<u64> = devices.iter().map(|d| d.available_memory_bytes).collect();

    // Reserve embedding on first device.
    if effective_memory[0] < model.embedding_param_bytes {
        bail!(
            "device '{}' has {} bytes but embedding alone requires {} bytes",
            devices[0].device_id,
            devices[0].available_memory_bytes,
            model.embedding_param_bytes
        );
    }
    effective_memory[0] -= model.embedding_param_bytes;

    // Reserve lm_head on last device.
    let last_idx = n_devices - 1;
    if effective_memory[last_idx] < model.lm_head_param_bytes {
        bail!(
            "device '{}' has {} bytes but lm_head alone requires {} bytes",
            devices[last_idx].device_id,
            devices[last_idx].available_memory_bytes,
            model.lm_head_param_bytes
        );
    }
    effective_memory[last_idx] -= model.lm_head_param_bytes;

    // Step 2: Check that each device can hold at least one layer.
    for (i, &mem) in effective_memory.iter().enumerate() {
        if mem < model.layer_param_bytes {
            bail!(
                "device '{}' has insufficient memory for even one layer \
                 ({} bytes available after reservations, {} bytes per layer)",
                devices[i].device_id,
                mem,
                model.layer_param_bytes
            );
        }
    }

    // Step 3: Distribute layers proportionally to effective memory.
    let total_effective: u64 = effective_memory.iter().sum();
    let total_layer_bytes = n_layers as u64 * model.layer_param_bytes;

    if total_effective < total_layer_bytes {
        bail!(
            "combined device memory ({total_effective} bytes after reservations) \
             is insufficient for {n_layers} layers ({total_layer_bytes} bytes)"
        );
    }

    let mut layer_counts: Vec<usize> = effective_memory
        .iter()
        .map(|&mem| {
            // Proportional share, floored.
            let share = (mem as f64 / total_effective as f64) * n_layers as f64;
            share.floor() as usize
        })
        .collect();

    // Ensure every device gets at least 1 layer.
    for count in &mut layer_counts {
        if *count == 0 {
            *count = 1;
        }
    }

    // Distribute remaining layers to devices with the most headroom.
    let assigned: usize = layer_counts.iter().sum();
    if assigned < n_layers {
        let mut remaining = n_layers - assigned;
        // Sort device indices by remaining capacity (descending).
        let mut indices: Vec<usize> = (0..n_devices).collect();
        indices.sort_by(|&a, &b| {
            let used_a = (layer_counts[a] as u64).saturating_mul(model.layer_param_bytes);
            let used_b = (layer_counts[b] as u64).saturating_mul(model.layer_param_bytes);
            let headroom_a = effective_memory[a].saturating_sub(used_a);
            let headroom_b = effective_memory[b].saturating_sub(used_b);
            headroom_b.cmp(&headroom_a)
        });
        for &idx in &indices {
            if remaining == 0 {
                break;
            }
            layer_counts[idx] += 1;
            remaining -= 1;
        }
    } else if assigned > n_layers {
        // Over-assigned due to minimum-1 enforcement; remove excess from
        // the device with the fewest effective bytes (least capacity).
        let mut excess = assigned - n_layers;
        let mut indices: Vec<usize> = (0..n_devices).collect();
        indices.sort_by_key(|&i| effective_memory[i]);
        for &idx in &indices {
            if excess == 0 {
                break;
            }
            if layer_counts[idx] > 1 {
                let can_remove = (layer_counts[idx] - 1).min(excess);
                layer_counts[idx] -= can_remove;
                excess -= can_remove;
            }
        }
        if excess > 0 {
            bail!(
                "cannot distribute {n_layers} layers across {n_devices} devices \
                 while giving each device at least one layer"
            );
        }
    }

    // Step 4: Build stage assignments.
    let mut assignments = Vec::with_capacity(n_devices);
    let mut layer_start = 0;

    for (i, &count) in layer_counts.iter().enumerate() {
        let layer_end = layer_start + count;
        let is_first = i == 0;
        let is_last = i == last_idx;

        let mut mem = count as u64 * model.layer_param_bytes;
        if is_first {
            mem += model.embedding_param_bytes;
        }
        if is_last {
            mem += model.lm_head_param_bytes;
        }

        assignments.push(StageAssignment {
            stage_index: i,
            device_id: devices[i].device_id.clone(),
            layer_range: layer_start..layer_end,
            has_embedding: is_first,
            has_lm_head: is_last,
            estimated_memory_bytes: mem,
        });

        layer_start = layer_end;
    }

    // Step 5: Validate the result.
    validate_partition(&assignments, n_layers)?;
    validate_memory_fit(&assignments, devices)?;

    Ok(assignments)
}

/// Build stage assignments from manually specified layer ranges.
///
/// Pairs each range with the corresponding device (by index), sets embedding
/// on the first stage and lm_head on the last, computes memory estimates,
/// and validates the result.
pub fn build_manual_assignments(
    ranges: &[Range<usize>],
    model: &ModelProfile,
    devices: &[DeviceSpec],
) -> Result<Vec<StageAssignment>> {
    ensure!(
        ranges.len() == devices.len(),
        "manual partition has {} ranges but {} devices were provided",
        ranges.len(),
        devices.len()
    );

    let n_stages = ranges.len();
    let mut assignments = Vec::with_capacity(n_stages);

    for (i, range) in ranges.iter().enumerate() {
        let is_first = i == 0;
        let is_last = i == n_stages - 1;
        let layer_count = range.end - range.start;

        let mut mem = layer_count as u64 * model.layer_param_bytes;
        if is_first {
            mem += model.embedding_param_bytes;
        }
        if is_last {
            mem += model.lm_head_param_bytes;
        }

        assignments.push(StageAssignment {
            stage_index: i,
            device_id: devices[i].device_id.clone(),
            layer_range: range.clone(),
            has_embedding: is_first,
            has_lm_head: is_last,
            estimated_memory_bytes: mem,
        });
    }

    validate_partition(&assignments, model.num_layers)?;
    validate_memory_fit(&assignments, devices)?;

    Ok(assignments)
}

#[cfg(test)]
#[path = "partition_tests.rs"]
mod tests;
