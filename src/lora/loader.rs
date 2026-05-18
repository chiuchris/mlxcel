//! LoRA adapter loading and weight fusion
//!
//! This module handles loading LoRA adapter weights and fusing them
//! with base model weights for efficient inference.
//!
//! Uses mlxcel-core types (WeightMap, UniquePtr<MlxArray>) for compatibility
//! with the rest of the codebase.

use anyhow::Result;
use mlxcel_core::MlxArray;
use mlxcel_core::UniquePtr;
use mlxcel_core::weights::WeightMap;
use std::path::Path;

use super::config::AdapterConfig;

/// Load adapter weights from a safetensors file
fn load_adapter_weights(adapter_path: &Path) -> Result<WeightMap> {
    let weights_path = adapter_path.join("adapters.safetensors");

    // Try adapters.safetensors first, then adapter_model.safetensors (HuggingFace format)
    let weights_path = if weights_path.exists() {
        weights_path
    } else {
        let alt_path = adapter_path.join("adapter_model.safetensors");
        if alt_path.exists() {
            alt_path
        } else {
            anyhow::bail!(
                "No adapter weights found. Expected adapters.safetensors or adapter_model.safetensors in {:?}",
                adapter_path
            );
        }
    };

    let weights = mlxcel_core::weights::load_safetensors(&weights_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to load adapter weights from {:?}: {}",
            weights_path,
            e
        )
    })?;

    Ok(weights)
}

/// Fuse LoRA weights into base model weights
///
/// LoRA formula: W_fused = W_base + scale * (lora_b @ lora_a)
///
/// Where:
/// - W_base: original weight matrix (out_features, in_features)
/// - lora_a: low-rank matrix A (rank, in_features)
/// - lora_b: low-rank matrix B (out_features, rank)
/// - scale: scaling factor (often alpha / rank)
///
/// Returns a new HashMap with fused weights
pub fn fuse_lora_weights(
    base_weights: &WeightMap,
    adapter_weights: &WeightMap,
    scale: f32,
) -> Result<WeightMap> {
    let mut fused_weights: WeightMap = base_weights
        .iter()
        .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
        .collect();

    // Group adapter weights by their base layer name
    // LoRA weights are typically named like:
    // - layers.0.self_attn.q_proj.lora_a (rank, in_features)
    // - layers.0.self_attn.q_proj.lora_b (out_features, rank)
    let mut lora_pairs: std::collections::HashMap<
        String,
        (Option<UniquePtr<MlxArray>>, Option<UniquePtr<MlxArray>>),
    > = std::collections::HashMap::new();

    for (name, weight) in adapter_weights {
        if name.ends_with(".lora_a") {
            let base_name = name.trim_end_matches(".lora_a").to_string();
            lora_pairs.entry(base_name).or_insert((None, None)).0 = Some(mlxcel_core::copy(weight));
        } else if name.ends_with(".lora_b") {
            let base_name = name.trim_end_matches(".lora_b").to_string();
            lora_pairs.entry(base_name).or_insert((None, None)).1 = Some(mlxcel_core::copy(weight));
        }
        // Ignore other weights (like scales for DoRA)
    }

    // Fuse each LoRA pair with the corresponding base weight
    for (base_name, (lora_a_opt, lora_b_opt)) in lora_pairs {
        let (Some(lora_a), Some(lora_b)) = (lora_a_opt, lora_b_opt) else {
            tracing::warn!(
                "Incomplete LoRA pair for {}: missing lora_a or lora_b",
                base_name
            );
            continue;
        };

        // Find the corresponding base weight
        let base_weight_name = find_base_weight_name(&base_name, base_weights)?;

        let Some(base_weight) = fused_weights.get(&base_weight_name) else {
            tracing::warn!(
                "Base weight not found for LoRA layer {}: tried {}",
                base_name,
                base_weight_name
            );
            continue;
        };

        // Compute the LoRA delta: scale * (lora_b @ lora_a)
        let delta = compute_lora_delta(&lora_a, &lora_b, scale)?;

        // Fuse: W_fused = W_base + delta
        let fused = mlxcel_core::add(base_weight, &delta);

        fused_weights.insert(base_weight_name, fused);
    }

    Ok(fused_weights)
}

/// Find the base weight name that corresponds to a LoRA layer name
fn find_base_weight_name(lora_name: &str, base_weights: &WeightMap) -> Result<String> {
    // Common patterns to try:
    // 1. Direct match with .weight suffix
    // 2. Replace specific LoRA naming conventions
    let candidates = vec![
        format!("{}.weight", lora_name),
        lora_name.to_string(),
        // HuggingFace PEFT format uses base_layer
        lora_name.replace(".base_layer", ".weight"),
    ];

    for candidate in &candidates {
        if base_weights.contains_key(candidate) {
            return Ok(candidate.clone());
        }
    }

    // If no direct match, return the most likely candidate
    Ok(format!("{}.weight", lora_name))
}

/// Compute the LoRA delta: scale * (lora_b @ lora_a)
///
/// Handles different matrix orientations based on shapes
fn compute_lora_delta(
    lora_a: &MlxArray,
    lora_b: &MlxArray,
    scale: f32,
) -> Result<UniquePtr<MlxArray>> {
    let a_shape = mlxcel_core::array_shape(lora_a);
    let b_shape = mlxcel_core::array_shape(lora_b);

    // Determine orientation based on shapes
    // We need: delta shape = (out_features, in_features) for Linear weight
    //
    // mlx-lm convention:
    // - lora_a: (in_features, rank)
    // - lora_b: (rank, out_features)
    // - delta = (lora_a @ lora_b).T = lora_b.T @ lora_a.T
    //
    // Standard convention (HuggingFace PEFT):
    // - lora_a: (rank, in_features)
    // - lora_b: (out_features, rank)
    // - delta = lora_b @ lora_a

    let delta = if a_shape.len() == 2 && b_shape.len() == 2 {
        // Check if shapes are compatible for either convention
        if a_shape[1] == b_shape[0] {
            // mlx-lm: a=(in, rank), b=(rank, out) -> need transpose result
            let product = mlxcel_core::matmul(lora_a, lora_b);
            mlxcel_core::transpose_axes(&product, &[1, 0])
        } else if a_shape[0] == b_shape[1] {
            // Standard PEFT: a=(rank, in), b=(out, rank) -> b @ a
            mlxcel_core::matmul(lora_b, lora_a)
        } else {
            anyhow::bail!(
                "Incompatible LoRA shapes: lora_a={:?}, lora_b={:?}",
                a_shape,
                b_shape
            );
        }
    } else {
        anyhow::bail!(
            "Expected 2D LoRA matrices, got lora_a={:?}, lora_b={:?}",
            a_shape,
            b_shape
        );
    };

    // Scale the delta
    let scale_arr = mlxcel_core::full_f32(&[1], scale, mlxcel_core::dtype::FLOAT32);
    let scaled_delta = mlxcel_core::multiply(&delta, &scale_arr);

    Ok(scaled_delta)
}

/// Apply LoRA adapters to base model weights by fusion
///
/// This function loads the adapter configuration and weights,
/// then fuses the LoRA weights with the base model weights.
///
/// # Arguments
///
/// * `base_weights` - The base model weights to modify
/// * `adapter_path` - Path to the adapter directory containing adapter_config.json and adapters.safetensors
///
/// # Returns
///
/// A new HashMap containing the fused weights
pub fn apply_lora_adapters(base_weights: &WeightMap, adapter_path: &Path) -> Result<WeightMap> {
    // Load adapter configuration
    let config = AdapterConfig::load(adapter_path)?;

    tracing::info!(
        "Loading LoRA adapter: rank={}, scale={:.2}, type={:?}",
        config.rank(),
        config.effective_scale(),
        config.fine_tune_type
    );

    if !config.is_lora() {
        anyhow::bail!(
            "Adapter is not LoRA type: {:?}. Full fine-tuning adapters should be loaded directly.",
            config.fine_tune_type
        );
    }

    // Load adapter weights
    let adapter_weights = load_adapter_weights(adapter_path)?;

    tracing::info!("Loaded {} adapter weight tensors", adapter_weights.len());

    // Fuse weights
    let fused = fuse_lora_weights(base_weights, &adapter_weights, config.effective_scale())?;

    // Count how many weights were modified
    let modified_count = adapter_weights
        .keys()
        .filter(|k| k.ends_with(".lora_a"))
        .count();

    tracing::info!("Fused LoRA adapters into {} layers", modified_count);

    Ok(fused)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_lora_delta_mlx_format() {
        // mlx-lm format: a=(in=4, rank=2), b=(rank=2, out=3)
        // Result should be (out=3, in=4)
        let lora_a = mlxcel_core::from_slice_f32(&[1.0f32; 8], &[4, 2]);
        let lora_b = mlxcel_core::from_slice_f32(&[1.0f32; 6], &[2, 3]);
        let scale = 1.0;

        let delta = compute_lora_delta(&lora_a, &lora_b, scale).unwrap();
        assert_eq!(mlxcel_core::array_shape(&delta), vec![3, 4]);
    }

    #[test]
    fn test_compute_lora_delta_peft_format() {
        // PEFT format: a=(rank=2, in=4), b=(out=3, rank=2)
        // Result should be (out=3, in=4)
        let lora_a = mlxcel_core::from_slice_f32(&[1.0f32; 8], &[2, 4]);
        let lora_b = mlxcel_core::from_slice_f32(&[1.0f32; 6], &[3, 2]);
        let scale = 1.0;

        let delta = compute_lora_delta(&lora_a, &lora_b, scale).unwrap();
        assert_eq!(mlxcel_core::array_shape(&delta), vec![3, 4]);
    }

    #[test]
    fn test_fuse_lora_weights_basic() {
        // Create base weights
        let mut base_weights = WeightMap::new();
        base_weights.insert(
            "layer.weight".to_string(),
            mlxcel_core::ones(&[3, 4], mlxcel_core::dtype::FLOAT32),
        );

        // Create adapter weights (mlx-lm format)
        let mut adapter_weights = WeightMap::new();
        adapter_weights.insert(
            "layer.lora_a".to_string(),
            mlxcel_core::ones(&[4, 2], mlxcel_core::dtype::FLOAT32),
        );
        adapter_weights.insert(
            "layer.lora_b".to_string(),
            mlxcel_core::ones(&[2, 3], mlxcel_core::dtype::FLOAT32),
        );

        let fused = fuse_lora_weights(&base_weights, &adapter_weights, 1.0).unwrap();

        // Should have the same key
        assert!(fused.contains_key("layer.weight"));
        let fused_weight = fused.get("layer.weight").unwrap();
        let shape = mlxcel_core::array_shape(fused_weight);
        assert_eq!(shape, vec![3, 4]);

        // Original was all 1s, delta should be scale * (lora_b.T @ lora_a.T) = 2s matrix
        // So fused should be > 1.0
        mlxcel_core::eval(fused_weight);
        let sum = mlxcel_core::sum_all(fused_weight);
        mlxcel_core::eval(&sum);
        let sum_val = mlxcel_core::item_f32(&sum);
        assert!(sum_val > 12.0); // 3*4 = 12 base + delta > 0
    }
}
