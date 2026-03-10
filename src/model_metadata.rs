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

//! Centralized model metadata and loading-policy descriptors.
//!
//! This module is the single control-plane reference for model kind and route
//! selection. Loader code should consume these descriptors instead of
//! re-deriving multimodal status, adapter support, or directory/weight routes
//! ad hoc.

use anyhow::Result;

use crate::loading::config_backed::is_config_backed_model_type;
use crate::loading::nonstandard::is_nonstandard_model_type;
use crate::loading::special::{adapter_loading_unsupported_message, is_special_weight_model_type};
use crate::models::ModelType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelKind {
    Text,
    Vlm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelCapabilities {
    pub(crate) kind: ModelKind,
    pub(crate) adapter_unsupported_message: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectoryLoadRoute {
    /// `Mistral3` wrapper with `ministral3` text tower in `text_config`.
    Mistral3TextWrapper,
    /// `Mistral3` wrapper whose inner text tower should fall back to `Llama`.
    Mistral3LlamaFallback,
    /// Vision-language model families routed through `src/loading/vlm*.rs`.
    Vlm,
    /// Text families with directory loaders that do not fit the standard registry.
    Nonstandard,
    /// Standard config-backed text families.
    ConfigBacked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WeightLoadRoute {
    /// `Llama` plus the `Mistral3` text-wrapper fallback path.
    LlamaFamily,
    /// Architectures that require owned weights or custom sanitization.
    Special,
    /// Standard config-backed text families.
    ConfigBacked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelLoadPolicy {
    pub(crate) capabilities: ModelCapabilities,
    pub(crate) directory_route: DirectoryLoadRoute,
    pub(crate) weight_route: Option<WeightLoadRoute>,
}

pub(crate) fn is_ministral3_config(config: &serde_json::Value) -> bool {
    config
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .and_then(|mt| mt.as_str())
        .map(|mt| mt == "ministral3")
        .unwrap_or(false)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_vlm_model_type(model_type: ModelType) -> bool {
    matches!(
        model_type,
        ModelType::Llama4VLM
            | ModelType::Qwen35VLM
            | ModelType::Qwen35MoeVLM
            | ModelType::Gemma3VLM
            | ModelType::LlavaVLM
            | ModelType::LlavaBunnyVLM
            | ModelType::AyaVisionVLM
            | ModelType::PaliGemmaVLM
            | ModelType::PixtralVLM
            | ModelType::Mistral3VLM
            | ModelType::Qwen2VL
            | ModelType::Qwen25VL
            | ModelType::Qwen3VL
            | ModelType::Qwen3VLMoe
            | ModelType::Gemma3nVLM
            | ModelType::Phi3VLM
            | ModelType::Molmo2VLM
    )
}

pub(crate) fn model_capabilities(model_type: ModelType) -> ModelCapabilities {
    ModelCapabilities {
        kind: if is_vlm_model_type(model_type) {
            ModelKind::Vlm
        } else {
            ModelKind::Text
        },
        adapter_unsupported_message: adapter_loading_unsupported_message(model_type),
    }
}

pub(crate) fn directory_load_route(
    model_type: ModelType,
    config: Option<&serde_json::Value>,
) -> Result<DirectoryLoadRoute> {
    if model_type == ModelType::Mistral3 {
        return Ok(if config.is_some_and(is_ministral3_config) {
            DirectoryLoadRoute::Mistral3TextWrapper
        } else {
            DirectoryLoadRoute::Mistral3LlamaFallback
        });
    }

    if is_vlm_model_type(model_type) {
        return Ok(DirectoryLoadRoute::Vlm);
    }

    if is_nonstandard_model_type(model_type) {
        return Ok(DirectoryLoadRoute::Nonstandard);
    }

    if is_config_backed_model_type(model_type) {
        return Ok(DirectoryLoadRoute::ConfigBacked);
    }

    Err(anyhow::anyhow!(
        "Missing directory loader for model type: {:?}",
        model_type
    ))
}

pub(crate) fn weight_load_route(model_type: ModelType) -> Result<WeightLoadRoute> {
    if matches!(model_type, ModelType::Llama | ModelType::Mistral3) {
        return Ok(WeightLoadRoute::LlamaFamily);
    }

    if is_special_weight_model_type(model_type) {
        return Ok(WeightLoadRoute::Special);
    }

    if is_config_backed_model_type(model_type) {
        return Ok(WeightLoadRoute::ConfigBacked);
    }

    Err(anyhow::anyhow!(
        "Missing weight loader for model type: {:?}",
        model_type
    ))
}

pub(crate) fn model_load_policy(
    model_type: ModelType,
    config: Option<&serde_json::Value>,
) -> Result<ModelLoadPolicy> {
    let capabilities = model_capabilities(model_type);
    let directory_route = directory_load_route(model_type, config)?;
    let weight_route = if capabilities.adapter_unsupported_message.is_some() {
        None
    } else {
        Some(weight_load_route(model_type)?)
    };

    Ok(ModelLoadPolicy {
        capabilities,
        directory_route,
        weight_route,
    })
}
