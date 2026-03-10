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

//! Centralized model descriptors and loading-policy metadata.
//!
//! This module is the single control-plane registry for model kind, directory
//! load family, weight load family, and adapter support. Loader code should
//! consume these descriptors instead of maintaining parallel support matches in
//! multiple modules.

use anyhow::Result;

use crate::models::ModelType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelKind {
    Text,
    Vlm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectoryRouteFamily {
    /// `Mistral3` wrapper with nested `text_config` that picks the final route.
    Mistral3Dynamic,
    /// Vision-language families routed through `src/loading/vlm*.rs`.
    Vlm,
    /// Text families with directory loaders outside the standard registry.
    Nonstandard,
    /// Standard config-backed text families.
    ConfigBacked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StaticModelDescriptor {
    pub(crate) model_type: ModelType,
    pub(crate) kind: ModelKind,
    pub(crate) directory_family: DirectoryRouteFamily,
    pub(crate) adapter_weight_route: Option<WeightLoadRoute>,
    pub(crate) adapter_unsupported_message: Option<&'static str>,
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
    pub(crate) descriptor: StaticModelDescriptor,
    pub(crate) capabilities: ModelCapabilities,
    pub(crate) directory_route: DirectoryLoadRoute,
    pub(crate) weight_route: Option<WeightLoadRoute>,
}

macro_rules! for_each_model_descriptor {
    ($macro:ident) => {
        $macro! {
            Llama => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Llama4 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Llama4VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Llama4 VLM cannot be loaded with LoRA adapters yet") };
            Qwen2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Qwen3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Qwen3Moe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Qwen3Next => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Qwen35 => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Qwen35VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Qwen3.5 VLM does not support adapter loading") };
            Qwen35Moe => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Qwen35MoeVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Qwen3.5 VLM does not support adapter loading") };
            Gemma => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Gemma2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Gemma3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Gemma3VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Gemma3 VLM cannot be loaded with LoRA adapters yet") };
            LlavaVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("LLaVA VLM cannot be loaded with LoRA adapters yet") };
            LlavaBunnyVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("LLaVA VLM cannot be loaded with LoRA adapters yet") };
            AyaVisionVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Aya Vision VLM cannot be loaded with LoRA adapters yet") };
            PaliGemmaVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("PaliGemma VLM cannot be loaded with LoRA adapters yet") };
            PixtralVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Pixtral/Mistral3 VLM cannot be loaded with LoRA adapters yet") };
            Mistral3VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Pixtral/Mistral3 VLM cannot be loaded with LoRA adapters yet") };
            Qwen2VL => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Qwen2-VL cannot be loaded with LoRA adapters yet") };
            Qwen25VL => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Qwen VL models cannot be loaded with LoRA adapters yet") };
            Qwen3VL => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Qwen VL models cannot be loaded with LoRA adapters yet") };
            Qwen3VLMoe => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Qwen VL models cannot be loaded with LoRA adapters yet") };
            Gemma3n => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Gemma3nVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Gemma3n VLM cannot be loaded with LoRA adapters yet") };
            Phi => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Phi3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Phi3VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Phi3V VLM does not support adapter loading; use load_model() instead") };
            Molmo2VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Molmo2 VLM does not support adapter loading; use load_model() instead") };
            Phi3Small => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            PhiMoe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            MiniMax => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Mixtral => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Qwen2Moe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            OLMoE => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            DeepSeek => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            DeepSeekV2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            DeepSeekV3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            DeepSeekV32 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Cohere => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Cohere2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            InternLM2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            InternLM3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Baichuan => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Glm4 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Glm4Moe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Glm4MoeLite => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            GlmMoeDsa => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Ernie45 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Ernie45Moe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            HunyuanMoe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            HunyuanV1Dense => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            MiMo => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            ExaOne => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            ExaOne4 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            ExaOneMoe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            SolarOpen => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Olmo => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Olmo2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Olmo3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            StarCoder2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            MiniCPM => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            MiniCPM3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            StableLM => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            SmolLM3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Ministral3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Mistral3 => { kind: Text, directory: Mistral3Dynamic, weight: Some(WeightLoadRoute::LlamaFamily), adapter: None };
            Nemotron => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Mamba => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Mamba2 => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Jamba => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            NemotronH => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            NemotronNAS => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            KimiLinear => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            LongcatFlash => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            LongcatFlashNgram => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Step3p5 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None };
            Rwkv7 => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            RecurrentGemma => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
        }
    };
}

macro_rules! descriptor_match {
    ($( $variant:ident => { kind: $kind:ident, directory: $directory:ident, weight: $weight:expr, adapter: $adapter:expr }; )*) => {
        pub(crate) fn static_model_descriptor(model_type: ModelType) -> StaticModelDescriptor {
            match model_type {
                $(
                    ModelType::$variant => StaticModelDescriptor {
                        model_type,
                        kind: ModelKind::$kind,
                        directory_family: DirectoryRouteFamily::$directory,
                        adapter_weight_route: $weight,
                        adapter_unsupported_message: $adapter,
                    },
                )*
            }
        }
    };
}

for_each_model_descriptor!(descriptor_match);

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
    static_model_descriptor(model_type).kind == ModelKind::Vlm
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_config_backed_model_type(model_type: ModelType) -> bool {
    static_model_descriptor(model_type).directory_family == DirectoryRouteFamily::ConfigBacked
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_nonstandard_model_type(model_type: ModelType) -> bool {
    static_model_descriptor(model_type).directory_family == DirectoryRouteFamily::Nonstandard
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn adapter_loading_unsupported_message(model_type: ModelType) -> Option<&'static str> {
    static_model_descriptor(model_type).adapter_unsupported_message
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_special_weight_model_type(model_type: ModelType) -> bool {
    static_model_descriptor(model_type).adapter_weight_route == Some(WeightLoadRoute::Special)
}

pub(crate) fn model_capabilities(model_type: ModelType) -> ModelCapabilities {
    let descriptor = static_model_descriptor(model_type);
    ModelCapabilities {
        kind: descriptor.kind,
        adapter_unsupported_message: descriptor.adapter_unsupported_message,
    }
}

pub(crate) fn directory_load_route(
    model_type: ModelType,
    config: Option<&serde_json::Value>,
) -> Result<DirectoryLoadRoute> {
    let descriptor = static_model_descriptor(model_type);
    Ok(match descriptor.directory_family {
        DirectoryRouteFamily::Mistral3Dynamic => {
            if config.is_some_and(is_ministral3_config) {
                DirectoryLoadRoute::Mistral3TextWrapper
            } else {
                DirectoryLoadRoute::Mistral3LlamaFallback
            }
        }
        DirectoryRouteFamily::Vlm => DirectoryLoadRoute::Vlm,
        DirectoryRouteFamily::Nonstandard => DirectoryLoadRoute::Nonstandard,
        DirectoryRouteFamily::ConfigBacked => DirectoryLoadRoute::ConfigBacked,
    })
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn weight_load_route(model_type: ModelType) -> Result<WeightLoadRoute> {
    static_model_descriptor(model_type)
        .adapter_weight_route
        .ok_or_else(|| anyhow::anyhow!("Missing weight loader for model type: {:?}", model_type))
}

pub(crate) fn model_load_policy(
    model_type: ModelType,
    config: Option<&serde_json::Value>,
) -> Result<ModelLoadPolicy> {
    let descriptor = static_model_descriptor(model_type);
    let capabilities = model_capabilities(model_type);
    let directory_route = directory_load_route(model_type, config)?;
    let weight_route = if capabilities.adapter_unsupported_message.is_some() {
        None
    } else {
        descriptor.adapter_weight_route
    };

    Ok(ModelLoadPolicy {
        descriptor,
        capabilities,
        directory_route,
        weight_route,
    })
}
