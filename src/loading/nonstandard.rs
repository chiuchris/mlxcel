//! Non-standard directory loading paths.
//!
//! These families still load directly from a model directory, but they do not
//! fit the standard config-backed text-model registry.

use anyhow::Result;
use std::fmt::Display;
use std::path::Path;

use crate::LoadedModel;
use crate::models::{self, ModelType};

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_nonstandard_model_type(model_type: ModelType) -> bool {
    matches!(
        model_type,
        ModelType::Qwen35
            | ModelType::Qwen35Moe
            | ModelType::Gemma3n
            | ModelType::Mamba
            | ModelType::Mamba2
            | ModelType::Jamba
            | ModelType::NemotronH
            | ModelType::NemotronNAS
            | ModelType::KimiLinear
            | ModelType::LongcatFlash
            | ModelType::LongcatFlashNgram
            | ModelType::Rwkv7
            | ModelType::RecurrentGemma
    )
}

pub(crate) fn try_load_nonstandard_model_from_dir(
    model_type: ModelType,
    model_path: &Path,
    path_str: &str,
) -> Result<Option<LoadedModel>> {
    Ok(match model_type {
        ModelType::Qwen35 => Some(
            super::load_pair_from_dir(path_str, models::Qwen35Model::load)
                .map(LoadedModel::Qwen35)?,
        ),
        ModelType::Qwen35Moe => Some(
            super::load_pair_from_dir(path_str, models::Qwen35Model::load)
                .map(LoadedModel::Qwen35Moe)?,
        ),
        ModelType::Gemma3n => {
            Some(load_from_dir(path_str, models::Gemma3nModel::load).map(LoadedModel::Gemma3n)?)
        }
        ModelType::Mamba => Some(
            super::load_pair_from_dir(path_str, |path| models::MambaModel::load(&path))
                .map(LoadedModel::Mamba)?,
        ),
        ModelType::Mamba2 => Some(
            super::load_pair_from_dir(path_str, |path| models::Mamba2Model::load(&path))
                .map(LoadedModel::Mamba2)?,
        ),
        ModelType::Jamba => Some(
            super::load_pair_from_dir(path_str, |path| models::JambaModel::load(&path))
                .map(LoadedModel::Jamba)?,
        ),
        ModelType::NemotronH => Some(
            super::load_pair_from_dir(path_str, |path| models::NemotronHModel::load(&path))
                .map(LoadedModel::NemotronH)?,
        ),
        ModelType::NemotronNAS => Some(
            super::load_pair_from_dir(path_str, |path| models::NemotronNASModel::load(&path))
                .map(LoadedModel::NemotronNAS)?,
        ),
        ModelType::KimiLinear => Some(
            super::load_pair_from_dir(path_str, models::KimiLinearModel::load)
                .map(LoadedModel::KimiLinear)?,
        ),
        ModelType::LongcatFlash => Some(
            super::load_pair_from_dir(path_str, |path| models::LongcatFlashNgramModel::load(&path))
                .map(LoadedModel::LongcatFlash)?,
        ),
        ModelType::LongcatFlashNgram => Some(
            super::load_pair_from_dir(path_str, |path| models::LongcatFlashNgramModel::load(&path))
                .map(LoadedModel::LongcatFlashNgram)?,
        ),
        ModelType::Rwkv7 => {
            Some(load_from_path(model_path, models::Rwkv7::load).map(LoadedModel::Rwkv7)?)
        }
        ModelType::RecurrentGemma => Some(
            super::load_pair_from_dir(path_str, |path| models::GriffinModel::load(&path))
                .map(LoadedModel::RecurrentGemma)?,
        ),
        _ => None,
    })
}

fn load_from_dir<T, E, F>(path_str: &str, load: F) -> Result<T>
where
    F: FnOnce(String) -> std::result::Result<T, E>,
    E: Display,
{
    load(path_str.to_owned()).map_err(|err| anyhow::anyhow!("{}", err))
}

fn load_from_path<T, E, F>(path: &Path, load: F) -> Result<T>
where
    F: FnOnce(&Path) -> std::result::Result<T, E>,
    E: Display,
{
    load(path).map_err(|err| anyhow::anyhow!("{}", err))
}

#[cfg(test)]
#[path = "nonstandard_tests.rs"]
mod tests;
