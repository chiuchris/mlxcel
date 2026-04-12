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

use std::cell::{RefCell, RefMut};
use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, copy};

use crate::models::gpt_oss::{Cache as GptOssCache, GptOssStageModel, gpt_oss_cache_offset};

use super::{FamilyStageExecutor, LayerFilter, StageExecutionInput, StageExecutionOutput};

pub struct GptOssStageExecutor {
    model: GptOssStageModel,
    cache_sets: RefCell<HashMap<usize, Vec<GptOssCache>>>,
}

impl GptOssStageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        Ok(Self {
            model: GptOssStageModel::load(model_dir, filter, stage_index)
                .map_err(anyhow::Error::msg)?,
            cache_sets: RefCell::new(HashMap::new()),
        })
    }

    fn cache_key(caches: &[KVCache]) -> usize {
        caches.as_ptr() as usize
    }

    fn sequence_needs_reset(external_caches: &[KVCache], internal_caches: &[GptOssCache]) -> bool {
        external_caches.iter().all(|cache| cache.offset == 0)
            && internal_caches
                .iter()
                .any(|cache| gpt_oss_cache_offset(cache) > 0)
    }

    fn sync_external_offsets(external_caches: &mut [KVCache], internal_caches: &[GptOssCache]) {
        for (external, internal) in external_caches.iter_mut().zip(internal_caches.iter()) {
            external.offset = gpt_oss_cache_offset(internal);
        }
    }

    fn caches_for_key<'a>(
        &'a self,
        cache_key: usize,
        external_caches: &[KVCache],
    ) -> RefMut<'a, Vec<GptOssCache>> {
        let needs_reset = {
            let cache_sets = self.cache_sets.borrow();
            cache_sets
                .get(&cache_key)
                .is_some_and(|internal_caches| {
                    Self::sequence_needs_reset(external_caches, internal_caches)
                })
        };

        let mut cache_sets = self.cache_sets.borrow_mut();
        if needs_reset || !cache_sets.contains_key(&cache_key) {
            cache_sets.insert(cache_key, self.model.make_caches());
        }
        RefMut::map(cache_sets, |cache_sets| {
            cache_sets
                .get_mut(&cache_key)
                .expect("gpt-oss sequence cache entry must exist")
        })
    }
}

impl FamilyStageExecutor for GptOssStageExecutor {
    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.model.num_layers()).map(|_| KVCache::new()).collect()
    }

    fn release_caches(&self, caches: &[KVCache]) {
        self.cache_sets
            .borrow_mut()
            .remove(&Self::cache_key(caches));
    }

    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        let cache_key = Self::cache_key(caches);
        let mut internal_caches = self.caches_for_key(cache_key, caches);

        let output = match input {
            StageExecutionInput::TokenIds(input_ids) => {
                self.model.execute_from_token_ids(input_ids, &mut internal_caches)
            }
            StageExecutionInput::HiddenStates(hidden_states) => self
                .model
                .execute_from_hidden_states(copy(hidden_states), &mut internal_caches),
        }
        .map_err(anyhow::Error::msg)?;

        Self::sync_external_offsets(caches, &internal_caches);
        Ok(output)
    }
}
