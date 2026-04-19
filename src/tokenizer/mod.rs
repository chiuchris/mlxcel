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

mod tiktoken;

use anyhow::Result;
use hf_hub::api::sync::Api;
use sentencepiece::SentencePieceProcessor;
use std::collections::HashMap;
use std::path::Path;

pub use tiktoken::TiktokenTokenizer;

/// Unified tokenizer supporting HuggingFace (tokenizer.json), SentencePiece (tokenizer.model),
/// and Tiktoken (.tiktoken) formats
pub enum MlxcelTokenizer {
    HuggingFace(tokenizers::Tokenizer),
    SentencePiece(SentencePieceTokenizer),
    Tiktoken(TiktokenTokenizer),
}

pub struct SentencePieceTokenizer {
    processor: SentencePieceProcessor,
    special_token_to_id: HashMap<String, u32>,
    id_to_special_token: HashMap<u32, String>,
    /// Special tokens sorted by length descending for greedy longest-match-first splitting
    special_tokens_sorted: Vec<(String, u32)>,
    bos_id: Option<u32>,
    add_bos: bool,
}

impl MlxcelTokenizer {
    /// Create a stub tokenizer for unit tests.
    ///
    /// The stub returns empty/identity results; it exists so that types like
    /// `StreamingDecodeState` can be constructed without loading a real model.
    #[cfg(test)]
    pub(crate) fn stub() -> Self {
        // Build a minimal HuggingFace tokenizer with a single-character
        // alphabet so encode/decode never panic.
        use tokenizers::models::bpe::BPE;
        let model = BPE::default();
        let tokenizer = tokenizers::Tokenizer::new(model);
        Self::HuggingFace(tokenizer)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        match self {
            Self::HuggingFace(t) => {
                let encoding = t
                    .encode(text, add_special_tokens)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                Ok(encoding.get_ids().to_vec())
            }
            Self::SentencePiece(t) => t.encode(text, add_special_tokens),
            Self::Tiktoken(t) => t.encode(text, add_special_tokens),
        }
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self {
            Self::HuggingFace(t) => t
                .decode(ids, skip_special_tokens)
                .map_err(|e| anyhow::anyhow!("Decode failed: {}", e)),
            Self::SentencePiece(t) => t.decode(ids, skip_special_tokens),
            Self::Tiktoken(t) => t.decode(ids, skip_special_tokens),
        }
    }

    /// Returns the underlying HuggingFace `tokenizers::Tokenizer` when this
    /// instance was constructed from a `tokenizer.json` file.
    ///
    /// `None` for SentencePiece or Tiktoken tokenizers. Used by Axis B
    /// (#362) language steering to feed the tokenizer vocabulary into the
    /// [`mlxcel_core::lang_analyzer`] classifier.
    pub fn hf_tokenizer(&self) -> Option<&tokenizers::Tokenizer> {
        match self {
            Self::HuggingFace(t) => Some(t),
            Self::SentencePiece(_) | Self::Tiktoken(_) => None,
        }
    }
}

impl SentencePieceTokenizer {
    fn new(
        processor: SentencePieceProcessor,
        special_tokens: HashMap<String, u32>,
        bos_id: Option<u32>,
        add_bos: bool,
    ) -> Self {
        let id_to_special_token: HashMap<u32, String> = special_tokens
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        let mut special_tokens_sorted: Vec<(String, u32)> = special_tokens
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        // Sort by length descending for greedy longest-match-first
        special_tokens_sorted.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Self {
            processor,
            special_token_to_id: special_tokens,
            id_to_special_token,
            special_tokens_sorted,
            bos_id,
            add_bos,
        }
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut result = Vec::new();

        // Prepend BOS if configured
        if add_special_tokens
            && self.add_bos
            && let Some(bos) = self.bos_id
        {
            result.push(bos);
        }

        if self.special_tokens_sorted.is_empty() {
            // No special tokens to handle — encode directly
            let pieces = self
                .processor
                .encode(text)
                .map_err(|e| anyhow::anyhow!("SentencePiece encode failed: {}", e))?;
            for piece in &pieces {
                result.push(piece.id);
            }
            return Ok(result);
        }

        // Split text at special token boundaries (greedy longest-match-first)
        let segments = self.split_with_special_tokens(text);

        for segment in segments {
            if let Some(&id) = self.special_token_to_id.get(&segment) {
                // This segment is a special token — insert its ID directly
                result.push(id);
            } else {
                // Regular text — encode via sentencepiece
                let pieces = self
                    .processor
                    .encode(&segment)
                    .map_err(|e| anyhow::anyhow!("SentencePiece encode failed: {}", e))?;
                for piece in &pieces {
                    result.push(piece.id);
                }
            }
        }

        Ok(result)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut result = String::new();
        let mut regular_ids: Vec<u32> = Vec::new();

        for &id in ids {
            if let Some(special) = self.id_to_special_token.get(&id) {
                // Flush any accumulated regular IDs first
                if !regular_ids.is_empty() {
                    let text = self
                        .processor
                        .decode_piece_ids(&regular_ids)
                        .map_err(|e| anyhow::anyhow!("SentencePiece decode failed: {}", e))?;
                    result.push_str(&text);
                    regular_ids.clear();
                }
                if !skip_special_tokens {
                    result.push_str(special);
                }
            } else {
                regular_ids.push(id);
            }
        }

        // Flush remaining regular IDs
        if !regular_ids.is_empty() {
            let text = self
                .processor
                .decode_piece_ids(&regular_ids)
                .map_err(|e| anyhow::anyhow!("SentencePiece decode failed: {}", e))?;
            result.push_str(&text);
        }

        Ok(result)
    }

    /// Split text into segments, alternating between special tokens and regular text.
    /// Uses greedy longest-match-first strategy.
    fn split_with_special_tokens(&self, text: &str) -> Vec<String> {
        let mut segments = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Try to match a special token at the current position
            let mut matched = false;
            for (token, _id) in &self.special_tokens_sorted {
                if remaining.starts_with(token.as_str()) {
                    segments.push(token.clone());
                    remaining = &remaining[token.len()..];
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Find the next special token occurrence
                let mut next_pos = remaining.len();
                for (token, _id) in &self.special_tokens_sorted {
                    if let Some(pos) = remaining.find(token.as_str())
                        && pos < next_pos
                    {
                        next_pos = pos;
                    }
                }
                // Everything before the next special token is regular text
                segments.push(remaining[..next_pos].to_string());
                remaining = &remaining[next_pos..];
            }
        }

        segments
    }
}

/// Parse special tokens from tokenizer_config.json's `added_tokens_decoder` field
fn parse_special_tokens(model_path: &Path) -> (HashMap<String, u32>, bool) {
    let config_path = model_path.join("tokenizer_config.json");
    let mut special_tokens = HashMap::new();
    let mut add_bos = false;

    if let Ok(content) = std::fs::read_to_string(&config_path)
        && let Ok(config) = serde_json::from_str::<serde_json::Value>(&content)
    {
        // Parse add_bos_token
        if let Some(v) = config.get("add_bos_token").and_then(|v| v.as_bool()) {
            add_bos = v;
        }

        // Parse added_tokens_decoder: { "128132": { "content": "<|im_start|>", "special": true }, ... }
        if let Some(decoder) = config
            .get("added_tokens_decoder")
            .and_then(|v| v.as_object())
        {
            for (id_str, entry) in decoder {
                if let (Ok(id), Some(content)) = (
                    id_str.parse::<u32>(),
                    entry.get("content").and_then(|v| v.as_str()),
                ) {
                    let is_special = entry
                        .get("special")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if is_special {
                        special_tokens.insert(content.to_string(), id);
                    }
                }
            }
        }
    }

    (special_tokens, add_bos)
}

/// Find a `.tiktoken` file in the model directory.
/// Tries `tiktoken.model` first, then any `*.tiktoken` file.
fn find_tiktoken_file(model_path: &Path) -> Option<std::path::PathBuf> {
    // Try tiktoken.model first (standard name used by some models)
    let tiktoken_model = model_path.join("tiktoken.model");
    if tiktoken_model.exists() {
        return Some(tiktoken_model);
    }

    // Try any *.tiktoken file
    let pattern = model_path.join("*.tiktoken");
    if let Ok(paths) = glob::glob(pattern.to_str()?) {
        return paths.flatten().next();
    }
    None
}

fn remote_tokenizer_repo_for_model_type(model_type: &str) -> Option<&'static str> {
    match model_type {
        "moondream3" => Some("moondream/starmie-v1"),
        _ => None,
    }
}

fn remote_tokenizer_repo_for_model(model_path: &Path) -> Option<&'static str> {
    let config_path = model_path.join("config.json");
    let content = std::fs::read_to_string(config_path).ok()?;
    let config = serde_json::from_str::<serde_json::Value>(&content).ok()?;
    let model_type = config.get("model_type").and_then(|value| value.as_str())?;
    remote_tokenizer_repo_for_model_type(model_type)
}

fn download_remote_tokenizer(repo_id: &str) -> Result<tokenizers::Tokenizer> {
    let api = Api::new()
        .map_err(|err| anyhow::anyhow!("Failed to initialize Hugging Face API: {}", err))?;
    let repo = api.model(repo_id.to_string());
    let tokenizer_path = repo.get("tokenizer.json").map_err(|err| {
        anyhow::anyhow!(
            "Failed to download tokenizer.json from {}: {}",
            repo_id,
            err
        )
    })?;
    tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|err| anyhow::anyhow!(err))
}

pub fn load_tokenizer(model_path: &Path) -> Result<MlxcelTokenizer> {
    // Try HuggingFace tokenizer.json first
    let tokenizer_json_path = model_path.join("tokenizer.json");
    if tokenizer_json_path.exists() {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_json_path)
            .map_err(|e| anyhow::anyhow!(e))?;
        return Ok(MlxcelTokenizer::HuggingFace(tokenizer));
    }

    // Fall back to SentencePiece tokenizer.model
    let tokenizer_model_path = model_path.join("tokenizer.model");
    if tokenizer_model_path.exists() {
        let processor = SentencePieceProcessor::open(&tokenizer_model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer.model: {}", e))?;

        let bos_id = processor.bos_id();

        let (special_tokens, add_bos) = parse_special_tokens(model_path);

        let sp_tokenizer = SentencePieceTokenizer::new(processor, special_tokens, bos_id, add_bos);
        return Ok(MlxcelTokenizer::SentencePiece(sp_tokenizer));
    }

    // Fall back to tiktoken (.tiktoken files)
    if let Some(tiktoken_path) = find_tiktoken_file(model_path) {
        let tokenizer = TiktokenTokenizer::from_file(&tiktoken_path, model_path)?;
        return Ok(MlxcelTokenizer::Tiktoken(tokenizer));
    }

    if let Some(repo_id) = remote_tokenizer_repo_for_model(model_path) {
        let tokenizer = download_remote_tokenizer(repo_id).map_err(|err| {
            anyhow::anyhow!(
                "Failed to resolve fallback tokenizer {} for {:?}: {}",
                repo_id,
                model_path,
                err
            )
        })?;
        return Ok(MlxcelTokenizer::HuggingFace(tokenizer));
    }

    Err(anyhow::anyhow!(
        "No tokenizer found in {:?} (tried tokenizer.json, tokenizer.model, and *.tiktoken)",
        model_path
    ))
}

#[cfg(test)]
mod tests {
    use super::{remote_tokenizer_repo_for_model, remote_tokenizer_repo_for_model_type};

    #[test]
    fn remote_tokenizer_repo_for_model_type_matches_moondream3() {
        assert_eq!(
            remote_tokenizer_repo_for_model_type("moondream3"),
            Some("moondream/starmie-v1")
        );
        assert_eq!(remote_tokenizer_repo_for_model_type("llama"), None);
    }

    #[test]
    fn remote_tokenizer_repo_for_model_reads_config_json_model_type() {
        let temp_dir =
            std::env::temp_dir().join(format!("mlxcel-tokenizer-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::fs::write(
            temp_dir.join("config.json"),
            r#"{"model_type":"moondream3"}"#,
        )
        .unwrap();

        assert_eq!(
            remote_tokenizer_repo_for_model(&temp_dir),
            Some("moondream/starmie-v1")
        );

        let _ = std::fs::remove_dir_all(temp_dir);
    }
}
