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

//! CLI text-generation command handler.
//!
//! This module keeps the user-facing `generate` flow readable by separating
//! prompt preparation, generation-mode selection, and terminal output helpers.

use anyhow::Result;
use std::io::{self, Write as IoWrite};
use std::path::Path;
use std::time::{Duration, Instant};

use mlxcel::{
    CxxGenerator, GenerationStats, LanguageModel, RuntimeSetup, SamplingConfig,
    SpeculativeGenerator, initialize_runtime, load_model, load_model_with_adapter,
    sampling::{ResolvedSamplingParams, build_sampling_config},
    server::chat_template::{ChatMessage, ChatTemplateProcessor},
    vision::merge::InputEmbeddings,
    vlm_runtime::prepared_embedding_refs,
};

use super::generate_vlm;
use crate::GenerateArgs;

fn generation_stats_from_duration(
    prompt_tokens: usize,
    generated_tokens: usize,
    total_time: Duration,
) -> GenerationStats {
    let decode_time_ms = total_time.as_secs_f64() * 1000.0;
    let decode_tok_per_sec = if total_time.as_secs_f64() > 0.0 {
        generated_tokens as f64 / total_time.as_secs_f64()
    } else {
        0.0
    };

    GenerationStats {
        prompt_tokens,
        generated_tokens,
        prefill_time_ms: 0.0,
        decode_time_ms,
        prefill_tok_per_sec: 0.0,
        decode_tok_per_sec,
    }
}

fn print_runtime_setup(runtime: &RuntimeSetup) {
    if let Some(invalid) = runtime.invalid_device_override.as_deref() {
        eprintln!(
            "Ignoring invalid MLXCEL_DEVICE value {:?}; using gpu.",
            invalid
        );
    }
    println!("Runtime device: {}", runtime.device);
    if let Some(max_memory) = runtime.wired_limit_bytes {
        println!(
            "Wired memory limit: {:.1} GB",
            max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else if runtime.device == mlxcel::RuntimeDevice::Gpu {
        let max_memory = mlxcel_core::gpu_max_memory_size();
        println!(
            "GPU memory: {:.1} GB (no wired limit)",
            max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
}

fn load_generation_model(
    args: &GenerateArgs,
) -> Result<(mlxcel::LoadedModel, mlxcel::tokenizer::MlxcelTokenizer)> {
    println!("Loading model from {:?}...", args.model.model);
    let load_start = Instant::now();
    let result = if let Some(ref adapter_path) = args.model.adapter {
        println!("Loading LoRA adapter from {:?}...", adapter_path);
        load_model_with_adapter(&args.model.model, adapter_path)
    } else {
        load_model(&args.model.model)
    }?;
    let load_elapsed = load_start.elapsed();
    println!("Model loaded in {:.3}s.", load_elapsed.as_secs_f64());
    Ok(result)
}

fn apply_user_chat_template(processor: &ChatTemplateProcessor, user_prompt: &str) -> String {
    let messages = [ChatMessage {
        role: "user".to_string(),
        content: user_prompt.to_string(),
    }];

    processor
        .apply(&messages)
        .unwrap_or_else(|_| user_prompt.to_string())
}

fn resolve_cli_prompt(
    user_prompt: &str,
    no_chat_template: bool,
    processor: Option<&ChatTemplateProcessor>,
) -> String {
    if no_chat_template {
        return user_prompt.to_string();
    }

    processor.map_or_else(
        || user_prompt.to_string(),
        |processor| apply_user_chat_template(processor, user_prompt),
    )
}

fn load_cli_prompt(model_path: &Path, user_prompt: &str, no_chat_template: bool) -> String {
    let processor = if no_chat_template {
        None
    } else {
        ChatTemplateProcessor::from_model_path(model_path)
            .ok()
            .flatten()
    };

    resolve_cli_prompt(user_prompt, no_chat_template, processor.as_ref())
}

fn tokenize_prompt(
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    prompt: &str,
) -> Result<Vec<i32>> {
    let prompt_token_ids = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    Ok(prompt_token_ids.iter().map(|&x| x as i32).collect())
}

fn build_cli_sampling_config(args: &GenerateArgs, stop_token_ids: Vec<i32>) -> SamplingConfig {
    build_sampling_config(ResolvedSamplingParams {
        temperature: args.sampling.temp,
        top_k: args.sampling.top_k,
        top_p: args.sampling.top_p,
        min_p: args.sampling.min_p,
        seed: None,
        repetition_penalty: args.sampling.repetition_penalty,
        dry_multiplier: args.sampling.dry_multiplier,
        dry_base: args.sampling.dry_base,
        dry_allowed_length: args.sampling.dry_allowed_length,
        dry_penalty_last_n: args.sampling.dry_penalty_last_n,
        dry_sequence_breakers: Vec::new(),
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop_token_ids,
    })
}

fn print_generation_preamble(user_prompt: &str) -> Result<()> {
    println!("Generating...");
    print!("{}", user_prompt);
    io::stdout().flush()?;
    Ok(())
}

fn generated_suffix<'a>(full_text: &'a str, prompt_text: &str) -> &'a str {
    full_text.strip_prefix(prompt_text).unwrap_or(full_text)
}

fn decode_generated_text(
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    prompt_tokens: &[i32],
    generated_tokens: &[i32],
) -> String {
    let all_tokens: Vec<u32> = prompt_tokens
        .iter()
        .map(|&x| x as u32)
        .chain(generated_tokens.iter().map(|&x| x as u32))
        .collect();
    let full_text = tokenizer.decode(&all_tokens, false).unwrap_or_default();
    let prompt_decoded = tokenizer
        .decode(
            &prompt_tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            false,
        )
        .unwrap_or_default();

    generated_suffix(&full_text, &prompt_decoded).to_string()
}

fn print_generation_result(
    generated_text: &str,
    stats: &GenerationStats,
    profile: bool,
) -> Result<()> {
    print!("{}", generated_text);
    io::stdout().flush()?;

    println!();
    println!();

    if profile {
        println!("[Profile Results]");
        stats.print();
    } else {
        let total_time_sec = stats.decode_time_ms / 1000.0;
        println!(
            "[Generated {} tokens in {:.2}s = {:.2} tok/s]",
            stats.generated_tokens, total_time_sec, stats.decode_tok_per_sec
        );
    }

    Ok(())
}

fn generate_standard<M: LanguageModel>(
    model: &M,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    profile: bool,
) -> (Vec<i32>, GenerationStats) {
    let mut generator = CxxGenerator::new(model.num_layers());

    if profile {
        return generator.generate_with_stats(model, prompt_tokens, max_tokens, sampling_config);
    }

    let _ = generator.generate(model, prompt_tokens, 1, sampling_config);
    generator.reset_with_model(model);

    let start_time = Instant::now();
    let tokens = generator.generate(model, prompt_tokens, max_tokens, sampling_config);
    let total_time = start_time.elapsed();
    let generated_len = tokens.len();

    (
        tokens,
        generation_stats_from_duration(prompt_tokens.len(), generated_len, total_time),
    )
}

fn generate_with_embeddings<M: LanguageModel>(
    model: &M,
    prompt_tokens: &[i32],
    embeddings: &InputEmbeddings,
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    profile: bool,
) -> Result<(Vec<i32>, GenerationStats)> {
    let mut generator = CxxGenerator::new(model.num_layers());
    let (input_embeds, mask_ref) = prepared_embedding_refs(embeddings)?;

    if profile {
        return Ok(generator.generate_with_stats_and_embeddings(
            model,
            prompt_tokens,
            Some(input_embeds),
            mask_ref,
            max_tokens,
            sampling_config,
        ));
    }

    let start_time = Instant::now();
    let tokens = generator.generate_streaming_with_embeddings(
        model,
        prompt_tokens,
        Some(input_embeds),
        mask_ref,
        max_tokens,
        sampling_config,
        |_| true,
    );
    let total_time = start_time.elapsed();
    let generated_len = tokens.len();

    Ok((
        tokens,
        generation_stats_from_duration(prompt_tokens.len(), generated_len, total_time),
    ))
}

fn run_generation_mode<M: LanguageModel>(
    model: &M,
    args: &GenerateArgs,
    prompt_tokens: &[i32],
    sampling_config: &SamplingConfig,
    vlm_embeddings: Option<&InputEmbeddings>,
) -> Result<(Vec<i32>, GenerationStats)> {
    let output = if let Some(ref draft_model_path) = args.model.draft_model {
        println!("Loading draft model from {:?}...", draft_model_path);
        let (draft_model, _draft_tokenizer) = load_model(draft_model_path)?;
        println!("Draft model loaded.");

        let draft_num_layers = draft_model.num_layers();
        let main_num_layers = model.num_layers();
        let mut spec_generator = SpeculativeGenerator::new(main_num_layers, draft_num_layers);

        spec_generator.generate(
            model,
            &draft_model,
            prompt_tokens,
            args.generation.max_tokens,
            args.model.num_draft_tokens,
            sampling_config,
        )
    } else if let Some(embeddings) = vlm_embeddings {
        generate_with_embeddings(
            model,
            prompt_tokens,
            embeddings,
            args.generation.max_tokens,
            sampling_config,
            args.generation.profile,
        )?
    } else {
        generate_standard(
            model,
            prompt_tokens,
            args.generation.max_tokens,
            sampling_config,
            args.generation.profile,
        )
    };

    Ok(output)
}

pub(crate) fn run_generate(args: GenerateArgs) -> Result<()> {
    let runtime = initialize_runtime();
    print_runtime_setup(&runtime);

    let (model, tokenizer) = load_generation_model(&args)?;
    let prompt = load_cli_prompt(
        &args.model.model,
        &args.generation.prompt,
        args.generation.no_chat_template,
    );
    let mut prompt_tokens = tokenize_prompt(&tokenizer, &prompt)?;

    print_generation_preamble(&args.generation.prompt)?;

    let sampling_config =
        build_cli_sampling_config(&args, mlxcel::read_eos_token_ids(&args.model.model));

    // Check for VLM image mode
    let vlm_embeddings = generate_vlm::compute_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        &prompt,
        &args.generation.image,
        &tokenizer,
    )?;

    let (generated_tokens, stats) = run_generation_mode(
        &model,
        &args,
        &prompt_tokens,
        &sampling_config,
        vlm_embeddings.as_ref(),
    )?;
    let generated_text = decode_generated_text(&tokenizer, &prompt_tokens, &generated_tokens);
    print_generation_result(&generated_text, &stats, args.generation.profile)?;

    // Cleanup
    mlxcel_core::clear_memory_cache();

    Ok(())
}

#[cfg(test)]
#[path = "generate_tests.rs"]
mod tests;
