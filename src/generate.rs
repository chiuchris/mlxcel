use anyhow::Result;
use std::io::{self, Write as IoWrite};
use std::time::Instant;

use mlxcel::{
    CxxGenerator, GenerationStats, LanguageModel, SamplingConfig, SpeculativeGenerator, load_model,
    load_model_with_adapter,
    server::chat_template::{ChatMessage, ChatTemplateProcessor},
};

use super::{GenerateArgs, generate_vlm};

pub(crate) fn run_generate(args: GenerateArgs) -> Result<()> {
    // Set wired memory limit for GPU
    let max_memory = mlxcel_core::gpu_max_memory_size();
    mlxcel_core::set_wired_limit(max_memory);
    println!(
        "Wired memory limit: {:.1} GB",
        max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    println!("Loading model from {:?}...", args.model.model);
    let (model, tokenizer) = if let Some(ref adapter_path) = args.model.adapter {
        println!("Loading LoRA adapter from {:?}...", adapter_path);
        load_model_with_adapter(&args.model.model, adapter_path)?
    } else {
        load_model(&args.model.model)?
    };
    println!("Model loaded.");

    // Apply chat template if available (unless --no-chat-template is set)
    let prompt = if args.generation.no_chat_template {
        args.generation.prompt.clone()
    } else {
        match ChatTemplateProcessor::from_model_path(&args.model.model) {
            Ok(Some(processor)) => {
                let messages = vec![ChatMessage {
                    role: "user".to_string(),
                    content: args.generation.prompt.clone(),
                }];
                match processor.apply(&messages) {
                    Ok(result) => result,
                    Err(_) => args.generation.prompt.clone(),
                }
            }
            _ => args.generation.prompt.clone(),
        }
    };

    // Tokenize prompt (add_special_tokens=true to include BOS for models that need it)
    let prompt_token_ids = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let mut prompt_tokens: Vec<i32> = prompt_token_ids.iter().map(|&x| x as i32).collect();

    // For VLM: ensure image tokens are in the prompt
    let has_images = !args.generation.image.is_empty();
    if has_images && model.is_vlm() {
        generate_vlm::prepare_vlm_tokens(
            &model,
            &mut prompt_tokens,
            &prompt,
            &args.generation.image,
            &tokenizer,
        )?;
    }

    println!("Generating...");
    print!("{}", args.generation.prompt);
    io::stdout().flush()?;

    // Read EOS tokens from generation_config.json
    let config_eos = mlxcel::read_eos_token_ids(&args.model.model);

    // Create sampling config
    let sampling_config = if args.sampling.temp <= 0.0 {
        SamplingConfig {
            min_p: args.sampling.min_p,
            repetition_penalty: args.sampling.repetition_penalty,
            dry_multiplier: args.sampling.dry_multiplier,
            dry_base: args.sampling.dry_base,
            dry_allowed_length: args.sampling.dry_allowed_length,
            dry_penalty_last_n: args.sampling.dry_penalty_last_n,
            stop_token_ids: config_eos.clone(),
            ..SamplingConfig::greedy()
        }
    } else {
        SamplingConfig {
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
            stop_token_ids: config_eos,
        }
    };

    // Check for VLM image mode
    let vlm_embeddings = generate_vlm::compute_vlm_embeddings(
        &model,
        &prompt_tokens,
        &args.generation.image,
        has_images,
    )?;

    // Generate tokens (speculative or standard)
    let (generated_tokens, stats) = if let Some(ref draft_model_path) = args.model.draft_model {
        // Speculative decoding mode
        println!("Loading draft model from {:?}...", draft_model_path);
        let (draft_model, _draft_tokenizer) = load_model(draft_model_path)?;
        println!("Draft model loaded.");

        let draft_num_layers = draft_model.num_layers();
        let main_num_layers = model.num_layers();
        let mut spec_generator = SpeculativeGenerator::new(main_num_layers, draft_num_layers);

        spec_generator.generate(
            &model,
            &draft_model,
            &prompt_tokens,
            args.generation.max_tokens,
            args.model.num_draft_tokens,
            &sampling_config,
        )
    } else if let Some(ref embeddings) = vlm_embeddings {
        // VLM generation with pre-computed embeddings
        let num_layers = model.num_layers();
        let mut generator = CxxGenerator::new(num_layers);

        let mask_ref = embeddings
            .attention_mask_4d
            .as_ref()
            .map(|m| m.as_ref().unwrap());

        if args.generation.profile {
            generator.generate_with_stats_and_embeddings(
                &model,
                &prompt_tokens,
                Some(embeddings.inputs_embeds.as_ref().unwrap()),
                mask_ref,
                args.generation.max_tokens,
                &sampling_config,
            )
        } else {
            let start_time = Instant::now();
            let tokens = generator.generate_streaming_with_embeddings(
                &model,
                &prompt_tokens,
                Some(embeddings.inputs_embeds.as_ref().unwrap()),
                mask_ref,
                args.generation.max_tokens,
                &sampling_config,
                |_| true,
            );
            let total_time = start_time.elapsed();

            let stats = GenerationStats {
                prompt_tokens: prompt_tokens.len(),
                generated_tokens: tokens.len(),
                prefill_time_ms: 0.0,
                decode_time_ms: total_time.as_secs_f64() * 1000.0,
                prefill_tok_per_sec: 0.0,
                decode_tok_per_sec: if total_time.as_secs_f64() > 0.0 {
                    tokens.len() as f64 / total_time.as_secs_f64()
                } else {
                    0.0
                },
            };

            (tokens, stats)
        }
    } else {
        // Standard generation
        let num_layers = model.num_layers();
        let mut generator = CxxGenerator::new(num_layers);

        if args.generation.profile {
            // Profile mode: use generate_with_stats for detailed timing
            generator.generate_with_stats(
                &model,
                &prompt_tokens,
                args.generation.max_tokens,
                &sampling_config,
            )
        } else {
            // Normal mode: warmup + regular generation
            let _ = generator.generate(&model, &prompt_tokens, 1, &sampling_config);
            generator.reset_with_model(&model);

            let start_time = Instant::now();
            let tokens = generator.generate(
                &model,
                &prompt_tokens,
                args.generation.max_tokens,
                &sampling_config,
            );
            let total_time = start_time.elapsed();

            // Create stats from total time (no prefill/decode separation)
            let stats = GenerationStats {
                prompt_tokens: prompt_tokens.len(),
                generated_tokens: tokens.len(),
                prefill_time_ms: 0.0,
                decode_time_ms: total_time.as_secs_f64() * 1000.0,
                prefill_tok_per_sec: 0.0,
                decode_tok_per_sec: if total_time.as_secs_f64() > 0.0 {
                    tokens.len() as f64 / total_time.as_secs_f64()
                } else {
                    0.0
                },
            };

            (tokens, stats)
        }
    };

    // Decode and print tokens
    // We need to decode with context to get proper spacing for sentencepiece tokenizers
    // The simplest approach is to decode prompt+generated and strip the prompt part
    let all_tokens: Vec<u32> = prompt_tokens
        .iter()
        .map(|&x| x as u32)
        .chain(generated_tokens.iter().map(|&x| x as u32))
        .collect();
    let full_text = tokenizer.decode(&all_tokens, false).unwrap_or_default();
    // Strip the prompt (decode prompt alone to get exact length)
    let prompt_decoded = tokenizer
        .decode(
            &prompt_tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            false,
        )
        .unwrap_or_default();
    let generated_text = &full_text[prompt_decoded.len()..];
    print!("{}", generated_text);
    io::stdout().flush()?;

    // Print stats
    println!();
    println!();

    if args.generation.profile {
        // Detailed profile output
        println!("[Profile Results]");
        stats.print();
    } else {
        // Simple output for normal mode
        let total_time_sec = stats.decode_time_ms / 1000.0;
        println!(
            "[Generated {} tokens in {:.2}s = {:.2} tok/s]",
            stats.generated_tokens, total_time_sec, stats.decode_tok_per_sec
        );
    }

    // Cleanup
    mlxcel_core::clear_memory_cache();

    Ok(())
}
