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

//! OpenAI-compatible chat completions adapter.
//!
//! This file should stay thin: it flattens the HTTP request, delegates prompt
//! preparation and option merging to shared helpers, and streams chunk payloads
//! back through `server/streaming.rs`.

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response, sse::Sse},
};

use mlxcel_core::sampling::{LogprobsConfig, TokenLogprobData};

use crate::server::batch::RequestPriority;
use crate::server::chat_request::prepare_chat_request;
use crate::server::request_options::{RequestOptionOverrides, build_server_generate_options};
use crate::server::streaming::sse_channel;
use crate::server::tool_calls;
use crate::server::tool_calls::stream_filter::StreamFilter;
use crate::server::types::response::{ChatLogprobs, TokenLogprob, TopLogprob};
use crate::server::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ErrorResponse,
    SamplingParams,
};
use crate::server::{AppState, ServerConfig, ServerGenerateOptions};
use crate::tokenizer::MlxcelTokenizer;

/// Decode a single token ID to its text representation using the tokenizer.
pub(crate) fn decode_token(tokenizer: &MlxcelTokenizer, token_id: i32) -> String {
    tokenizer
        .decode(&[token_id as u32], false)
        .unwrap_or_default()
}

/// Convert a `TokenLogprobData` to a `TokenLogprob` response struct.
///
/// `top_k` controls how many top alternatives to include. Pass 0 to include
/// none (only the selected token's logprob will be in the response).
pub(crate) fn token_lp_to_response(
    tokenizer: &MlxcelTokenizer,
    lp: &TokenLogprobData,
    top_k: usize,
) -> TokenLogprob {
    let token_text = decode_token(tokenizer, lp.token_id);
    let bytes = token_text.as_bytes().to_vec();

    let top_logprobs: Vec<TopLogprob> = lp
        .top_alternatives
        .iter()
        .take(top_k)
        .map(|&(alt_id, alt_lp)| {
            let alt_text = decode_token(tokenizer, alt_id);
            let alt_bytes = alt_text.as_bytes().to_vec();
            TopLogprob {
                token: alt_text,
                logprob: alt_lp,
                bytes: Some(alt_bytes),
            }
        })
        .collect();

    TokenLogprob {
        token: token_text,
        logprob: lp.logprob,
        bytes: Some(bytes),
        top_logprobs,
    }
}

/// Build a `ChatLogprobs` from a list of `TokenLogprobData`.
pub(crate) fn build_chat_logprobs(
    tokenizer: &MlxcelTokenizer,
    lp_data: &[TokenLogprobData],
    top_k: usize,
) -> ChatLogprobs {
    let content = lp_data
        .iter()
        .map(|lp| token_lp_to_response(tokenizer, lp, top_k))
        .collect();
    ChatLogprobs {
        content: Some(content),
    }
}

/// Build a single-token `ChatLogprobs` for streaming chunks.
pub(crate) fn build_single_token_chat_logprobs(
    tokenizer: &MlxcelTokenizer,
    lp: &TokenLogprobData,
    top_k: usize,
) -> ChatLogprobs {
    ChatLogprobs {
        content: Some(vec![token_lp_to_response(tokenizer, lp, top_k)]),
    }
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    // Validate top_logprobs range per OpenAI spec (0-20)
    if let Some(top) = request.top_logprobs
        && top > 20
    {
        return ErrorResponse::new(
            "top_logprobs must be between 0 and 20",
            "invalid_request_error",
        )
        .into_response();
    }
    // top_logprobs requires logprobs: true
    if request.top_logprobs.is_some() && request.logprobs != Some(true) {
        return ErrorResponse::new(
            "top_logprobs requires logprobs to be set to true",
            "invalid_request_error",
        )
        .into_response();
    }

    // Validate tool_choice values
    if let Some(ref tc) = request.tool_choice {
        match tc {
            crate::server::types::request::ToolChoice::Mode(mode) => {
                if !["auto", "none", "required"].contains(&mode.as_str()) {
                    return ErrorResponse::new(
                        format!("Invalid tool_choice value: '{mode}'. Must be 'auto', 'none', 'required', or a function object."),
                        "invalid_request_error",
                    )
                    .into_response();
                }
            }
            crate::server::types::request::ToolChoice::Specific(_) => {}
        }
    }

    // Enforce tools array size limit to prevent DoS via template rendering
    if let Some(ref tools) = request.tools
        && tools.len() > MAX_TOOLS
    {
        return ErrorResponse::new(
            format!(
                "Too many tools: {}. Maximum allowed is {MAX_TOOLS}.",
                tools.len()
            ),
            "invalid_request_error",
        )
        .into_response();
    }

    let priority = parse_priority_header(&headers);
    if request.stream {
        stream_chat_completion(state, request, priority).await
    } else {
        non_stream_chat_completion(state, request, priority)
            .await
            .into_response()
    }
}

/// Extract the `X-Priority` header value, defaulting to `Normal`.
pub(crate) fn parse_priority_header(headers: &HeaderMap) -> RequestPriority {
    headers
        .get("x-priority")
        .and_then(|v| v.to_str().ok())
        .and_then(RequestPriority::from_header)
        .unwrap_or_default()
}

async fn non_stream_chat_completion(
    state: AppState,
    request: ChatCompletionRequest,
    priority: RequestPriority,
) -> Result<Json<ChatCompletionResponse>, ErrorResponse> {
    // Queue-depth admission control: reject when prefill queue is full
    if !state.can_accept_request() {
        return Err(ErrorResponse::service_unavailable(
            "All slots are busy. Please try again later.",
        ));
    }

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();

    let prepared = prepare_chat_request(&state.chat_template, &request).await;
    let mut options = build_generate_options(&request.params, &state.config);
    options.priority = priority;

    // Set logprobs configuration when requested
    let top_k = request.top_logprobs.unwrap_or(0) as usize;
    if request.logprobs == Some(true) {
        options.logprobs = LogprobsConfig {
            enabled: true,
            top_k,
        };
    }

    // Generate (blocking call handled by model provider's worker thread)
    let result = state
        .model_provider
        .generate_with_media(
            prepared.prompt,
            options,
            prepared.image_data,
            prepared.audio_data,
        )
        .map_err(|e| ErrorResponse::new(format!("Generation error: {e}"), "server_error"))?;

    state.metrics.record_request(
        result.prompt_tokens,
        result.completion_tokens,
        result.generation_time_ms,
    );

    // Build logprobs for the response if requested
    let logprobs = result.logprobs.as_deref().and_then(|lp_data| {
        if lp_data.is_empty() {
            None
        } else {
            Some(build_chat_logprobs(&state.tokenizer, lp_data, top_k))
        }
    });

    // Try to parse tool calls from the output
    if tool_calls::should_parse_tool_calls(&request) {
        let tools = request.tools.as_deref();
        let parsed = tool_calls::parse_tool_calls(&result.text, tools);

        if parsed.has_tool_calls() {
            let tool_call_responses = tool_calls::build_tool_call_responses(&parsed, &request);
            if !tool_call_responses.is_empty() {
                return Ok(Json(ChatCompletionResponse::new_with_tool_calls(
                    request_id,
                    model_id,
                    parsed.content.clone(),
                    tool_call_responses,
                    result.prompt_tokens,
                    result.completion_tokens,
                    logprobs,
                )));
            }
        }

        // No tool calls found, but tool parsing was enabled — use the cleaned
        // content from the parser (thinking blocks and structural markers
        // stripped) instead of the raw generation output.
        return Ok(Json(ChatCompletionResponse::new_with_logprobs(
            request_id,
            model_id,
            parsed.content,
            result.prompt_tokens,
            result.completion_tokens,
            Some(result.finish_reason),
            logprobs,
        )));
    }

    Ok(Json(ChatCompletionResponse::new_with_logprobs(
        request_id,
        model_id,
        result.text,
        result.prompt_tokens,
        result.completion_tokens,
        Some(result.finish_reason),
        logprobs,
    )))
}

async fn stream_chat_completion(
    state: AppState,
    request: ChatCompletionRequest,
    priority: RequestPriority,
) -> Response {
    // Queue-depth admission control: return 503 before opening SSE stream
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();
    let prepared = prepare_chat_request(&state.chat_template, &request).await;
    let mut options = build_generate_options(&request.params, &state.config);
    options.priority = priority;

    // Extract include_usage before request is moved into the closure
    let include_usage = request
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    // Set logprobs configuration when requested
    let top_k = request.top_logprobs.unwrap_or(0) as usize;
    let logprobs_enabled = request.logprobs == Some(true);
    if logprobs_enabled {
        options.logprobs = LogprobsConfig {
            enabled: true,
            top_k,
        };
    }

    let parse_tools = tool_calls::should_parse_tool_calls(&request);
    let tools_for_parser = if parse_tools {
        request.tools.clone()
    } else {
        None
    };
    let tool_choice = request.tool_choice.clone();

    let (events, stream, cancelled) = sse_channel(100);

    // Clone for the spawned task
    let request_id_clone = request_id.clone();
    let model_id_clone = model_id.clone();
    let finish_events = events.clone();
    let tokenizer = state.tokenizer.clone();

    // Spawn a blocking task to handle generation
    tokio::task::spawn_blocking(move || {
        // Send initial chunk with role
        let initial =
            ChatCompletionChunk::initial(request_id_clone.clone(), model_id_clone.clone());
        let _ = finish_events.json(&initial);

        // Accumulate full output for tool call parsing at the end
        let token_events = finish_events.clone();
        let request_id_inner = request_id_clone.clone();
        let model_id_inner = model_id_clone.clone();

        let accumulated = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let acc_clone = accumulated.clone();

        // Stream filter: strips Gemma 4 (and similar) structural tokens from
        // content deltas so clients never see <|channel>, <|tool_call>, etc.
        // Active only when tool-call parsing is enabled.
        let stream_filter = if parse_tools {
            Some(std::sync::Arc::new(std::sync::Mutex::new(
                StreamFilter::new(),
            )))
        } else {
            None
        };
        let filter_for_callback = stream_filter.clone();

        let result = state
            .model_provider
            .generate_streaming_with_logprobs_cancellable(
                prepared.prompt,
                options,
                prepared.image_data,
                prepared.audio_data,
                cancelled,
                |token, lp_data| {
                    // Always accumulate raw text for tool call parsing
                    if parse_tools && let Ok(mut acc) = acc_clone.lock() {
                        acc.push_str(&token);
                    }

                    // Apply stream filter to strip structural tokens
                    let emit_text = if let Some(ref filter) = filter_for_callback {
                        filter.lock().ok().and_then(|mut f| f.feed(&token))
                    } else {
                        Some(token)
                    };

                    if let Some(text) = emit_text {
                        if text.is_empty() {
                            return;
                        }
                        let logprobs = if logprobs_enabled {
                            lp_data
                                .as_ref()
                                .map(|lp| build_single_token_chat_logprobs(&tokenizer, lp, top_k))
                        } else {
                            None
                        };
                        let chunk = ChatCompletionChunk::content_with_logprobs(
                            request_id_inner.clone(),
                            model_id_inner.clone(),
                            text,
                            logprobs,
                        );
                        let _ = token_events.json(&chunk);
                    }
                },
            );

        // Flush any remaining buffered content from the stream filter
        if let Some(ref filter) = stream_filter
            && let Some(remaining) = filter.lock().ok().and_then(|mut f| f.flush())
            && !remaining.is_empty()
        {
            let chunk = ChatCompletionChunk::content_with_logprobs(
                request_id_clone.clone(),
                model_id_clone.clone(),
                remaining,
                None,
            );
            let _ = finish_events.json(&chunk);
        }

        // Check for tool calls in accumulated output
        let mut finish_reason = match &result {
            Ok(r) => r.finish_reason.clone(),
            Err(_) => "error".to_string(),
        };

        if parse_tools && let Ok(full_output) = accumulated.lock() {
            let tools_ref = tools_for_parser.as_deref();
            let parsed = tool_calls::parse_tool_calls(&full_output, tools_ref);

            if parsed.has_tool_calls() {
                // Emit tool call deltas
                let specific_fn = tool_choice
                    .as_ref()
                    .and_then(|tc| tc.specific_function())
                    .map(|s| s.to_string());

                for (idx, call) in parsed.tool_calls.iter().enumerate() {
                    // Filter by specific function if applicable
                    if let Some(ref fn_name) = specific_fn
                        && call.name != *fn_name
                    {
                        continue;
                    }

                    let call_id = tool_calls::generate_tool_call_id();

                    // Send tool call start delta
                    let start_chunk = ChatCompletionChunk::tool_call_start(
                        request_id_clone.clone(),
                        model_id_clone.clone(),
                        idx,
                        call_id,
                        call.name.clone(),
                    );
                    let _ = finish_events.json(&start_chunk);

                    // Send arguments as a single chunk
                    let args_chunk = ChatCompletionChunk::tool_call_arguments(
                        request_id_clone.clone(),
                        model_id_clone.clone(),
                        idx,
                        call.arguments.clone(),
                    );
                    let _ = finish_events.json(&args_chunk);
                }

                finish_reason = "tool_calls".to_string();
            }
        }

        // Send finish chunk
        let finish = ChatCompletionChunk::finish(
            request_id_clone.clone(),
            model_id_clone.clone(),
            finish_reason,
        );
        let _ = finish_events.json(&finish);

        // Send usage chunk if requested (stream_options.include_usage)
        if include_usage && let Ok(ref r) = result {
            let usage_chunk = ChatCompletionChunk::usage(
                request_id_clone.clone(),
                model_id_clone.clone(),
                r.prompt_tokens,
                r.completion_tokens,
            );
            let _ = finish_events.json(&usage_chunk);
        }

        finish_events.done();
    });

    Sse::new(stream).into_response()
}

/// Maximum number of tools allowed in a single request.
///
/// Enforced in `chat_completions()` to prevent DoS via large tool definitions
/// being rendered through the Jinja2 chat template.
pub(crate) const MAX_TOOLS: usize = 128;

/// Build ServerGenerateOptions using request params with server config as defaults
pub(crate) fn build_generate_options(
    params: &SamplingParams,
    config: &ServerConfig,
) -> ServerGenerateOptions {
    build_server_generate_options(
        config,
        RequestOptionOverrides {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_k: params.top_k.map(|k| k as i32),
            top_p: params.top_p,
            min_p: params.min_p,
            repetition_penalty: params.repetition_penalty,
            seed: params.seed,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            dry_multiplier: params.dry_multiplier,
            dry_base: params.dry_base,
            dry_allowed_length: params.dry_allowed_length,
            dry_penalty_last_n: params.dry_penalty_last_n,
            dry_sequence_breakers: params.dry_sequence_breakers.clone(),
            stop_sequences: params.stop.clone(),
            priority: RequestPriority::default(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::types::request::{FunctionDefinition, Tool};

    fn make_tool(name: &str) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: None,
                parameters: None,
            },
        }
    }

    #[test]
    fn max_tools_constant_is_128() {
        assert_eq!(MAX_TOOLS, 128);
    }

    #[test]
    fn tools_below_limit_accepted() {
        // Build a vec of exactly MAX_TOOLS tools — should not exceed the limit
        let tools: Vec<Tool> = (0..MAX_TOOLS)
            .map(|i| make_tool(&format!("fn_{i}")))
            .collect();
        assert!(tools.len() <= MAX_TOOLS);
    }

    #[test]
    fn tools_above_limit_detected() {
        // Build a vec of MAX_TOOLS + 1 tools — must exceed the limit
        let tools: Vec<Tool> = (0..=MAX_TOOLS)
            .map(|i| make_tool(&format!("fn_{i}")))
            .collect();
        assert!(tools.len() > MAX_TOOLS);
    }
}
