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

use std::sync::mpsc;

use mlxcel_core::sampling::TokenLogprobData;

use super::{
    GenerateEvent, GenerationResult, ModelProvider, ModelRequest, drain_generation_events,
    send_shutdown_signal,
};

fn sample_result() -> GenerationResult {
    GenerationResult {
        text: "hello".to_string(),
        prompt_tokens: 3,
        completion_tokens: 2,
        generation_time_ms: 10,
        prompt_eval_ms: 4,
        generation_only_ms: 6,
        finish_reason: "stop".to_string(),
        logprobs: None,
    }
}

#[test]
fn drain_generation_events_forwards_tokens_before_done() {
    let (tx, rx) = mpsc::channel();
    tx.send(GenerateEvent::Token("A".to_string())).unwrap();
    tx.send(GenerateEvent::Token("B".to_string())).unwrap();
    tx.send(GenerateEvent::Done(sample_result())).unwrap();

    let mut streamed = Vec::new();
    let result = drain_generation_events(rx, |token| streamed.push(token)).unwrap();

    assert_eq!(streamed, vec!["A".to_string(), "B".to_string()]);
    assert_eq!(result.text, "hello");
    assert_eq!(result.finish_reason, "stop");
}

#[test]
fn drain_generation_events_returns_worker_error() {
    let (tx, rx) = mpsc::channel();
    tx.send(GenerateEvent::Error("boom".to_string())).unwrap();

    let err = drain_generation_events(rx, |_| {}).unwrap_err();
    assert!(err.to_string().contains("boom"));
}

#[test]
fn drain_generation_events_reports_closed_channel() {
    let (tx, rx) = mpsc::channel::<GenerateEvent>();
    drop(tx);
    let err = drain_generation_events(rx, |_| {}).unwrap_err();
    assert!(err.to_string().contains("Response channel closed"));
}

#[test]
fn drain_generation_events_accumulates_logprobs_from_token_with_logprobs() {
    let (tx, rx) = mpsc::channel();
    let lp = TokenLogprobData {
        token_id: 42,
        logprob: -0.5,
        top_alternatives: vec![(7, -1.2)],
    };
    tx.send(GenerateEvent::TokenWithLogprobs("Hi".to_string(), lp))
        .unwrap();
    tx.send(GenerateEvent::Done(sample_result())).unwrap();

    let mut streamed = Vec::new();
    let result = drain_generation_events(rx, |token| streamed.push(token)).unwrap();

    assert_eq!(streamed, vec!["Hi".to_string()]);
    let lp_data = result.logprobs.expect("logprobs should be Some");
    assert_eq!(lp_data.len(), 1);
    assert_eq!(lp_data[0].token_id, 42);
    assert!((lp_data[0].logprob - (-0.5)).abs() < 1e-6);
}

#[test]
fn send_shutdown_signal_enqueues_shutdown_request() {
    let (tx, rx) = mpsc::channel();
    assert!(send_shutdown_signal(&tx));
    assert!(matches!(rx.recv().unwrap(), ModelRequest::Shutdown));
}

#[test]
fn send_shutdown_signal_reports_closed_channel() {
    let (tx, rx) = mpsc::channel::<ModelRequest>();
    drop(rx);
    assert!(!send_shutdown_signal(&tx));
}

#[test]
fn model_provider_relies_on_auto_traits_for_shared_state() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ModelProvider>();
}
