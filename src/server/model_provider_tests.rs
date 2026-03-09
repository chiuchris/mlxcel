use std::sync::mpsc;

use super::{GenerateEvent, GenerationResult, drain_generation_events};

fn sample_result() -> GenerationResult {
    GenerationResult {
        text: "hello".to_string(),
        prompt_tokens: 3,
        completion_tokens: 2,
        generation_time_ms: 10,
        prompt_eval_ms: 4,
        generation_only_ms: 6,
        finish_reason: "stop".to_string(),
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
