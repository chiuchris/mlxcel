use std::sync::atomic::Ordering;

use super::Metrics;

#[test]
fn metrics_record_request_accumulates_counters() {
    let metrics = Metrics::new();

    metrics.record_request(10, 3, 120);
    metrics.record_request(4, 2, 30);

    assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 2);
    assert_eq!(metrics.prompt_tokens_total.load(Ordering::Relaxed), 14);
    assert_eq!(metrics.completion_tokens_total.load(Ordering::Relaxed), 5);
    assert_eq!(
        metrics.generation_time_ms_total.load(Ordering::Relaxed),
        150
    );
}
