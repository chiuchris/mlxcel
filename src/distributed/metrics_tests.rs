use std::collections::VecDeque;
use std::time::Duration;

use super::*;

#[test]
fn collector_record_latency() {
    let collector = MetricsCollector::new(MetricsConfig {
        latency_window_size: 5,
        throughput_window_size: 10,
    });

    collector.record_latency(Duration::from_millis(10));
    collector.record_latency(Duration::from_millis(20));
    collector.record_latency(Duration::from_millis(30));

    let snap = collector.snapshot();
    assert_eq!(snap.latency.min, Duration::from_millis(10));
    assert_eq!(snap.latency.max, Duration::from_millis(30));
}

#[test]
fn collector_latency_window_eviction() {
    let collector = MetricsCollector::new(MetricsConfig {
        latency_window_size: 3,
        throughput_window_size: 10,
    });

    // Fill window
    collector.record_latency(Duration::from_millis(100));
    collector.record_latency(Duration::from_millis(200));
    collector.record_latency(Duration::from_millis(300));
    // This should evict the 100ms sample
    collector.record_latency(Duration::from_millis(50));

    let snap = collector.snapshot();
    assert_eq!(snap.latency.min, Duration::from_millis(50));
    // 100ms was evicted, so max is still 300ms
    assert_eq!(snap.latency.max, Duration::from_millis(300));
}

#[test]
fn collector_request_tracking() {
    let collector = MetricsCollector::new(MetricsConfig::default());

    collector.request_started();
    collector.request_started();
    let snap = collector.snapshot();
    assert_eq!(snap.active_requests, 2);
    assert_eq!(snap.total_requests, 2);

    collector.request_completed();
    let snap = collector.snapshot();
    assert_eq!(snap.active_requests, 1);
    assert_eq!(snap.total_requests, 2);
}

#[test]
fn collector_request_completed_saturates_at_zero() {
    let collector = MetricsCollector::new(MetricsConfig::default());
    // Completing without starting should not underflow
    collector.request_completed();
    let snap = collector.snapshot();
    assert_eq!(snap.active_requests, 0);
}

#[test]
fn collector_network_bytes() {
    let collector = MetricsCollector::new(MetricsConfig::default());

    collector.record_bytes_sent(1024);
    collector.record_bytes_sent(2048);
    collector.record_bytes_recv(512);

    let snap = collector.snapshot();
    assert_eq!(snap.network_bytes_sent, 3072);
    assert_eq!(snap.network_bytes_recv, 512);
}

#[test]
fn collector_memory_update() {
    let collector = MetricsCollector::new(MetricsConfig::default());

    collector.update_memory(8_000_000_000, 16_000_000_000);
    let snap = collector.snapshot();
    assert_eq!(snap.memory_used_bytes, 8_000_000_000);
    assert_eq!(snap.memory_total_bytes, 16_000_000_000);
}

#[test]
fn collector_token_tracking() {
    let collector = MetricsCollector::new(MetricsConfig::default());

    collector.record_tokens(50);
    collector.record_tokens(30);

    let snap = collector.snapshot();
    assert_eq!(snap.total_tokens, 80);
}

#[test]
fn collector_reset() {
    let collector = MetricsCollector::new(MetricsConfig::default());

    collector.record_latency(Duration::from_millis(10));
    collector.record_tokens(100);
    collector.record_bytes_sent(1024);
    collector.request_started();

    collector.reset();
    let snap = collector.snapshot();
    assert_eq!(snap.total_tokens, 0);
    assert_eq!(snap.network_bytes_sent, 0);
    assert_eq!(snap.active_requests, 0);
    assert_eq!(snap.total_requests, 0);
}

#[test]
fn cluster_metrics_crud() {
    let cluster = ClusterMetrics::new();

    let metrics = NodeMetrics {
        throughput_tokens_per_sec: 150.0,
        total_tokens: 1000,
        ..Default::default()
    };

    cluster.update("node-0", metrics.clone());
    assert!(cluster.get("node-0").is_some());
    assert!(cluster.get("nonexistent").is_none());

    let all = cluster.all();
    assert_eq!(all.len(), 1);
    assert!(all.contains_key("node-0"));

    let removed = cluster.remove("node-0");
    assert!(removed.is_some());
    assert!(cluster.get("node-0").is_none());
}

#[test]
fn percentiles_empty_samples() {
    let p = compute_percentiles(&VecDeque::new());
    assert_eq!(p.p50, Duration::ZERO);
    assert_eq!(p.min, Duration::ZERO);
}

#[test]
fn percentiles_single_sample() {
    let samples: VecDeque<Duration> = vec![Duration::from_millis(42)].into();
    let p = compute_percentiles(&samples);
    assert_eq!(p.p50, Duration::from_millis(42));
    assert_eq!(p.min, Duration::from_millis(42));
    assert_eq!(p.max, Duration::from_millis(42));
}

#[test]
fn percentiles_ordered_correctly() {
    let samples: VecDeque<Duration> = (1..=100).map(|i| Duration::from_millis(i)).collect();
    let p = compute_percentiles(&samples);
    assert_eq!(p.min, Duration::from_millis(1));
    assert_eq!(p.max, Duration::from_millis(100));
    // With 100 samples (1..=100ms), index 50 = 51ms, index 95 = 96ms, etc.
    assert_eq!(p.p50, Duration::from_millis(51));
    assert_eq!(p.p95, Duration::from_millis(96));
    assert_eq!(p.p99, Duration::from_millis(100));
}

#[test]
fn throughput_empty() {
    assert_eq!(compute_throughput(&VecDeque::new()), 0.0);
}

#[test]
fn collector_concurrent_access() {
    let collector = MetricsCollector::new(MetricsConfig::default());

    let handles: Vec<_> = (0..10)
        .map(|_| {
            let c = collector.clone();
            std::thread::spawn(move || {
                c.record_latency(Duration::from_millis(10));
                c.record_tokens(5);
                c.record_bytes_sent(100);
                c.request_started();
                c.request_completed();
                let _ = c.snapshot();
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let snap = collector.snapshot();
    assert_eq!(snap.total_tokens, 50);
    assert_eq!(snap.network_bytes_sent, 1000);
    assert_eq!(snap.total_requests, 10);
    assert_eq!(snap.active_requests, 0);
}
