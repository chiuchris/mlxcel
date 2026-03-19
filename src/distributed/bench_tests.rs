use super::*;

#[test]
fn bench_config_defaults() {
    let config = BenchConfig::default();
    assert_eq!(config.payload_size, 1024 * 1024);
    assert_eq!(config.num_messages, 100);
    assert_eq!(config.num_rpc_roundtrips, 100);
    assert_eq!(config.warmup_iterations, 5);
}

#[test]
fn throughput_result_display() {
    let result = ThroughputResult {
        payload_size: 1024,
        num_messages: 10,
        total_bytes: 10240,
        duration: Duration::from_millis(100),
        bytes_per_second: 102400.0,
        mb_per_second: 0.098,
        gbps: 0.00082,
    };
    let display = result.to_string();
    assert!(display.contains("MB/s"));
    assert!(display.contains("Gbps"));
    assert!(display.contains("10 messages"));
}

#[test]
fn latency_result_display() {
    let result = LatencyResult {
        num_roundtrips: 50,
        min: Duration::from_micros(100),
        max: Duration::from_micros(5000),
        mean: Duration::from_micros(500),
        median: Duration::from_micros(450),
        p95: Duration::from_micros(2000),
        p99: Duration::from_micros(4000),
    };
    let display = result.to_string();
    assert!(display.contains("50 roundtrips"));
    assert!(display.contains("min="));
    assert!(display.contains("p95="));
}
