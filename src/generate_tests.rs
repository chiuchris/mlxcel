use super::generation_stats_from_duration;
use std::time::Duration;

#[test]
fn generation_stats_from_duration_uses_elapsed_time_for_decode_rate() {
    let stats = generation_stats_from_duration(12, 6, Duration::from_secs(2));

    assert_eq!(stats.prompt_tokens, 12);
    assert_eq!(stats.generated_tokens, 6);
    assert_eq!(stats.decode_time_ms, 2000.0);
    assert_eq!(stats.decode_tok_per_sec, 3.0);
}

#[test]
fn generation_stats_from_duration_handles_zero_elapsed_time() {
    let stats = generation_stats_from_duration(4, 2, Duration::ZERO);

    assert_eq!(stats.decode_time_ms, 0.0);
    assert_eq!(stats.decode_tok_per_sec, 0.0);
}
