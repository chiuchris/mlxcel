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

use std::time::Duration;

use super::*;

#[test]
fn stage_utilization_all_compute() {
    let mut s = StageMetrics::new(0);
    s.compute_time = Duration::from_millis(100);
    assert!((s.utilization() - 1.0).abs() < 1e-9);
}

#[test]
fn stage_utilization_half_compute() {
    let mut s = StageMetrics::new(0);
    s.compute_time = Duration::from_millis(50);
    s.wait_time = Duration::from_millis(50);
    assert!((s.utilization() - 0.5).abs() < 1e-9);
}

#[test]
fn stage_utilization_zero_time() {
    let s = StageMetrics::new(0);
    assert!((s.utilization()).abs() < 1e-9);
}

#[test]
fn bubble_ratio_perfect_pipeline() {
    // All stages compute for the full wall time -> bubble = 0.
    let mut m = PipelineMetrics::new(2, 4);
    m.wall_time = Duration::from_millis(100);
    m.stages.get_mut(&0).unwrap().compute_time = Duration::from_millis(100);
    m.stages.get_mut(&1).unwrap().compute_time = Duration::from_millis(100);
    assert!((m.bubble_ratio()).abs() < 1e-9);
}

#[test]
fn bubble_ratio_serial_pipeline() {
    // 2 stages, only one computes at a time -> bubble = 0.5.
    let mut m = PipelineMetrics::new(2, 1);
    m.wall_time = Duration::from_millis(100);
    m.stages.get_mut(&0).unwrap().compute_time = Duration::from_millis(50);
    m.stages.get_mut(&1).unwrap().compute_time = Duration::from_millis(50);
    assert!((m.bubble_ratio() - 0.5).abs() < 1e-9);
}

#[test]
fn bubble_ratio_zero_wall_time() {
    let m = PipelineMetrics::new(2, 4);
    assert!((m.bubble_ratio()).abs() < 1e-9);
}

#[test]
fn theoretical_bubble_ratio_gpipe() {
    // S=4, M=4: theoretical = 3 / (3 + 4) = 3/7 ~ 0.4286
    let m = PipelineMetrics::new(4, 4);
    let expected = 3.0 / 7.0;
    assert!((m.theoretical_bubble_ratio() - expected).abs() < 1e-6);
}

#[test]
fn theoretical_bubble_ratio_many_micro_batches() {
    // S=2, M=8: theoretical = 1 / (1 + 8) = 1/9 ~ 0.111
    let m = PipelineMetrics::new(2, 8);
    let expected = 1.0 / 9.0;
    assert!((m.theoretical_bubble_ratio() - expected).abs() < 1e-6);
}

#[test]
fn theoretical_bubble_ratio_single_stage() {
    let m = PipelineMetrics::new(1, 4);
    assert!((m.theoretical_bubble_ratio()).abs() < 1e-9);
}

#[test]
fn record_operations() {
    let mut m = PipelineMetrics::new(2, 2);
    m.record_compute(0, Duration::from_millis(10));
    m.record_wait(0, Duration::from_millis(5));
    m.record_transfer(0, Duration::from_millis(2));
    m.record_micro_batch_processed(0);

    let s0 = m.stages.get(&0).unwrap();
    assert_eq!(s0.compute_time, Duration::from_millis(10));
    assert_eq!(s0.wait_time, Duration::from_millis(5));
    assert_eq!(s0.transfer_time, Duration::from_millis(2));
    assert_eq!(s0.micro_batches_processed, 1);
}

#[test]
fn metrics_collector_accumulation() {
    let mut collector = MetricsCollector::new(2);

    // Step 1.
    collector.begin_step();
    let mut m1 = PipelineMetrics::new(2, 4);
    m1.sequences_completed = 2;
    collector.end_step(m1);

    // Step 2.
    collector.begin_step();
    let mut m2 = PipelineMetrics::new(2, 4);
    m2.sequences_completed = 3;
    collector.end_step(m2);

    assert_eq!(collector.num_steps(), 2);
    assert_eq!(collector.total_sequences_completed(), 5);

    let summary = collector.summary();
    assert_eq!(summary.num_stages, 2);
    assert_eq!(summary.num_steps, 2);
    assert_eq!(summary.total_sequences_completed, 5);
    assert_eq!(summary.total_micro_batches, 8);
}

#[test]
fn pipeline_metrics_display() {
    let mut m = PipelineMetrics::new(2, 2);
    m.wall_time = Duration::from_millis(50);
    m.record_compute(0, Duration::from_millis(20));
    m.record_compute(1, Duration::from_millis(15));
    let display = format!("{m}");
    assert!(display.contains("Pipeline"));
    assert!(display.contains("Stage 0"));
    assert!(display.contains("Stage 1"));
}

#[test]
fn metrics_summary_display() {
    let collector = MetricsCollector::new(4);
    let summary = collector.summary();
    let display = format!("{summary}");
    assert!(display.contains("stages=4"));
}
