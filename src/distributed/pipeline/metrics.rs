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

//! Pipeline parallelism metrics collection and reporting.
//!
//! Tracks bubble ratio, per-stage utilization, and latency breakdown for
//! the pipeline schedule. Metrics are collected per-step and can be
//! aggregated for reporting.
//!
//! Also exposes counters/histograms for elastic repartition events (issue
//! #349). The sink API is intentionally transport-agnostic — the Prometheus
//! endpoint (#350) reads from [`RepartitionMetricsSnapshot`] while unit tests
//! inspect the raw [`RepartitionMetrics`] struct.
//!
//! Used by: pipeline schedule, pipeline execution loop, server metrics endpoint

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::elastic::{RepartitionEvent, RepartitionEventSink, RepartitionOutcome};

/// Per-stage timing breakdown for a single pipeline step.
#[derive(Debug, Clone)]
pub struct StageMetrics {
    /// Index of this stage (0-based).
    pub stage_index: u32,
    /// Time spent in forward computation (model forward pass).
    pub compute_time: Duration,
    /// Time spent waiting for input activations from the previous stage.
    pub wait_time: Duration,
    /// Time spent transferring activations to the next stage.
    pub transfer_time: Duration,
    /// Number of micro-batches processed in this step.
    pub micro_batches_processed: u32,
}

impl StageMetrics {
    /// Create empty metrics for a stage.
    pub fn new(stage_index: u32) -> Self {
        Self {
            stage_index,
            compute_time: Duration::ZERO,
            wait_time: Duration::ZERO,
            transfer_time: Duration::ZERO,
            micro_batches_processed: 0,
        }
    }

    /// Total time this stage was active (compute + wait + transfer).
    pub fn total_time(&self) -> Duration {
        self.compute_time + self.wait_time + self.transfer_time
    }

    /// Stage utilization: fraction of total time spent computing.
    /// Returns 0.0 if total time is zero.
    pub fn utilization(&self) -> f64 {
        let total = self.total_time();
        if total.is_zero() {
            return 0.0;
        }
        self.compute_time.as_secs_f64() / total.as_secs_f64()
    }
}

impl fmt::Display for StageMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Stage {} | compute={:.2}ms wait={:.2}ms transfer={:.2}ms util={:.1}% mbs={}",
            self.stage_index,
            self.compute_time.as_secs_f64() * 1000.0,
            self.wait_time.as_secs_f64() * 1000.0,
            self.transfer_time.as_secs_f64() * 1000.0,
            self.utilization() * 100.0,
            self.micro_batches_processed,
        )
    }
}

/// Aggregated pipeline metrics for a single step or across steps.
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    /// Per-stage metrics, keyed by stage index.
    pub stages: HashMap<u32, StageMetrics>,
    /// Total wall-clock time for this pipeline step.
    pub wall_time: Duration,
    /// Number of micro-batches in this step.
    pub num_micro_batches: u32,
    /// Number of pipeline stages.
    pub num_stages: u32,
    /// Number of completed (flushed) sequences in this step.
    pub sequences_completed: u32,
}

impl PipelineMetrics {
    /// Create empty pipeline metrics.
    pub fn new(num_stages: u32, num_micro_batches: u32) -> Self {
        let stages = (0..num_stages).map(|i| (i, StageMetrics::new(i))).collect();
        Self {
            stages,
            wall_time: Duration::ZERO,
            num_micro_batches,
            num_stages,
            sequences_completed: 0,
        }
    }

    /// Compute the pipeline bubble ratio.
    ///
    /// Bubble ratio = 1 - (sum of compute times) / (num_stages * wall_time).
    /// A perfect pipeline has bubble ratio 0; a fully serial pipeline has
    /// bubble ratio (N-1)/N.
    ///
    /// Returns 0.0 if wall_time is zero.
    pub fn bubble_ratio(&self) -> f64 {
        if self.wall_time.is_zero() || self.num_stages == 0 {
            return 0.0;
        }
        let total_compute: f64 = self
            .stages
            .values()
            .map(|s| s.compute_time.as_secs_f64())
            .sum();
        let ideal_total = self.num_stages as f64 * self.wall_time.as_secs_f64();
        1.0 - (total_compute / ideal_total)
    }

    /// Average stage utilization across all stages.
    pub fn average_utilization(&self) -> f64 {
        if self.stages.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.stages.values().map(|s| s.utilization()).sum();
        sum / self.stages.len() as f64
    }

    /// Theoretical optimal bubble ratio for GPipe schedule.
    ///
    /// For GPipe with S stages and M micro-batches:
    /// bubble_fraction = (S - 1) / (S - 1 + M)
    pub fn theoretical_bubble_ratio(&self) -> f64 {
        if self.num_stages <= 1 || self.num_micro_batches == 0 {
            return 0.0;
        }
        let s = self.num_stages as f64;
        let m = self.num_micro_batches as f64;
        (s - 1.0) / (s - 1.0 + m)
    }

    /// Record stage compute time.
    pub fn record_compute(&mut self, stage_index: u32, duration: Duration) {
        if let Some(stage) = self.stages.get_mut(&stage_index) {
            stage.compute_time += duration;
        }
    }

    /// Record stage wait time.
    pub fn record_wait(&mut self, stage_index: u32, duration: Duration) {
        if let Some(stage) = self.stages.get_mut(&stage_index) {
            stage.wait_time += duration;
        }
    }

    /// Record stage transfer time.
    pub fn record_transfer(&mut self, stage_index: u32, duration: Duration) {
        if let Some(stage) = self.stages.get_mut(&stage_index) {
            stage.transfer_time += duration;
        }
    }

    /// Increment the micro-batch count for a stage.
    pub fn record_micro_batch_processed(&mut self, stage_index: u32) {
        if let Some(stage) = self.stages.get_mut(&stage_index) {
            stage.micro_batches_processed += 1;
        }
    }
}

impl fmt::Display for PipelineMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Pipeline | stages={} micro_batches={} wall={:.2}ms bubble={:.1}% avg_util={:.1}%",
            self.num_stages,
            self.num_micro_batches,
            self.wall_time.as_secs_f64() * 1000.0,
            self.bubble_ratio() * 100.0,
            self.average_utilization() * 100.0,
        )?;
        let mut stage_indices: Vec<u32> = self.stages.keys().copied().collect();
        stage_indices.sort();
        for idx in stage_indices {
            if let Some(stage) = self.stages.get(&idx) {
                writeln!(f, "  {stage}")?;
            }
        }
        Ok(())
    }
}

/// Accumulates pipeline metrics across multiple steps for aggregate reporting.
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    /// Number of pipeline stages (fixed for the lifetime of the pipeline).
    num_stages: u32,
    /// All recorded step metrics.
    step_metrics: Vec<PipelineMetrics>,
    /// Running wall-clock start time for the current step.
    step_start: Option<Instant>,
    /// Total sequences completed across all steps.
    total_sequences_completed: u64,
    /// Total micro-batches processed across all steps.
    total_micro_batches: u64,
}

impl MetricsCollector {
    /// Create a new collector for a pipeline with `num_stages` stages.
    pub fn new(num_stages: u32) -> Self {
        Self {
            num_stages,
            step_metrics: Vec::new(),
            step_start: None,
            total_sequences_completed: 0,
            total_micro_batches: 0,
        }
    }

    /// Mark the start of a new pipeline step.
    pub fn begin_step(&mut self) {
        self.step_start = Some(Instant::now());
    }

    /// Finalize the current step with the given metrics.
    pub fn end_step(&mut self, mut metrics: PipelineMetrics) {
        if let Some(start) = self.step_start.take() {
            metrics.wall_time = start.elapsed();
        }
        self.total_sequences_completed += metrics.sequences_completed as u64;
        self.total_micro_batches += metrics.num_micro_batches as u64;
        self.step_metrics.push(metrics);
    }

    /// Number of steps recorded so far.
    pub fn num_steps(&self) -> usize {
        self.step_metrics.len()
    }

    /// Total sequences completed across all steps.
    pub fn total_sequences_completed(&self) -> u64 {
        self.total_sequences_completed
    }

    /// Compute the average bubble ratio across all recorded steps.
    pub fn average_bubble_ratio(&self) -> f64 {
        if self.step_metrics.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.step_metrics.iter().map(|m| m.bubble_ratio()).sum();
        sum / self.step_metrics.len() as f64
    }

    /// Compute the average stage utilization across all steps.
    pub fn average_utilization(&self) -> f64 {
        if self.step_metrics.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .step_metrics
            .iter()
            .map(|m| m.average_utilization())
            .sum();
        sum / self.step_metrics.len() as f64
    }

    /// Get a snapshot summary of collected metrics.
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            num_stages: self.num_stages,
            num_steps: self.step_metrics.len() as u64,
            total_sequences_completed: self.total_sequences_completed,
            total_micro_batches: self.total_micro_batches,
            average_bubble_ratio: self.average_bubble_ratio(),
            average_utilization: self.average_utilization(),
        }
    }
}

/// Summary snapshot of pipeline metrics.
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub num_stages: u32,
    pub num_steps: u64,
    pub total_sequences_completed: u64,
    pub total_micro_batches: u64,
    pub average_bubble_ratio: f64,
    pub average_utilization: f64,
}

impl fmt::Display for MetricsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pipeline Summary | stages={} steps={} seqs={} mbs={} bubble={:.1}% util={:.1}%",
            self.num_stages,
            self.num_steps,
            self.total_sequences_completed,
            self.total_micro_batches,
            self.average_bubble_ratio * 100.0,
            self.average_utilization * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Elastic repartition counters (consumed by #350's /metrics endpoint)
// ---------------------------------------------------------------------------

/// Atomic counters + histogram totals for elastic repartition events.
///
/// This is the low-level container the coordinator writes to; the
/// Prometheus endpoint reads a [`RepartitionMetricsSnapshot`] to render
/// stable text output. Used by: #349 elastic coordinator (writer),
/// #350 Prometheus endpoint (reader).
#[derive(Debug, Default)]
pub struct RepartitionMetrics {
    /// Total repartition events, partitioned by trigger kind + outcome.
    ///
    /// Key format: `(trigger_kind, outcome_label)`.
    ///
    /// `trigger_kind` is one of: "explicit", "memory_pressure".
    /// `outcome_label` is one of: "completed", "aborted", "failed", "progress"
    /// (the latter is emitted for intermediate state transitions).
    counters: Mutex<HashMap<(&'static str, &'static str), u64>>,
    /// Sum of drain durations observed on `completed` outcomes, in
    /// microseconds.
    drain_us_total: AtomicU64,
    /// Number of completed drains (the histogram count).
    drain_count: AtomicU64,
    /// Sum of total durations observed across *all* terminal events, in
    /// microseconds. Useful for computing average repartition wall time.
    total_us_total: AtomicU64,
    /// Number of terminal events observed across all outcomes.
    total_count: AtomicU64,
    /// Maximum drain duration observed, microseconds. Simple O(1) proxy for
    /// a p99 until the full histogram lands.
    drain_us_max: AtomicU64,
}

impl RepartitionMetrics {
    /// Create a fresh metrics container.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot for the Prometheus endpoint.
    pub fn snapshot(&self) -> RepartitionMetricsSnapshot {
        let counters: HashMap<String, u64> = self
            .counters
            .lock()
            .expect("repartition metrics poisoned")
            .iter()
            .map(|((k, v), count)| (format!("{k}:{v}"), *count))
            .collect();
        RepartitionMetricsSnapshot {
            counters,
            drain_us_total: self.drain_us_total.load(Ordering::Relaxed),
            drain_count: self.drain_count.load(Ordering::Relaxed),
            total_us_total: self.total_us_total.load(Ordering::Relaxed),
            total_count: self.total_count.load(Ordering::Relaxed),
            drain_us_max: self.drain_us_max.load(Ordering::Relaxed),
        }
    }

    fn bump(&self, key: (&'static str, &'static str)) {
        let mut map = self.counters.lock().expect("repartition metrics poisoned");
        *map.entry(key).or_insert(0) += 1;
    }

    fn record_drain(&self, drain_us: u64) {
        self.drain_us_total.fetch_add(drain_us, Ordering::Relaxed);
        self.drain_count.fetch_add(1, Ordering::Relaxed);
        let mut current = self.drain_us_max.load(Ordering::Relaxed);
        while drain_us > current {
            match self.drain_us_max.compare_exchange_weak(
                current,
                drain_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }
}

impl RepartitionEventSink for RepartitionMetrics {
    fn record_event(&self, event: &RepartitionEvent) {
        let trigger_kind = event.trigger.kind_label();
        let outcome_label = match event.outcome {
            Some(RepartitionOutcome::Completed) => "completed",
            Some(RepartitionOutcome::Aborted) => "aborted",
            Some(RepartitionOutcome::Failed) => "failed",
            None => "progress",
        };
        self.bump((trigger_kind, outcome_label));
        if event.outcome.is_some() {
            self.total_us_total.fetch_add(
                event.total_duration.as_micros().min(u128::from(u64::MAX)) as u64,
                Ordering::Relaxed,
            );
            self.total_count.fetch_add(1, Ordering::Relaxed);
        }
        if matches!(event.outcome, Some(RepartitionOutcome::Completed)) {
            self.record_drain(event.drain_duration.as_micros().min(u128::from(u64::MAX)) as u64);
        }
    }
}

/// Snapshot returned by [`RepartitionMetrics::snapshot`] for rendering.
#[derive(Debug, Clone, Default)]
pub struct RepartitionMetricsSnapshot {
    /// Counter values keyed by "trigger_kind:outcome".
    pub counters: HashMap<String, u64>,
    /// Sum of drain durations across completed events, microseconds.
    pub drain_us_total: u64,
    /// Number of completed drain observations.
    pub drain_count: u64,
    /// Sum of total repartition durations across terminal events, microseconds.
    pub total_us_total: u64,
    /// Number of terminal events observed.
    pub total_count: u64,
    /// Largest drain duration observed so far, microseconds.
    pub drain_us_max: u64,
}

impl RepartitionMetricsSnapshot {
    /// Total events counted, regardless of outcome.
    pub fn total_events(&self) -> u64 {
        self.counters.values().copied().sum()
    }

    /// Mean drain duration (microseconds) across completed events, or zero
    /// when no completed drain has been observed yet.
    pub fn mean_drain_us(&self) -> u64 {
        if self.drain_count == 0 {
            0
        } else {
            self.drain_us_total / self.drain_count
        }
    }
}

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod tests;
