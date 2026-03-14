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

//! Sequence lifecycle state machine and per-request context.
//!
//! [`SequenceState`] models the valid lifecycle:
//!
//! ```text
//! Queued --> Prefilling --> Decoding --> Finished(reason)
//!   \            \                         ^
//!    \            \--- Finished(Cancelled/Error)
//!     \----- Finished(Cancelled/Error)
//! ```
//!
//! Any non-terminal state may transition to `Finished(Cancelled)` or
//! `Finished(Error)` so the scheduler can abort sequences when a client
//! disconnects or an error occurs during prefill.
//!
//! [`SequenceInfo`] bundles every piece of context needed by the batch
//! scheduler and generation loop: prompt tokens, sampling config, VLM
//! embeddings, generated output, response channel, and timing data.

use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;

use crate::server::model_provider::GenerateEvent;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::vision::merge::InputEmbeddings;

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

/// Lifecycle state of a single generation sequence.
///
/// Valid transitions:
///
/// ```text
/// Queued --> Prefilling --> Decoding --> Finished(reason)
/// ```
///
/// Additionally, any non-terminal state may transition to
/// `Finished(Cancelled)` or `Finished(Error)` for early abort (e.g. client
/// disconnect, prefill failure).
///
/// The scheduler is the sole authority for driving transitions; HTTP handlers
/// only observe the current state.
#[derive(Debug)]
#[non_exhaustive]
pub enum SequenceState {
    /// Waiting in the prefill queue.
    Queued,
    /// Currently being prefilled (prompt processing).
    Prefilling,
    /// In the active decode batch, generating tokens.
    Decoding,
    /// Generation complete.
    Finished(FinishReason),
}

/// Why a sequence finished generating.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    /// An EOS token was emitted.
    Stop,
    /// `max_tokens` was reached.
    Length,
    /// The client disconnected or cancelled the request.
    Cancelled,
    /// An internal error occurred during generation.
    Error(String),
}

impl SequenceState {
    /// Attempt to transition from the current state to `next`.
    ///
    /// Returns `Ok(())` if the transition is valid, or `Err` with a message
    /// describing the illegal transition.
    ///
    /// Valid transitions:
    /// - `Queued -> Prefilling` (scheduler begins prefill)
    /// - `Prefilling -> Decoding` (prefill complete, enter decode loop)
    /// - `Decoding -> Finished(*)` (normal completion)
    /// - Any non-terminal state -> `Finished(Cancelled | Error)` (early abort)
    pub fn transition_to(&mut self, next: SequenceState) -> Result<(), String> {
        let valid = match (&*self, &next) {
            // Normal forward progression
            (SequenceState::Queued, SequenceState::Prefilling) => true,
            (SequenceState::Prefilling, SequenceState::Decoding) => true,
            (SequenceState::Decoding, SequenceState::Finished(_)) => true,
            // Early abort: any non-terminal state can transition to
            // Finished(Cancelled) or Finished(Error) so the scheduler can
            // clean up sequences when a client disconnects or an error occurs
            // during prefill.
            (_, SequenceState::Finished(FinishReason::Cancelled | FinishReason::Error(_)))
                if !self.is_finished() =>
            {
                true
            }
            _ => false,
        };

        if valid {
            *self = next;
            Ok(())
        } else {
            Err(format!("invalid state transition: {self:?} -> {next:?}"))
        }
    }

    /// Returns `true` if the sequence has reached a terminal state.
    pub fn is_finished(&self) -> bool {
        matches!(self, SequenceState::Finished(_))
    }
}

// ---------------------------------------------------------------------------
// Per-request context
// ---------------------------------------------------------------------------

/// Full context for a single generation request.
///
/// Created when the HTTP handler enqueues a request and consumed by the batch
/// scheduler through prefill and decode phases. The `response_tx` channel
/// carries streaming `GenerateEvent` values back to the HTTP handler.
pub struct SequenceInfo {
    /// Unique identifier assigned by the `CachePool`.
    pub seq_id: SequenceId,
    /// Current lifecycle state.
    pub state: SequenceState,

    // -- Request context --
    /// Tokenized prompt (integer token IDs).
    pub prompt_tokens: Vec<i32>,
    /// Sampling parameters for this request.
    pub sampling: SamplingConfig,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// EOS token IDs that terminate generation.
    pub eos_token_ids: Vec<i32>,

    // -- VLM context (optional) --
    /// Pre-computed vision-language embeddings for VLM requests.
    pub vlm_embeddings: Option<InputEmbeddings>,
    /// Raw image bytes for VLM requests (empty for text-only).
    pub images: Vec<Vec<u8>>,

    // -- Generation state --
    /// Tokens produced so far during decode.
    pub generated_tokens: Vec<i32>,
    /// Accumulated decoded text.
    pub generated_text: String,
    /// Streaming decode helper for incremental text emission.
    // Field is not read yet -- the batch scheduler will use it when the
    // decode loop is wired up. Allow dead_code until then.
    #[allow(dead_code)]
    pub(crate) decode_state: StreamingDecodeState,

    // -- Response channel --
    /// Sender for streaming events back to the HTTP handler.
    pub response_tx: mpsc::Sender<GenerateEvent>,

    // -- Timing --
    /// Wall-clock time when the request was received.
    pub created_at: Instant,
    /// Wall-clock time when prefill started (set on `Queued -> Prefilling`).
    pub prefill_start: Option<Instant>,
    /// Wall-clock time when the first decode token was produced.
    pub first_token_time: Option<Instant>,
}

impl SequenceInfo {
    /// Convenience: returns the number of generated tokens so far.
    pub fn completion_count(&self) -> usize {
        self.generated_tokens.len()
    }

    /// Returns `true` if this request carries VLM image data.
    pub fn is_vlm_request(&self) -> bool {
        self.vlm_embeddings.is_some() || !self.images.is_empty()
    }
}

// We cannot derive `Debug` automatically because `InputEmbeddings` contains
// `UniquePtr<MlxArray>` which is not `Debug`. A manual implementation keeps
// the struct debuggable in logs.
impl std::fmt::Debug for SequenceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequenceInfo")
            .field("seq_id", &self.seq_id)
            .field("state", &self.state)
            .field("prompt_tokens_len", &self.prompt_tokens.len())
            .field("max_tokens", &self.max_tokens)
            .field("generated_tokens_len", &self.generated_tokens.len())
            .field("is_vlm", &self.is_vlm_request())
            .field("created_at", &self.created_at)
            .field("prefill_start", &self.prefill_start)
            .field("first_token_time", &self.first_token_time)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Scheduler action
// ---------------------------------------------------------------------------

/// Decision emitted by the batch scheduler each tick.
///
/// The scheduler inspects the prefill queue and active batch, then returns one
/// of these actions for the execution engine to carry out.
#[derive(Debug)]
#[non_exhaustive]
pub enum BatchSchedulerAction {
    /// Prefill the next queued request.
    Prefill(SequenceId),
    /// Run one decode step for the listed active sequences.
    Decode(Vec<SequenceId>),
    /// No work available -- the engine should block until a new request
    /// arrives.
    Idle,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SequenceState transition tests --

    #[test]
    fn valid_transition_queued_to_prefilling() {
        let mut state = SequenceState::Queued;
        assert!(state.transition_to(SequenceState::Prefilling).is_ok());
        assert!(matches!(state, SequenceState::Prefilling));
    }

    #[test]
    fn valid_transition_prefilling_to_decoding() {
        let mut state = SequenceState::Prefilling;
        assert!(state.transition_to(SequenceState::Decoding).is_ok());
        assert!(matches!(state, SequenceState::Decoding));
    }

    #[test]
    fn valid_transition_decoding_to_finished_stop() {
        let mut state = SequenceState::Decoding;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Stop))
                .is_ok()
        );
        assert!(matches!(state, SequenceState::Finished(FinishReason::Stop)));
    }

    #[test]
    fn valid_transition_decoding_to_finished_length() {
        let mut state = SequenceState::Decoding;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Length))
                .is_ok()
        );
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Length)
        ));
    }

    #[test]
    fn valid_transition_decoding_to_finished_cancelled() {
        let mut state = SequenceState::Decoding;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Cancelled))
                .is_ok()
        );
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Cancelled)
        ));
    }

    #[test]
    fn valid_transition_decoding_to_finished_error() {
        let mut state = SequenceState::Decoding;
        let reason = FinishReason::Error("out of memory".to_string());
        assert!(state.transition_to(SequenceState::Finished(reason)).is_ok());
        match &state {
            SequenceState::Finished(FinishReason::Error(msg)) => {
                assert_eq!(msg, "out of memory");
            }
            other => panic!("expected Finished(Error), got {other:?}"),
        }
    }

    // -- Early cancellation from non-Decoding states --

    #[test]
    fn valid_transition_queued_to_finished_cancelled() {
        let mut state = SequenceState::Queued;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Cancelled))
                .is_ok()
        );
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Cancelled)
        ));
    }

    #[test]
    fn valid_transition_queued_to_finished_error() {
        let mut state = SequenceState::Queued;
        let reason = FinishReason::Error("validation failed".to_string());
        assert!(state.transition_to(SequenceState::Finished(reason)).is_ok());
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Error(_))
        ));
    }

    #[test]
    fn valid_transition_prefilling_to_finished_cancelled() {
        let mut state = SequenceState::Prefilling;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Cancelled))
                .is_ok()
        );
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Cancelled)
        ));
    }

    #[test]
    fn valid_transition_prefilling_to_finished_error() {
        let mut state = SequenceState::Prefilling;
        let reason = FinishReason::Error("OOM during prefill".to_string());
        assert!(state.transition_to(SequenceState::Finished(reason)).is_ok());
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Error(_))
        ));
    }

    // -- Still-invalid transitions --

    #[test]
    fn invalid_transition_queued_to_decoding() {
        let mut state = SequenceState::Queued;
        let result = state.transition_to(SequenceState::Decoding);
        assert!(result.is_err());
        // State should remain unchanged after a failed transition.
        assert!(matches!(state, SequenceState::Queued));
    }

    #[test]
    fn invalid_transition_queued_to_finished_stop() {
        // Queued can only go to Finished via Cancelled or Error, not Stop/Length
        let mut state = SequenceState::Queued;
        let result = state.transition_to(SequenceState::Finished(FinishReason::Stop));
        assert!(result.is_err());
        assert!(matches!(state, SequenceState::Queued));
    }

    #[test]
    fn invalid_transition_queued_to_finished_length() {
        let mut state = SequenceState::Queued;
        let result = state.transition_to(SequenceState::Finished(FinishReason::Length));
        assert!(result.is_err());
        assert!(matches!(state, SequenceState::Queued));
    }

    #[test]
    fn invalid_transition_prefilling_to_queued() {
        let mut state = SequenceState::Prefilling;
        let result = state.transition_to(SequenceState::Queued);
        assert!(result.is_err());
        assert!(matches!(state, SequenceState::Prefilling));
    }

    #[test]
    fn invalid_transition_prefilling_to_finished_stop() {
        // Prefilling can only go to Finished via Cancelled or Error, not Stop/Length
        let mut state = SequenceState::Prefilling;
        let result = state.transition_to(SequenceState::Finished(FinishReason::Stop));
        assert!(result.is_err());
        assert!(matches!(state, SequenceState::Prefilling));
    }

    #[test]
    fn invalid_transition_decoding_to_prefilling() {
        let mut state = SequenceState::Decoding;
        let result = state.transition_to(SequenceState::Prefilling);
        assert!(result.is_err());
        assert!(matches!(state, SequenceState::Decoding));
    }

    #[test]
    fn invalid_transition_finished_to_anything() {
        let mut state = SequenceState::Finished(FinishReason::Stop);
        assert!(state.transition_to(SequenceState::Queued).is_err());
        assert!(state.transition_to(SequenceState::Prefilling).is_err());
        assert!(state.transition_to(SequenceState::Decoding).is_err());
        // Also cannot re-finish (e.g. Finished -> Finished(Cancelled))
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Cancelled))
                .is_err()
        );
    }

    #[test]
    fn is_finished_returns_correct_values() {
        assert!(!SequenceState::Queued.is_finished());
        assert!(!SequenceState::Prefilling.is_finished());
        assert!(!SequenceState::Decoding.is_finished());
        assert!(SequenceState::Finished(FinishReason::Stop).is_finished());
        assert!(SequenceState::Finished(FinishReason::Length).is_finished());
        assert!(SequenceState::Finished(FinishReason::Cancelled).is_finished());
        assert!(SequenceState::Finished(FinishReason::Error("err".into())).is_finished());
    }

    // -- Full lifecycle test --

    #[test]
    fn full_lifecycle_queued_through_finished() {
        let mut state = SequenceState::Queued;

        state.transition_to(SequenceState::Prefilling).unwrap();
        assert!(matches!(state, SequenceState::Prefilling));

        state.transition_to(SequenceState::Decoding).unwrap();
        assert!(matches!(state, SequenceState::Decoding));

        state
            .transition_to(SequenceState::Finished(FinishReason::Stop))
            .unwrap();
        assert!(state.is_finished());
    }

    // -- FinishReason equality --

    #[test]
    fn finish_reason_equality() {
        assert_eq!(FinishReason::Stop, FinishReason::Stop);
        assert_eq!(FinishReason::Length, FinishReason::Length);
        assert_eq!(FinishReason::Cancelled, FinishReason::Cancelled);
        assert_eq!(
            FinishReason::Error("x".into()),
            FinishReason::Error("x".into())
        );
        assert_ne!(FinishReason::Stop, FinishReason::Length);
        assert_ne!(
            FinishReason::Error("a".into()),
            FinishReason::Error("b".into())
        );
    }

    // -- BatchSchedulerAction --

    #[test]
    fn batch_scheduler_action_debug() {
        // Ensure Debug is implemented (compile-time check via formatting).
        let action = BatchSchedulerAction::Idle;
        let _ = format!("{action:?}");
    }
}
