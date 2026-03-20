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

//! Disaggregated inference: prefill/decode separation.
//!
//! This module provides the scheduling and handoff logic for disaggregated
//! serving, where prefill and decode phases run on separate nodes.
//!
//! - [`prefill_scheduler`] — Prefill-only scheduler that processes prompts
//!   and hands off KV cache + first token to decode nodes.

pub mod prefill_scheduler;

pub use prefill_scheduler::{
    ChunkedPrefillCoordinator, HandoffProtocol, HandoffStatus, PrefillHandoff, PrefillRequest,
    PrefillResult, PrefillScheduler, PrefillSchedulerConfig,
};
