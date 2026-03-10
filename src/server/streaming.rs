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

//! Shared SSE helpers for server routes.
//!
//! Chat, completion, and llama-server-compatible routes all stream over the
//! same blocking channel pattern even though their payload shapes differ.

use std::convert::Infallible;

use axum::response::sse::Event;
use futures::{Stream, StreamExt};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub(crate) const DONE_MARKER: &str = "[DONE]";

type SsePayload = Result<String, Infallible>;

#[derive(Clone)]
pub(crate) struct BlockingSseSender {
    tx: mpsc::Sender<SsePayload>,
}

pub(crate) fn sse_channel(
    buffer: usize,
) -> (
    BlockingSseSender,
    impl Stream<Item = Result<Event, Infallible>>,
) {
    let (sender, rx) = payload_channel(buffer);
    let stream = ReceiverStream::new(rx).map(|payload| payload.map(sse_event));
    (sender, stream)
}

impl BlockingSseSender {
    pub(crate) fn json<T: Serialize>(&self, value: &T) -> Result<(), serde_json::Error> {
        self.text(serialize_json_data(value)?);
        Ok(())
    }

    pub(crate) fn text(&self, data: impl Into<String>) {
        let _ = self.tx.blocking_send(Ok(data.into()));
    }

    pub(crate) fn done(&self) {
        self.text(DONE_MARKER);
    }
}

fn payload_channel(buffer: usize) -> (BlockingSseSender, mpsc::Receiver<SsePayload>) {
    let (tx, rx) = mpsc::channel(buffer);
    (BlockingSseSender { tx }, rx)
}

fn serialize_json_data<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    serde_json::to_string(value)
}

fn sse_event(data: String) -> Event {
    Event::default().data(data)
}

#[cfg(test)]
#[path = "streaming_tests.rs"]
mod tests;
