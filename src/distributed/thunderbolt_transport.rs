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

//! Thunderbolt transport backend (stubbed).
//!
//! This module provides the API surface for high-bandwidth local
//! communication over Thunderbolt / IP-over-Thunderbolt Bridge on macOS.
//!
//! The Thunderbolt backend is intended for Mac Studio clusters connected via
//! Thunderbolt cables, where shared-memory or DMA-based zero-copy transfers
//! can dramatically reduce latency for large tensor payloads.
//!
//! # Current Status
//!
//! All trait methods return `Err(ThunderboltUnavailable)` at runtime.
//! The API is fully specified so that a future hardware-specific
//! implementation can be dropped in without changing callers.
//!
//! # Future Implementation Notes
//!
//! - macOS exposes Thunderbolt bridges as standard network interfaces
//!   (e.g., `bridge0`). IP-over-Thunderbolt typically provides ~20 Gbps.
//! - For true zero-copy, a future implementation could use IOKit to set up
//!   shared memory regions between two Thunderbolt-connected machines.
//! - The Thunderbolt discovery mechanism may rely on Bonjour / mDNS to
//!   detect peers on the local bridge interface.

use std::pin::Pin;

use anyhow::Result;
use bytes::Bytes;

use super::transport::{BoxedAsyncRead, RpcHandler, Transport, TransportBackend, TransportMessage};

/// Error returned by all Thunderbolt transport operations while the backend
/// is unimplemented.
#[derive(Debug, thiserror::Error)]
#[error("Thunderbolt transport is not available (stub implementation)")]
pub struct ThunderboltUnavailable;

/// Configuration for the Thunderbolt transport backend.
///
/// Fields are defined to match the expected future implementation so that
/// configuration files can be prepared ahead of time.
#[derive(Debug, Clone)]
pub struct ThunderboltTransportConfig {
    /// Thunderbolt bridge interface name (e.g., `"bridge0"`).
    pub interface: String,
    /// Port to bind on the bridge interface.
    pub port: u16,
    /// Whether to attempt shared-memory (zero-copy DMA) transfers.
    pub use_shared_memory: bool,
    /// Maximum in-flight transfer size in bytes (default 1 GiB).
    pub max_transfer_size: usize,
}

impl Default for ThunderboltTransportConfig {
    fn default() -> Self {
        Self {
            interface: "bridge0".to_string(),
            port: 9200,
            use_shared_memory: true,
            max_transfer_size: 1024 * 1024 * 1024,
        }
    }
}

/// Stubbed Thunderbolt transport.
///
/// All operations return [`ThunderboltUnavailable`]. This type exists so
/// that the distributed layer can reference the Thunderbolt backend in
/// configuration and dispatch logic without conditional compilation.
pub struct ThunderboltTransport {
    config: ThunderboltTransportConfig,
}

impl ThunderboltTransport {
    /// Create a new (non-functional) Thunderbolt transport.
    pub fn new(config: ThunderboltTransportConfig) -> Self {
        tracing::warn!(
            "Thunderbolt transport created in stub mode (interface={}, port={}). \
             All operations will return ThunderboltUnavailable.",
            config.interface,
            config.port,
        );
        Self { config }
    }

    /// Return the configured interface name.
    pub fn interface(&self) -> &str {
        &self.config.interface
    }

    /// Return the configured port.
    pub fn port(&self) -> u16 {
        self.config.port
    }

    /// Check whether shared-memory mode is configured.
    pub fn use_shared_memory(&self) -> bool {
        self.config.use_shared_memory
    }

    /// Detect whether a Thunderbolt bridge interface is available on this
    /// machine.
    ///
    /// Currently always returns `false` in the stub implementation.
    pub fn is_available() -> bool {
        false
    }
}

impl Transport for ThunderboltTransport {
    fn connect(
        &self,
        _peers: &[String],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async { Err(ThunderboltUnavailable.into()) })
    }

    fn send(
        &self,
        _peer: &str,
        _message: TransportMessage,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async { Err(ThunderboltUnavailable.into()) })
    }

    fn recv(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(String, TransportMessage)>> + Send + '_>>
    {
        Box::pin(async { Err(ThunderboltUnavailable.into()) })
    }

    fn send_stream(
        &self,
        _peer: &str,
        _data: Bytes,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async { Err(ThunderboltUnavailable.into()) })
    }

    fn recv_stream(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(String, BoxedAsyncRead)>> + Send + '_>>
    {
        Box::pin(async { Err(ThunderboltUnavailable.into()) })
    }

    fn rpc_call(
        &self,
        _peer: &str,
        _request: &[u8],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>>> + Send + '_>> {
        Box::pin(async { Err(ThunderboltUnavailable.into()) })
    }

    fn serve_rpc(
        &self,
        _handler: RpcHandler,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async { Err(ThunderboltUnavailable.into()) })
    }

    fn shutdown(&self) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }

    fn backend(&self) -> TransportBackend {
        TransportBackend::Thunderbolt
    }

    fn local_addr(&self) -> Result<String> {
        Ok(format!("{}:{}", self.config.interface, self.config.port))
    }
}

#[cfg(test)]
#[path = "thunderbolt_transport_tests.rs"]
mod tests;
