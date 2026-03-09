//! Shared execution-plane helpers used by both the CLI and the HTTP server.
//!
//! Keeping runtime/device resolution and sampling assembly together makes new
//! entry points easier to add without re-implementing environment parsing or
//! generation defaults in multiple places.

pub mod runtime;
pub mod sampling;
