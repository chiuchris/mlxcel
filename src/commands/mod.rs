//! Binary-only command handlers.
//!
//! Keeping subcommand implementations out of `main.rs` leaves the root file as
//! argument/schema wiring while command-specific execution logic evolves in
//! isolated modules.

pub(crate) mod generate;
mod generate_vlm;

pub(crate) use generate::run_generate;
