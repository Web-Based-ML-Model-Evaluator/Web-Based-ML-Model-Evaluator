//! On-device ML engine.
//!
//! The whole engine runs inside the visitor's browser: it is compiled to WebAssembly,
//! loaded into a Web Worker, and keeps every dataset and result in that tab's memory.
//! Nothing is sent to a server. [`session::Session`] is the API; [`wasm`] exposes it to
//! JavaScript. Native builds exist only so the logic can be tested with `cargo test`.

pub mod core;
pub mod models;
pub mod session;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
