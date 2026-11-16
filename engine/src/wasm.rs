//! JavaScript bindings. Only compiled for `wasm32`.
//!
//! The browser's Web Worker creates one `Engine` and sends it messages:
//! `engine.call(method, argsJson, bytes)` returns the result as a JSON string, or throws
//! a JSON string in the structured error format (FR-13). JSON strings keep the boundary
//! simple and let the worker pass results straight back to the page.

use crate::session::Session;
use serde_json::Value;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Engine {
    session: Session,
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Engine {
        Engine {
            session: Session::new(),
        }
    }

    /// Runs one engine method. `bytes` carries file contents (e.g. an uploaded CSV).
    pub fn call(
        &mut self,
        method: &str,
        args_json: &str,
        bytes: Option<Vec<u8>>,
    ) -> Result<String, JsValue> {
        let args: Value = if args_json.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str(args_json).map_err(|e| {
                to_js_error(&crate::models::StandardErrorResponse::new(
                    "INVALID_REQUEST",
                    &format!("Arguments are not valid JSON: {}", e),
                    None,
                ))
            })?
        };
        match self.session.call(method, args, bytes.as_deref()) {
            Ok(value) => Ok(value.to_string()),
            Err(error) => Err(to_js_error(&error)),
        }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

fn to_js_error(error: &crate::models::StandardErrorResponse) -> JsValue {
    JsValue::from_str(&serde_json::to_string(error).unwrap_or_else(|_| {
        r#"{"error":{"code":"ENGINE_ERROR","message":"Unknown engine error."}}"#.to_string()
    }))
}
