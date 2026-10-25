use crate::models::StandardErrorResponse;

pub const MAX_FILE_SIZE_BYTES: usize = 10 * 1024 * 1024; // 10 MB (NFR-4)

/// Validates CSV upload (File format, size limit, UTF-8 encoding)
///
/// TODO: Assigned to Sneha Ajith (PES2UG24CS480)
/// Requirements:
/// - Enforce maximum 10MB file limit (NFR-4) -> Return "FILE_TOO_LARGE" (413)
/// - Validate .csv file extension -> Return "INVALID_FILE_FORMAT" (400)
/// - Validate non-empty file -> Return "EMPTY_FILE" (400)
/// - Validate UTF-8 character encoding -> Return "INVALID_ENCODING" (400)
pub fn validate_csv_upload(
    _filename: &str,
    bytes: &[u8],
) -> Result<(), StandardErrorResponse> {
    // 1. Check file size restriction (NFR-4)
    if bytes.len() > MAX_FILE_SIZE_BYTES {
        return Err(StandardErrorResponse::new(
            "FILE_TOO_LARGE",
            &format!(
                "Uploaded file size ({:.2} MB) exceeds maximum allowed limit of 10 MB.",
                bytes.len() as f64 / (1024.0 * 1024.0)
            ),
            Some(serde_json::json!({
                "actual_bytes": bytes.len(),
                "max_allowed_bytes": MAX_FILE_SIZE_BYTES
            })),
        ));
    }

    // TODO: Sneha Ajith - Add your custom file extension, empty file, and encoding checks here

    Ok(())
}
