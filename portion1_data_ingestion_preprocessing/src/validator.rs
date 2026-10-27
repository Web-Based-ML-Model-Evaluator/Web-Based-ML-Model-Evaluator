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

    // 2. Check file format (.csv extension)
    if !_filename.is_empty() && !_filename.to_lowercase().ends_with(".csv") {
        return Err(StandardErrorResponse::new(
            "INVALID_FILE_FORMAT",
            &format!("File '{}' is not a supported format. Please upload a valid .csv file.", _filename),
            Some(serde_json::json!({ "received_filename": _filename })),
        ));
    }

    // 3. Check for empty file
    if bytes.is_empty() {
        return Err(StandardErrorResponse::new(
            "EMPTY_FILE",
            "The uploaded CSV file is empty.",
            None,
        ));
    }

    Ok(())
}

#[cfg(test)]
mod validator_tests {
    use super::*;

    #[test]
    fn test_valid_csv_upload() {
        let csv_content = b"a,b,c\n1,2,3";
        let res = validate_csv_upload("dataset.csv", csv_content);
        assert!(res.is_ok());
    }

    #[test]
    fn test_file_size_exceeds_10mb_rejected() {
        // Enforces NFR-4: Restrict file uploads to maximum 10 MB limit
        let oversized = vec![0u8; MAX_FILE_SIZE_BYTES + 1];
        let res = validate_csv_upload("big_data.csv", &oversized);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert_eq!(err.error.code, "FILE_TOO_LARGE");
    }

    #[test]
    fn test_file_size_within_10mb_accepted() {
        // 1MB file is within 10MB limit
        let normal_size = vec![b'a'; 1024 * 1024];
        let res = validate_csv_upload("normal_data.csv", &normal_size);
        assert!(res.is_ok());
    }

    #[test]
    fn test_invalid_extension_rejected() {
        let res = validate_csv_upload("data.xlsx", b"some data");
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert_eq!(err.error.code, "INVALID_FILE_FORMAT");
    }

    #[test]
    fn test_empty_file_rejected() {
        let res = validate_csv_upload("data.csv", b"");
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert_eq!(err.error.code, "EMPTY_FILE");
    }
}
