use crate::models::StandardErrorResponse;

pub const MAX_FILE_SIZE_BYTES: usize = 10 * 1024 * 1024; // 10 MB (NFR-4)

/// Validates CSV upload (size limit, file format, empty file, UTF-8 encoding)
pub fn validate_csv_upload(
    filename: &str,
    bytes: &[u8],
) -> Result<(), StandardErrorResponse> {
    // 1. File size limit (NFR-4) -> 413
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

    // 2. File extension must be .csv -> 400
    if !filename.to_lowercase().ends_with(".csv") {
        return Err(StandardErrorResponse::new(
            "INVALID_FILE_FORMAT",
            &format!("'{}' is not a CSV file. Only .csv files are accepted.", filename),
            Some(serde_json::json!({ "filename": filename })),
        ));
    }

    // 3. File must not be empty (or only whitespace) -> 400
    if bytes.iter().all(|b| b.is_ascii_whitespace()) {
        return Err(StandardErrorResponse::new(
            "EMPTY_FILE",
            "The uploaded file is empty.",
            None,
        ));
    }

    // 4. File must be valid UTF-8 text -> 400
    if std::str::from_utf8(bytes).is_err() {
        return Err(StandardErrorResponse::new(
            "INVALID_ENCODING",
            "The file is not valid UTF-8 text. Please save it as 'CSV UTF-8' and try again.",
            None,
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_csv() {
        assert!(validate_csv_upload("data.csv", b"a,b\n1,2\n").is_ok());
    }

    #[test]
    fn rejects_non_csv_extension() {
        let e = validate_csv_upload("data.xlsx", b"a,b\n1,2\n").unwrap_err();
        assert_eq!(e.error.code, "INVALID_FILE_FORMAT");
    }

    #[test]
    fn rejects_empty_file() {
        let e = validate_csv_upload("data.csv", b"").unwrap_err();
        assert_eq!(e.error.code, "EMPTY_FILE");
    }

    #[test]
    fn rejects_file_over_10mb() {
        let big = vec![b'a'; MAX_FILE_SIZE_BYTES + 1];
        let e = validate_csv_upload("data.csv", &big).unwrap_err();
        assert_eq!(e.error.code, "FILE_TOO_LARGE");
    }

    #[test]
    fn rejects_invalid_utf8() {
        let e = validate_csv_upload("data.csv", &[0xff, 0xfe, 0xfa]).unwrap_err();
        assert_eq!(e.error.code, "INVALID_ENCODING");
    }
}