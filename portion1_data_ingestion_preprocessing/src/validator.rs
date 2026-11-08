use crate::models::{CleanedDatasetDTO, StandardErrorResponse};

pub const MAX_FILE_SIZE_BYTES: usize = 10 * 1024 * 1024; // 10 MB (NFR-4)

/// Validates CSV upload (size limit, file format, empty file, UTF-8 encoding)
pub fn validate_csv_upload(filename: &str, bytes: &[u8]) -> Result<(), StandardErrorResponse> {
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
            &format!(
                "'{}' is not a CSV file. Only .csv files are accepted.",
                filename
            ),
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

/// Minimum data rows a dataset needs before it can be used for ML (SR-1, QA-5).
/// Portion 2 allows test splits up to 50%, so 10 rows leaves at least 5 rows each
/// for training and testing.
pub const MIN_DATA_ROWS: usize = 10;

/// Minimum columns: at least one feature plus a target (or two features for clustering).
pub const MIN_COLUMNS: usize = 2;

/// Checks that an uploaded dataset is big enough to be used for ML (SR-1, QA-5).
pub fn validate_dataset_size(
    num_rows: usize,
    num_cols: usize,
) -> Result<(), StandardErrorResponse> {
    check_size(num_rows, num_cols, "The dataset")
}

/// Checks that a cleaned dataset can still be handed to Portion 2 & 3 (SR-1, QA-2, QA-5):
/// it must keep enough rows and columns, and no column may be completely empty.
pub fn validate_cleaned_dataset(dto: &CleanedDatasetDTO) -> Result<(), StandardErrorResponse> {
    check_size(
        dto.total_rows,
        dto.columns.len(),
        "After preprocessing, the dataset",
    )?;

    let empty_columns: Vec<&String> = dto
        .columns
        .iter()
        .filter(|col| {
            dto.records
                .iter()
                .all(|rec| rec.get(*col).is_none_or(|v| v.is_null()))
        })
        .collect();
    if !empty_columns.is_empty() {
        let names: Vec<&str> = empty_columns.iter().map(|s| s.as_str()).collect();
        return Err(StandardErrorResponse::new(
            "EMPTY_COLUMN",
            &format!(
                "These columns have no values at all: {}. Choose 'Drop this column' for them.",
                names.join(", ")
            ),
            Some(serde_json::json!({ "columns": names })),
        ));
    }

    Ok(())
}

fn check_size(
    num_rows: usize,
    num_cols: usize,
    subject: &str,
) -> Result<(), StandardErrorResponse> {
    if num_cols < MIN_COLUMNS {
        return Err(StandardErrorResponse::new(
            "TOO_FEW_COLUMNS",
            &format!(
                "{} has {} column(s), but at least {} are needed (features plus a target).",
                subject, num_cols, MIN_COLUMNS
            ),
            Some(serde_json::json!({ "columns": num_cols, "min_columns": MIN_COLUMNS })),
        ));
    }
    if num_rows < MIN_DATA_ROWS {
        return Err(StandardErrorResponse::new(
            "TOO_FEW_ROWS",
            &format!(
                "{} has {} data row(s), but at least {} are needed to train and test a model.",
                subject, num_rows, MIN_DATA_ROWS
            ),
            Some(serde_json::json!({ "rows": num_rows, "min_rows": MIN_DATA_ROWS })),
        ));
    }
    Ok(())
}

/// Sanitizes uploaded filename to prevent directory traversal and injection attacks (SEC-2)
pub fn sanitize_filename(filename: &str) -> String {
    let clean = filename
        .replace('\\', "/")
        .split('/')
        .last()
        .unwrap_or("dataset.csv")
        .trim()
        .to_string();
    if clean.is_empty() {
        "dataset.csv".to_string()
    } else {
        clean
    }
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

    #[test]
    fn sanitizes_directory_traversal_filename() {
        assert_eq!(sanitize_filename("../../etc/passwd.csv"), "passwd.csv");
        assert_eq!(
            sanitize_filename("C:\\Users\\admin\\dataset.csv"),
            "dataset.csv"
        );
        assert_eq!(sanitize_filename("   "), "dataset.csv");
    }

    // ---- Issue 10: dataset size and cleaned-dataset checks (SR-1, QA-2, QA-5) ----

    fn dto(columns: &[&str], records: Vec<serde_json::Value>) -> CleanedDatasetDTO {
        CleanedDatasetDTO {
            dataset_id: "id".to_string(),
            total_rows: records.len(),
            columns: columns.iter().map(|c| c.to_string()).collect(),
            data_types: Default::default(),
            records: records
                .into_iter()
                .map(|r| serde_json::from_value(r).unwrap())
                .collect(),
        }
    }

    fn rows(n: usize) -> Vec<serde_json::Value> {
        (0..n)
            .map(|i| serde_json::json!({ "a": i, "b": "x" }))
            .collect()
    }

    #[test]
    fn accepts_dataset_at_minimum_size() {
        assert!(validate_dataset_size(MIN_DATA_ROWS, MIN_COLUMNS).is_ok());
    }

    #[test]
    fn rejects_single_column_dataset() {
        let e = validate_dataset_size(100, 1).unwrap_err();
        assert_eq!(e.error.code, "TOO_FEW_COLUMNS");
    }

    #[test]
    fn rejects_dataset_with_too_few_rows() {
        let e = validate_dataset_size(MIN_DATA_ROWS - 1, 5).unwrap_err();
        assert_eq!(e.error.code, "TOO_FEW_ROWS");
    }

    #[test]
    fn accepts_valid_cleaned_dataset() {
        assert!(validate_cleaned_dataset(&dto(&["a", "b"], rows(MIN_DATA_ROWS))).is_ok());
    }

    #[test]
    fn rejects_cleaned_dataset_that_lost_too_many_rows() {
        let e = validate_cleaned_dataset(&dto(&["a", "b"], rows(3))).unwrap_err();
        assert_eq!(e.error.code, "TOO_FEW_ROWS");
    }

    #[test]
    fn rejects_cleaned_dataset_with_a_completely_empty_column() {
        let records = (0..MIN_DATA_ROWS)
            .map(|i| serde_json::json!({ "a": i, "b": null }))
            .collect();
        let e = validate_cleaned_dataset(&dto(&["a", "b"], records)).unwrap_err();
        assert_eq!(e.error.code, "EMPTY_COLUMN");
        assert!(e.error.message.contains('b'));
    }
}
