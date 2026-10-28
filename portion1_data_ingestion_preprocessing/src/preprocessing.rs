use crate::models::{
    CleanedDatasetDTO, ColumnSummary, DatasetUploadResponse, PreprocessingRequest,
    PreprocessingResponse, StandardErrorResponse,
};
use chrono::Utc;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct RawDataset {
    pub dataset_id: String,
    pub filename: String,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// Treats empty strings and common null markers as missing values.
/// (pub so Issue 3's imputation code can reuse the same definition of "missing")
pub fn is_missing(value: &str) -> bool {
    let v = value.trim();
    v.is_empty()
        || matches!(
            v.to_lowercase().as_str(),
            "na" | "n/a" | "nan" | "null" | "none"
        )
}

/// Parses raw CSV content and computes summary, data types, missing counts, and 5-row preview
/// (FR-1, FR-2)
///
/// 1. Parses CSV bytes with headers
/// 2. Computes total row count and column count
/// 3. Infers column data types (Numeric vs Categorical)
/// 4. Counts missing/null/nan values per column
/// 5. Generates a 5-row preview map
pub fn parse_csv_summary(
    dataset_id: &str,
    filename: &str,
    csv_bytes: &[u8],
) -> Result<(RawDataset, DatasetUploadResponse), StandardErrorResponse> {
    // Excel's "CSV UTF-8" adds an invisible marker (BOM) at the start; remove it
    let content = csv_bytes
        .strip_prefix(&[0xEF, 0xBB, 0xBF])
        .unwrap_or(csv_bytes);

    // 1. Parse CSV bytes with headers
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true) // don't crash if some rows have too few/many fields
        .trim(csv::Trim::All)
        .from_reader(content);

    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| {
            StandardErrorResponse::new(
                "CSV_PARSE_ERROR",
                &format!("Could not read the CSV header row: {}", e),
                None,
            )
        })?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();

    if headers.is_empty() || headers.iter().all(|h| h.is_empty()) {
        return Err(StandardErrorResponse::new(
            "NO_HEADERS",
            "The CSV has no header row. The first line must contain column names.",
            None,
        ));
    }

    let num_cols = headers.len();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (i, record) in reader.records().enumerate() {
        let record = record.map_err(|e| {
            StandardErrorResponse::new(
                "CSV_PARSE_ERROR",
                &format!("Problem reading data row {}: {}", i + 1, e),
                None,
            )
        })?;
        let mut row: Vec<String> = record.iter().map(|f| f.to_string()).collect();
        row.resize(num_cols, String::new()); // pad short rows, trim long ones
        rows.push(row);
    }

    if rows.is_empty() {
        return Err(StandardErrorResponse::new(
            "NO_DATA_ROWS",
            "The CSV has a header row but no data rows.",
            None,
        ));
    }

    // 2. Row count (column count is num_cols)
    let total_rows = rows.len();

    // 3 & 4. Per-column type inference and missing counts
    let mut column_summaries: Vec<ColumnSummary> = Vec::new();
    for (c, name) in headers.iter().enumerate() {
        let mut missing = 0usize;
        let mut present = 0usize;
        let mut numeric = 0usize;
        let mut samples: Vec<String> = Vec::new();

        for row in &rows {
            let value = row[c].trim();
            if is_missing(value) {
                missing += 1;
                continue;
            }
            present += 1;
            if value.parse::<f64>().is_ok() {
                numeric += 1;
            }
            if samples.len() < 3 && !samples.iter().any(|s| s == value) {
                samples.push(value.to_string());
            }
        }

        // Numeric only if EVERY non-missing value is a number
        let inferred_type = if present > 0 && numeric == present {
            "Numeric"
        } else {
            "Categorical"
        };

        let missing_percentage =
            ((missing as f64 / total_rows as f64) * 100.0 * 100.0).round() / 100.0;

        column_summaries.push(ColumnSummary {
            name: name.clone(),
            inferred_type: inferred_type.to_string(),
            missing_count: missing,
            missing_percentage,
            sample_values: samples,
        });
    }

    // 5. 5-row preview
    let preview_rows: Vec<HashMap<String, String>> = rows
        .iter()
        .take(5)
        .map(|row| {
            headers
                .iter()
                .cloned()
                .zip(row.iter().cloned())
                .collect::<HashMap<String, String>>()
        })
        .collect();

    let raw_dataset = RawDataset {
        dataset_id: dataset_id.to_string(),
        filename: filename.to_string(),
        headers: headers.clone(),
        rows,
    };

    let upload_response = DatasetUploadResponse {
        dataset_id: dataset_id.to_string(),
        filename: filename.to_string(),
        total_rows,
        total_columns: num_cols,
        column_names: headers,
        column_summaries,
        preview_rows,
        file_size_bytes: csv_bytes.len(),
        uploaded_at: Utc::now(),
    };

    Ok((raw_dataset, upload_response))
}

/// Applies cleaning and imputation strategies
///
/// TODO: Assigned to Sneha Ajith (PES2UG24CS480)
/// Requirements (FR-3):
/// 1. Drop columns marked with `ImputationStrategy::DropColumn`
/// 2. Impute numeric columns with Mean or Median
/// 3. Impute categorical columns with Mode or Constant
/// 4. Drop rows with nulls if specified
/// 5. Drop duplicate rows if `req.drop_duplicates` is true
pub fn clean_and_impute(
    raw: &RawDataset,
    _req: &PreprocessingRequest,
) -> Result<(CleanedDatasetDTO, PreprocessingResponse), StandardErrorResponse> {
    // SKELETON PLACEHOLDER: Sneha to implement imputation algorithm
    let cleaned_dto = CleanedDatasetDTO {
        dataset_id: raw.dataset_id.clone(),
        total_rows: raw.rows.len(),
        columns: raw.headers.clone(),
        data_types: HashMap::new(),
        records: vec![],
    };

    let response = PreprocessingResponse {
        dataset_id: raw.dataset_id.clone(),
        original_rows: raw.rows.len(),
        cleaned_rows: 0,
        remaining_columns: raw.headers.clone(),
        missing_values_remaining: 0,
        preview_rows: vec![],
        status: "SKELETON_READY_FOR_IMPLEMENTATION".to_string(),
    };

    Ok((cleaned_dto, response))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CSV: &str = "a,b,c\n1,x,5\n2,,6\n,y,\n";

    #[test]
    fn counts_rows_columns_and_missing() {
        let (raw, s) = parse_csv_summary("id", "t.csv", CSV.as_bytes()).unwrap();
        assert_eq!(s.total_rows, 3);
        assert_eq!(s.total_columns, 3);
        assert_eq!(raw.rows.len(), 3);
        assert_eq!(s.column_summaries[0].missing_count, 1);
        assert_eq!(s.column_summaries[1].missing_count, 1);
    }

    #[test]
    fn infers_numeric_and_categorical() {
        let (_, s) = parse_csv_summary("id", "t.csv", CSV.as_bytes()).unwrap();
        assert_eq!(s.column_summaries[0].inferred_type, "Numeric");
        assert_eq!(s.column_summaries[1].inferred_type, "Categorical");
    }

    #[test]
    fn preview_has_at_most_five_rows() {
        let mut csv = String::from("n\n");
        for i in 0..8 {
            csv.push_str(&format!("{}\n", i));
        }
        let (_, s) = parse_csv_summary("id", "t.csv", csv.as_bytes()).unwrap();
        assert_eq!(s.total_rows, 8);
        assert_eq!(s.preview_rows.len(), 5);
    }

    #[test]
    fn header_only_csv_is_rejected() {
        let e = parse_csv_summary("id", "t.csv", b"a,b,c\n").unwrap_err();
        assert_eq!(e.error.code, "NO_DATA_ROWS");
    }

    #[test]
    fn utf8_bom_is_ignored() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(b"a,b\n1,2\n");
        let (_, s) = parse_csv_summary("id", "t.csv", &bytes).unwrap();
        assert_eq!(s.column_names[0], "a");
    }
}