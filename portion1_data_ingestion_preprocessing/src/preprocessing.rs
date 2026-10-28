use crate::models::{
    CleanedDatasetDTO, ColumnSummary, DatasetUploadResponse, ImputationStrategy,
    PreprocessingRequest, PreprocessingResponse, StandardErrorResponse,
};
use chrono::Utc;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Default)]
pub struct RawDataset {
    pub dataset_id: String,
    pub filename: String,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// Treats empty strings and common null markers as missing values.
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

// ----------------------------------------------------------------------------
// FR-3: Cleaning and imputation helpers
// ----------------------------------------------------------------------------

/// A column is numeric only if it has at least one value and EVERY non-missing value is a number.
fn column_is_numeric(rows: &[Vec<String>], c: usize) -> bool {
    let mut present = 0usize;
    let mut numeric = 0usize;
    for row in rows {
        let v = row[c].trim();
        if is_missing(v) {
            continue;
        }
        present += 1;
        if v.parse::<f64>().is_ok() {
            numeric += 1;
        }
    }
    present > 0 && numeric == present
}

/// Rounds to 4 decimal places and prints without trailing zeros (2.0 -> "2", 3.05714 -> "3.0571").
fn format_number(v: f64) -> String {
    let rounded = (v * 10_000.0).round() / 10_000.0;
    format!("{}", rounded)
}

/// Most common value. If two values tie, the one that appeared first wins.
/// The caller guarantees `values` is not empty.
fn most_frequent(values: &[&str]) -> String {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut order: Vec<&str> = Vec::new();
    for &v in values {
        let count = counts.entry(v).or_insert(0);
        if *count == 0 {
            order.push(v);
        }
        *count += 1;
    }
    let mut best = order[0];
    let mut best_count = counts[best];
    for &v in &order {
        let n = counts[v];
        if n > best_count {
            best = v;
            best_count = n;
        }
    }
    best.to_string()
}

/// Applies cleaning and imputation strategies (FR-3)
///
/// Order of operations:
/// 1. Validate the rules (unknown columns, mean/median on text columns, bad constants)
/// 2. Drop columns marked `DropColumn`, and trim whitespace if requested
/// 3. Drop duplicate rows if `req.drop_duplicates` is true
/// 4. Drop rows that have a missing value in a `DropRowsWithNull` column
/// 5. Impute remaining missing values with Mean, Median, Mode or Constant
/// 6. Build the cleaned records (numbers as JSON numbers, missing as null) for Portions 2 and 3
pub fn clean_and_impute(
    raw: &RawDataset,
    req: &PreprocessingRequest,
) -> Result<(CleanedDatasetDTO, PreprocessingResponse), StandardErrorResponse> {
    let fail = |code: &str, message: String| StandardErrorResponse::new(code, &message, None);

    // Column types come from the ORIGINAL data, before anything is changed
    let original_numeric: Vec<bool> = (0..raw.headers.len())
        .map(|c| column_is_numeric(&raw.rows, c))
        .collect();

    // 1. Validate rules
    for (col, strategy) in &req.imputation_rules {
        let idx = match raw.headers.iter().position(|h| h == col) {
            Some(i) => i,
            None => {
                return Err(fail(
                    "UNKNOWN_COLUMN",
                    format!("Column '{}' in imputation_rules does not exist in this dataset.", col),
                ));
            }
        };
        match strategy {
            ImputationStrategy::Mean | ImputationStrategy::Median if !original_numeric[idx] => {
                return Err(fail(
                    "INVALID_STRATEGY",
                    format!(
                        "Cannot use mean or median on column '{}' because it is not numeric. Use mode or a constant instead.",
                        col
                    ),
                ));
            }
            ImputationStrategy::Constant(value)
                if original_numeric[idx] && value.trim().parse::<f64>().is_err() =>
            {
                return Err(fail(
                    "INVALID_CONSTANT",
                    format!("Column '{}' is numeric, so its constant must be a number (got '{}').", col, value),
                ));
            }
            _ => {}
        }
    }

    // 2. Keep only the columns that are not dropped
    let kept: Vec<usize> = raw
        .headers
        .iter()
        .enumerate()
        .filter(|(_, name)| {
            !matches!(
                req.imputation_rules.get(*name),
                Some(ImputationStrategy::DropColumn)
            )
        })
        .map(|(i, _)| i)
        .collect();

    if kept.is_empty() {
        return Err(fail(
            "NO_COLUMNS_REMAINING",
            "All columns were dropped. Keep at least one column.".to_string(),
        ));
    }

    let headers: Vec<String> = kept.iter().map(|&i| raw.headers[i].clone()).collect();
    let numeric: Vec<bool> = kept.iter().map(|&i| original_numeric[i]).collect();

    let mut rows: Vec<Vec<String>> = raw
        .rows
        .iter()
        .map(|row| {
            kept.iter()
                .map(|&i| {
                    if req.trim_whitespace {
                        row[i].trim().to_string()
                    } else {
                        row[i].clone()
                    }
                })
                .collect()
        })
        .collect();

    // 3. Drop duplicate rows (done before imputation so duplicates don't skew the mean/median)
    if req.drop_duplicates {
        let mut seen: HashSet<Vec<String>> = HashSet::new();
        rows.retain(|row| seen.insert(row.clone()));
    }

    // 4. Drop rows that have a missing value in a DropRowsWithNull column
    let null_check_cols: Vec<usize> = (0..headers.len())
        .filter(|&j| {
            matches!(
                req.imputation_rules.get(&headers[j]),
                Some(ImputationStrategy::DropRowsWithNull)
            )
        })
        .collect();
    if !null_check_cols.is_empty() {
        rows.retain(|row| null_check_cols.iter().all(|&j| !is_missing(&row[j])));
    }

    if rows.is_empty() {
        return Err(fail(
            "NO_ROWS_REMAINING",
            "No rows remain after cleaning. Try less strict rules.".to_string(),
        ));
    }

    // 5. Impute missing values column by column
    for j in 0..headers.len() {
        let strategy = match req.imputation_rules.get(&headers[j]) {
            Some(s) => s,
            None => continue, // no rule for this column: leave it as is
        };

        let fill: String = {
            let present: Vec<&str> = rows
                .iter()
                .map(|r| r[j].as_str())
                .filter(|v| !is_missing(v))
                .collect();

            match strategy {
                ImputationStrategy::Mean | ImputationStrategy::Median => {
                    let mut nums: Vec<f64> = present
                        .iter()
                        .filter_map(|v| v.trim().parse::<f64>().ok())
                        .collect();
                    if nums.is_empty() {
                        return Err(fail(
                            "CANNOT_IMPUTE",
                            format!("Column '{}' has no values to calculate a mean or median from.", headers[j]),
                        ));
                    }
                    if matches!(strategy, ImputationStrategy::Mean) {
                        format_number(nums.iter().sum::<f64>() / nums.len() as f64)
                    } else {
                        nums.sort_by(|a, b| a.total_cmp(b));
                        let mid = nums.len() / 2;
                        let median = if nums.len() % 2 == 0 {
                            (nums[mid - 1] + nums[mid]) / 2.0
                        } else {
                            nums[mid]
                        };
                        format_number(median)
                    }
                }
                ImputationStrategy::Mode => {
                    if present.is_empty() {
                        return Err(fail(
                            "CANNOT_IMPUTE",
                            format!("Column '{}' has no values to find a most common value from.", headers[j]),
                        ));
                    }
                    most_frequent(&present)
                }
                ImputationStrategy::Constant(value) => value.trim().to_string(),
                // Already handled in steps 2 and 4
                ImputationStrategy::DropColumn | ImputationStrategy::DropRowsWithNull => continue,
            }
        };

        for row in rows.iter_mut() {
            if is_missing(&row[j]) {
                row[j] = fill.clone();
            }
        }
    }

    // 6. Build the cleaned dataset for downstream portions
    let mut records: Vec<HashMap<String, serde_json::Value>> = Vec::with_capacity(rows.len());
    let mut missing_values_remaining = 0usize;

    for row in &rows {
        let mut record: HashMap<String, serde_json::Value> = HashMap::new();
        for (j, name) in headers.iter().enumerate() {
            let cell = row[j].trim();
            let value = if is_missing(cell) {
                missing_values_remaining += 1;
                serde_json::Value::Null
            } else if numeric[j] {
                cell.parse::<f64>()
                    .ok()
                    .and_then(serde_json::Number::from_f64)
                    .map(serde_json::Value::Number)
                    .unwrap_or_else(|| serde_json::Value::String(cell.to_string()))
            } else {
                serde_json::Value::String(cell.to_string())
            };
            record.insert(name.clone(), value);
        }
        records.push(record);
    }

    let data_types: HashMap<String, String> = headers
        .iter()
        .zip(numeric.iter())
        .map(|(name, &is_num)| {
            let kind = if is_num { "Numeric" } else { "Categorical" };
            (name.clone(), kind.to_string())
        })
        .collect();

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

    let cleaned_dto = CleanedDatasetDTO {
        dataset_id: raw.dataset_id.clone(),
        total_rows: rows.len(),
        columns: headers.clone(),
        data_types,
        records,
    };

    let response = PreprocessingResponse {
        dataset_id: raw.dataset_id.clone(),
        original_rows: raw.rows.len(),
        cleaned_rows: rows.len(),
        remaining_columns: headers,
        missing_values_remaining,
        preview_rows,
        status: "COMPLETED".to_string(),
    };

    Ok((cleaned_dto, response))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CSV: &str = "a,b,c\n1,x,5\n2,,6\n,y,\n";

    // ---- Issue 2: parsing ----

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

    // ---- Issue 3: cleaning and imputation ----

    fn raw_from(csv: &str) -> RawDataset {
        parse_csv_summary("id", "t.csv", csv.as_bytes()).unwrap().0
    }

    fn request(rules: Vec<(&str, ImputationStrategy)>, drop_duplicates: bool) -> PreprocessingRequest {
        PreprocessingRequest {
            imputation_rules: rules
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            drop_duplicates,
            trim_whitespace: true,
        }
    }

    #[test]
    fn mean_imputation_fills_missing_numbers() {
        let raw = raw_from("a,b\n1,x\n,y\n3,z\n");
        let req = request(vec![("a", ImputationStrategy::Mean)], false);
        let (dto, resp) = clean_and_impute(&raw, &req).unwrap();
        assert_eq!(resp.missing_values_remaining, 0);
        assert_eq!(dto.records[1]["a"], serde_json::json!(2.0));
    }

    #[test]
    fn median_uses_average_of_middle_two() {
        let raw = raw_from("a,b\n1,x\n2,x\n3,x\n10,x\n,x\n");
        let req = request(vec![("a", ImputationStrategy::Median)], false);
        let (dto, _) = clean_and_impute(&raw, &req).unwrap();
        assert_eq!(dto.records[4]["a"], serde_json::json!(2.5));
    }

    #[test]
    fn mode_fills_most_common_text() {
        let raw = raw_from("a,b\nx,1\nx,2\ny,3\n,4\n");
        let req = request(vec![("a", ImputationStrategy::Mode)], false);
        let (dto, _) = clean_and_impute(&raw, &req).unwrap();
        assert_eq!(dto.records[3]["a"], serde_json::json!("x"));
    }

    #[test]
    fn constant_fills_with_given_value() {
        let raw = raw_from("a,b\nx,1\n,2\n");
        let req = request(
            vec![("a", ImputationStrategy::Constant("unknown".to_string()))],
            false,
        );
        let (dto, _) = clean_and_impute(&raw, &req).unwrap();
        assert_eq!(dto.records[1]["a"], serde_json::json!("unknown"));
    }

    #[test]
    fn drop_column_removes_it() {
        let raw = raw_from("a,b\n1,x\n2,y\n");
        let req = request(vec![("b", ImputationStrategy::DropColumn)], false);
        let (dto, resp) = clean_and_impute(&raw, &req).unwrap();
        assert_eq!(dto.columns, vec!["a".to_string()]);
        assert_eq!(resp.remaining_columns.len(), 1);
    }

    #[test]
    fn drop_rows_with_null_removes_those_rows() {
        let raw = raw_from("a,b\n1,x\n,y\n3,z\n");
        let req = request(vec![("a", ImputationStrategy::DropRowsWithNull)], false);
        let (_, resp) = clean_and_impute(&raw, &req).unwrap();
        assert_eq!(resp.original_rows, 3);
        assert_eq!(resp.cleaned_rows, 2);
    }

    #[test]
    fn duplicates_are_removed_when_requested() {
        let raw = raw_from("a,b\n1,x\n1,x\n2,y\n");
        let (_, resp) = clean_and_impute(&raw, &request(vec![], true)).unwrap();
        assert_eq!(resp.cleaned_rows, 2);
        let (_, resp) = clean_and_impute(&raw, &request(vec![], false)).unwrap();
        assert_eq!(resp.cleaned_rows, 3);
    }

    #[test]
    fn columns_without_a_rule_keep_missing_as_null() {
        let raw = raw_from("a,b\n1,x\n,y\n");
        let (dto, resp) = clean_and_impute(&raw, &request(vec![], false)).unwrap();
        assert_eq!(dto.records[0]["a"], serde_json::json!(1.0));
        assert_eq!(dto.records[1]["a"], serde_json::Value::Null);
        assert_eq!(resp.missing_values_remaining, 1);
    }

    #[test]
    fn mean_on_text_column_is_rejected() {
        let raw = raw_from("a,b\n1,x\n2,y\n");
        let req = request(vec![("b", ImputationStrategy::Mean)], false);
        let e = clean_and_impute(&raw, &req).unwrap_err();
        assert_eq!(e.error.code, "INVALID_STRATEGY");
    }

    #[test]
    fn unknown_column_is_rejected() {
        let raw = raw_from("a,b\n1,x\n2,y\n");
        let req = request(vec![("zzz", ImputationStrategy::Mode)], false);
        let e = clean_and_impute(&raw, &req).unwrap_err();
        assert_eq!(e.error.code, "UNKNOWN_COLUMN");
    }
}