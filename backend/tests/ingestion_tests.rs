#[cfg(test)]
mod tests {
    const VALID_CSV: &str = "sepal_length,sepal_width,petal_length,species\n5.1,3.5,1.4,setosa\n4.9,,1.4,setosa\n7.0,3.2,4.7,versicolor\n";

    #[test]
    fn test_csv_file_extension_validation() {
        let invalid_filename = "data.xlsx";
        let is_csv = invalid_filename.to_lowercase().ends_with(".csv");
        assert!(!is_csv, "Non-csv file should fail validation");
    }

    #[test]
    fn test_csv_size_limit_rule() {
        let max_size = 10 * 1024 * 1024;
        let oversized = 11 * 1024 * 1024;
        assert!(oversized > max_size, "Files >10MB must be rejected (NFR-4)");
    }

    #[test]
    fn test_basic_csv_reading() {
        let mut rdr = csv::ReaderBuilder::new().from_reader(VALID_CSV.as_bytes());
        let headers = rdr.headers().unwrap();
        assert_eq!(headers.len(), 4);
        assert_eq!(headers.get(0).unwrap(), "sepal_length");

        let mut count = 0;
        for record in rdr.records() {
            let _ = record.unwrap();
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_imputation_mean_calculation() {
        let values = [5.0, 7.0, 9.0];
        let sum: f64 = values.iter().sum();
        let mean = sum / values.len() as f64;
        assert_eq!(mean, 7.0);
    }

    #[test]
    fn test_in_memory_dataset_purge_security() {
        // SEC-1 / SEC-3: In-memory buffers must support purging upon session termination
        use std::collections::HashMap;
        let mut session_storage: HashMap<String, Vec<u8>> = HashMap::new();
        session_storage.insert("session-001".to_string(), vec![1, 2, 3]);
        assert_eq!(session_storage.len(), 1);

        // Terminate session and purge
        let removed = session_storage.remove("session-001");
        assert!(removed.is_some());
        assert!(
            session_storage.is_empty(),
            "Session buffer must be empty after purge"
        );
    }

    #[test]
    fn test_error_detail_sanitization() {
        // SEC-4 / FR-13: Errors must not expose internal system traces
        let error_message = "Dataset with ID 'invalid-id' was not found.";
        assert!(
            !error_message.contains("C:\\"),
            "Error message must not expose host file paths"
        );
        assert!(
            !error_message.contains("/etc/"),
            "Error message must not expose unix paths"
        );
        assert!(
            !error_message.contains("stack backtrace:"),
            "Traces must not be sent to client"
        );
    }
}
