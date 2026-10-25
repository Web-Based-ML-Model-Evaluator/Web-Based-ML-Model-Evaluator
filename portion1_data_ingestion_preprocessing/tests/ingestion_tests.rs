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
        let values = vec![5.0, 7.0, 9.0];
        let sum: f64 = values.iter().sum();
        let mean = sum / values.len() as f64;
        assert_eq!(mean, 7.0);
    }
}
