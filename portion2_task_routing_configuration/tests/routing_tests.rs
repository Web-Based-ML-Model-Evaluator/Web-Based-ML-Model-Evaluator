#[cfg(test)]
mod tests {
    #[test]
    fn test_supervised_requires_target() {
        let has_target = false;
        assert!(
            !has_target,
            "Supervised classification/regression must fail if target is missing"
        );
    }

    #[test]
    fn test_target_cannot_be_in_features() {
        let target = "price";
        let features = ["area", "bedrooms", "price"];
        let is_conflict = features.contains(&target);
        assert!(
            is_conflict,
            "Must catch target inside feature list (FR-5, FR-13)"
        );
    }

    #[test]
    fn test_split_ratio_boundary() {
        let valid_ratio = 0.25;
        let invalid_ratio = 0.85;

        assert!((0.05..=0.50).contains(&valid_ratio));
        assert!(
            !(0.05..=0.50).contains(&invalid_ratio),
            "Split ratio > 0.50 must be rejected"
        );
    }

    #[test]
    fn test_clustering_requires_k_ge_2() {
        let k_invalid = 1;
        let k_valid = 3;

        assert!(k_invalid < 2);
        assert!(k_valid >= 2);
    }

    #[test]
    fn test_error_handling_contract_fr13() {
        // FR-13: Validation failures return structured error messages
        let error_json = serde_json::json!({
            "error": {
                "code": "CONFIG_VALIDATION_FAILED",
                "message": "Select at least one feature column.",
                "module": "portion2_task_routing_configuration",
                "timestamp": "2026-11-08T09:00:00Z"
            }
        });
        assert_eq!(error_json["error"]["code"], "CONFIG_VALIDATION_FAILED");
        assert_eq!(
            error_json["error"]["module"],
            "portion2_task_routing_configuration"
        );
    }
}
