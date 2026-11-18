//! End-to-end checks of the engine's public API, the same calls the browser makes.

use ml_engine::session::Session;
use serde_json::json;

fn iris_like_csv() -> String {
    let mut csv = String::from("sepal_length,sepal_width,species\n");
    for i in 0..30 {
        let (base, species) = match i % 3 {
            0 => (1.0, "setosa"),
            1 => (4.0, "versicolor"),
            _ => (7.0, "virginica"),
        };
        csv.push_str(&format!(
            "{},{},{}\n",
            base + (i as f64) * 0.01,
            base * 0.5,
            species
        ));
    }
    csv
}

#[test]
fn full_pipeline_runs_in_memory_and_purges() {
    let mut session = Session::new();

    let empty_history = session
        .call("comparison_history", json!({ "dataset_id": "all" }), None)
        .unwrap();
    assert_eq!(empty_history["total_runs"], 0);

    let upload = session
        .call(
            "upload_dataset",
            json!({ "filename": "flowers.csv" }),
            Some(iris_like_csv().as_bytes()),
        )
        .unwrap();
    let id = upload["dataset_id"].as_str().unwrap().to_string();

    let config = json!({
        "dataset_id": id,
        "task_type": "classification",
        "algorithm": "K-Nearest Neighbors (KNN)",
        "target_column": "species",
        "feature_columns": ["sepal_length", "sepal_width"],
        "test_split_ratio": 0.2,
        "num_clusters": null,
        "random_seed": null
    });
    session
        .call("validate_config", config.clone(), None)
        .unwrap();
    let dispatched = session
        .call("dispatch_pipeline", json!({ "config": config }), None)
        .unwrap();
    let result = &dispatched["execution_result"];
    assert!(result["metrics"]["accuracy"].as_f64().unwrap() > 0.9);

    let history = session
        .call("comparison_history", json!({ "dataset_id": id }), None)
        .unwrap();
    assert_eq!(history["total_runs"], 1);

    // SEC-1 / SEC-3: nothing survives a purge
    session.call("purge_session", json!({}), None).unwrap();
    assert!(session
        .call("schema", json!({ "dataset_id": id }), None)
        .is_err());
    let history = session.call("comparison_history", json!({}), None).unwrap();
    assert_eq!(history["total_runs"], 0);
}

#[test]
fn errors_use_the_structured_format() {
    let mut session = Session::new();
    let err = session
        .call("schema", json!({ "dataset_id": "missing" }), None)
        .unwrap_err();
    let value = serde_json::to_value(&err).unwrap();
    assert_eq!(value["error"]["code"], "DATASET_NOT_FOUND");
    assert_eq!(value["error"]["module"], "ml_engine");
    assert!(value["error"]["timestamp"].is_string());
}
