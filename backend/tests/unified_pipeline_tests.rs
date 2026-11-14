use web_ml_backend::state::AppState;

#[test]
fn test_unified_state_in_memory_isolation_and_purge() {
    let state = AppState::default();

    // Verify initial clean state
    assert_eq!(state.raw_datasets.read().unwrap().len(), 0);
    assert_eq!(state.upload_summaries.read().unwrap().len(), 0);
    assert_eq!(state.cleaned_datasets.read().unwrap().len(), 0);

    // Record dummy data
    state.comparison_store.seed_mock_runs();
    assert!(!state.comparison_store.get_all_runs().is_empty());

    // Purge session
    state.purge_all();
    assert_eq!(state.comparison_store.get_all_runs().len(), 0);
}

#[test]
fn test_hardware_status_probe() {
    let state = AppState::default();
    assert!(state.hardware_status.logical_cores >= 1);
    assert!(state.hardware_status.gpu_acceleration_supported);
    assert_eq!(state.hardware_status.status, "OPERATIONAL_ULTRA_FAST");
}
