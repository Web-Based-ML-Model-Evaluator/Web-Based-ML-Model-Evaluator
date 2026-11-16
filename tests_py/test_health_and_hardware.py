"""
Tests for system health, fallback error routing, and hardware acceleration status.
"""

def test_health_check(client):
    """Verifies that the /health endpoint reports healthy status and service details."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "web_ml_unified_backend"
    assert "version" in data
    assert "Unified In-Memory Pipeline" in data.get("architecture", "")

def test_fallback_404_not_found(client):
    """Verifies that unmapped routes return a standard 404 error envelope."""
    res = client.get("/api/v1/nonexistent/endpoint")
    assert res.status_code == 404
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "RESOURCE_NOT_FOUND"

def test_hardware_status(client):
    """Verifies that the hardware telemetry endpoint returns CPU and acceleration info."""
    res = client.get("/api/v1/hardware/status")
    assert res.status_code == 200
    data = res.json()
    assert "logical_cores" in data
    assert data["logical_cores"] > 0
    assert "estimated_vram_mb" in data
    assert "webgpu_bridge_active" in data
    assert data["status"] == "OPERATIONAL_ULTRA_FAST"

def test_hardware_benchmark(client):
    """Verifies that the hardware matrix multiplication benchmark executes and returns metrics."""
    res = client.post("/api/v1/hardware/benchmark", json={"matrix_dim": 64})
    assert res.status_code == 200
    data = res.json()
    assert "elapsed_ms" in data
    assert "acceleration_grade" in data
    assert data["status"] == "COMPLETED"
