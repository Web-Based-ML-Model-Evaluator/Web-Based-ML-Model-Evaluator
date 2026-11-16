"""
Pytest configuration and shared fixtures for the Web-Based ML Model Evaluator test suite.
Provides:
  - Automatic backend process lifecycle management (starts backend if not already active).
  - Clean API client fixture with convenient helper methods.
  - Realistic datasets: Iris (Classification), Housing/Price (Regression), Clustering, and Dirty data with nulls.
  - Dummy pre-trained model file generators (.pkl, .joblib, .pt, .onnx, .h5).
"""

import os
import sys
import time
import socket
import subprocess
import tempfile
import urllib.request
import json
import pytest
import requests

# Root directory of the repository
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def find_free_port() -> int:
    """Finds an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def is_backend_alive(url: str, timeout: float = 1.0) -> bool:
    """Checks if the backend /health endpoint responds with 200 OK."""
    try:
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                return data.get("status") == "healthy"
    except Exception:
        pass
    return False

def resolve_backend_binary() -> str:
    """Locates or builds the web_ml_backend executable."""
    ext = ".exe" if sys.platform == "win32" else ""
    target_debug = os.path.join(ROOT_DIR, "target", "debug", f"web_ml_backend{ext}")
    target_release = os.path.join(ROOT_DIR, "target", "release", f"web_ml_backend{ext}")

    if os.path.exists(target_debug):
        return target_debug
    if os.path.exists(target_release):
        return target_release

    # Binary not found: build it
    print("\n[Pytest Setup] Building backend binary...")
    cargo_runner = os.path.join(ROOT_DIR, "scripts", "cargo-runner.js")
    if os.path.exists(cargo_runner):
        subprocess.run(["node", cargo_runner, "build", "--workspace"], cwd=ROOT_DIR, check=True)
    else:
        subprocess.run(["cargo", "build", "--workspace"], cwd=ROOT_DIR, check=True)

    if os.path.exists(target_debug):
        return target_debug
    raise FileNotFoundError(f"Could not locate web_ml_backend binary at {target_debug}")

@pytest.fixture(scope="session")
def base_url():
    """
    Session-scoped fixture providing the base URL of the running backend.
    Detects if a server is already running, or launches one automatically on a dedicated test port.
    """
    # 1. Check if user/CI set a specific URL or if default port is active
    env_url = os.environ.get("BACKEND_TEST_URL")
    if env_url and is_backend_alive(env_url):
        print(f"\n[Pytest Setup] Connecting to existing backend at {env_url}")
        yield env_url
        return

    # Check default 8080
    if is_backend_alive("http://127.0.0.1:8080"):
        print("\n[Pytest Setup] Found active backend on http://127.0.0.1:8080")
        yield "http://127.0.0.1:8080"
        return

    # 2. Spawn our own backend instance on an isolated port
    test_port = find_free_port()
    base = f"http://127.0.0.1:{test_port}"
    binary_path = resolve_backend_binary()

    env = os.environ.copy()
    env["PORT"] = str(test_port)

    print(f"\n[Pytest Setup] Launching backend binary on {base}...")
    proc = subprocess.Popen(
        [binary_path],
        cwd=ROOT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for /health to become ready
    deadline = time.time() + 25.0
    started = False
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(f"Backend process terminated prematurely: {stderr}\n{stdout}")
        if is_backend_alive(base, timeout=0.5):
            started = True
            break
        time.sleep(0.3)

    if not started:
        proc.kill()
        raise TimeoutError(f"Backend failed to respond on {base}/health within 25 seconds.")

    print(f"[Pytest Setup] Backend successfully initialized on {base}")
    yield base

    # Teardown: terminate backend
    print("\n[Pytest Teardown] Stopping test backend instance...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class APIClient:
    """Helper client for interacting with the ML backend in tests."""
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def get(self, endpoint: str, **kwargs):
        return self.session.get(f"{self.base_url}{endpoint}", **kwargs)

    def post(self, endpoint: str, **kwargs):
        return self.session.post(f"{self.base_url}{endpoint}", **kwargs)

    def delete(self, endpoint: str, **kwargs):
        return self.session.delete(f"{self.base_url}{endpoint}", **kwargs)

    def upload_dataset(self, filename: str, content: bytes or str):
        if isinstance(content, str):
            content = content.encode("utf-8")
        files = {"file": (filename, content, "text/csv")}
        return self.post("/api/v1/datasets/upload", files=files)

    def preprocess_dataset(self, dataset_id: str, request_data: dict):
        return self.post(f"/api/v1/datasets/{dataset_id}/preprocess", json=request_data)

    def execute_model(self, request_data: dict):
        return self.post("/api/v1/models/execute", json=request_data)

    def upload_model(self, filename: str, content: bytes, task_type: str = "classification", architecture: str = "MLP"):
        files = {"model": (filename, content, "application/octet-stream")}
        data = {"task_type": task_type, "architecture": architecture}
        return self.post("/api/v1/models/upload", files=files, data=data)

    def evaluate_ingested_model(self, request_data: dict):
        return self.post("/api/v1/models/evaluate-ingested", json=request_data)

    def compare_models(self, request_data: dict):
        return self.post("/api/v1/models/compare", json=request_data)

@pytest.fixture
def client(base_url):
    """Yields a configured APIClient instance pointing to the test backend."""
    return APIClient(base_url)

@pytest.fixture
def iris_csv_bytes():
    """Generates a valid 150-sample Iris dataset in CSV format."""
    lines = ["sepal_length,sepal_width,petal_length,petal_width,species"]
    # 50 setosa
    for i in range(50):
        sl = round(4.5 + (i % 10) * 0.1, 2)
        sw = round(3.0 + (i % 8) * 0.1, 2)
        pl = round(1.2 + (i % 6) * 0.1, 2)
        pw = round(0.1 + (i % 4) * 0.1, 2)
        lines.append(f"{sl},{sw},{pl},{pw},setosa")
    # 50 versicolor
    for i in range(50):
        sl = round(5.5 + (i % 12) * 0.1, 2)
        sw = round(2.5 + (i % 6) * 0.1, 2)
        pl = round(4.0 + (i % 10) * 0.1, 2)
        pw = round(1.2 + (i % 5) * 0.1, 2)
        lines.append(f"{sl},{sw},{pl},{pw},versicolor")
    # 50 virginica
    for i in range(50):
        sl = round(6.5 + (i % 14) * 0.1, 2)
        sw = round(2.8 + (i % 7) * 0.1, 2)
        pl = round(5.5 + (i % 12) * 0.1, 2)
        pw = round(1.8 + (i % 6) * 0.1, 2)
        lines.append(f"{sl},{sw},{pl},{pw},virginica")
    return "\n".join(lines).encode("utf-8")

@pytest.fixture
def regression_csv_bytes():
    """Generates a continuous regression dataset (housing features -> price)."""
    lines = ["area_sqft,bedrooms,bathrooms,age_years,price"]
    for i in range(80):
        area = 800 + i * 25
        bed = 1 + (i % 4)
        bath = 1 + (i % 3)
        age = (i % 30) + 1
        price = round(area * 150.0 + bed * 10000.0 + bath * 8000.0 - age * 500.0, 2)
        lines.append(f"{area},{bed},{bath},{age},{price}")
    return "\n".join(lines).encode("utf-8")

@pytest.fixture
def clustering_csv_bytes():
    """Generates a multi-dimensional numerical dataset for clustering."""
    lines = ["x1,x2,x3,x4"]
    # 3 natural clusters
    for i in range(30):
        lines.append(f"{1.0 + (i%5)*0.1},{2.0 + (i%5)*0.1},{1.5 + (i%5)*0.1},{2.5 + (i%5)*0.1}")
    for i in range(30):
        lines.append(f"{10.0 + (i%5)*0.1},{12.0 + (i%5)*0.1},{11.5 + (i%5)*0.1},{12.5 + (i%5)*0.1}")
    for i in range(30):
        lines.append(f"{25.0 + (i%5)*0.1},{24.0 + (i%5)*0.1},{26.5 + (i%5)*0.1},{25.5 + (i%5)*0.1}")
    return "\n".join(lines).encode("utf-8")

@pytest.fixture
def dirty_csv_bytes():
    """Generates a dataset with missing values (nulls) and duplicate rows."""
    lines = [
        "feature_a,feature_b,category,target",
        "10.5,20.0,alpha,1",
        "12.0,,beta,1",            # missing feature_b
        ",18.5,alpha,0",           # missing feature_a
        "15.0,22.0,,0",            # missing category
        "10.5,20.0,alpha,1",       # duplicate row
        "11.0,21.0,gamma,1",
        "14.2,23.1,beta,0",
        "10.5,20.0,alpha,1",       # duplicate row
        "13.5,19.0,gamma,1",
        "16.0,25.0,beta,0",
        "12.5,22.4,alpha,1",
        "17.1,24.0,gamma,0",
        "18.0,26.5,beta,1",
        "19.2,,alpha,0",           # missing feature_b
        "11.5,19.8,gamma,1",
        "13.8,21.5,beta,0",
        ",20.5,alpha,1",           # missing feature_a
        "14.9,23.0,gamma,0",
        "15.5,24.2,beta,1",
        "16.7,25.1,alpha,0"
    ]
    return "\n".join(lines).encode("utf-8")
