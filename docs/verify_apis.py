import subprocess
import time
import urllib.request
import json
import os

workspace_dir = r"c:\B.Tech\Fifth_sem\SE\ite1"
target_dir = os.path.join(workspace_dir, "target", "debug")

exe1 = os.path.join(target_dir, "portion1_data_ingestion_preprocessing.exe")
exe2 = os.path.join(target_dir, "portion2_task_routing_configuration.exe")
exe3 = os.path.join(target_dir, "portion3_model_execution_evaluation.exe")

# Build all binaries if needed
cargo_exe = os.path.expanduser(r"~\.cargo\bin\cargo.exe")
if not (os.path.exists(exe1) and os.path.exists(exe2) and os.path.exists(exe3)):
    print("Building all binary targets...")
    subprocess.run([cargo_exe, "build", "--workspace"], cwd=workspace_dir, check=True)

print("--- Launching Binary 1 (Port 8001) ---")
p1 = subprocess.Popen([exe1], cwd=workspace_dir)

print("--- Launching Binary 2 (Port 8002) ---")
p2 = subprocess.Popen([exe2], cwd=workspace_dir)

print("--- Launching Binary 3 (Port 8003) ---")
p3 = subprocess.Popen([exe3], cwd=workspace_dir)

# Give 2 seconds for TCP bind
time.sleep(2)

results = {}

try:
    # 1. Test Health Checks
    for port, name in [(8001, "Portion 1 (Sneha)"), (8002, "Portion 2 (Satvik)"), (8003, "Portion 3 (Shreya)")]:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3) as res:
                body = json.loads(res.read().decode())
                results[f"{name} /health"] = ("SUCCESS", body.get("status"))
        except Exception as e:
            results[f"{name} /health"] = ("FAILED", str(e))

    # 2. Test Portion 2 Tasks Catalog API
    try:
        with urllib.request.urlopen("http://localhost:8002/api/v1/config/tasks", timeout=3) as res:
            tasks = json.loads(res.read().decode())
            results["Portion 2 /tasks"] = ("SUCCESS", f"{len(tasks)} tasks supported")
    except Exception as e:
        results["Portion 2 /tasks"] = ("FAILED", str(e))

    # 3. Test Portion 2 Validate Configuration API
    try:
        val_data = json.dumps({
            "dataset_id": "test-dataset-01",
            "task_type": "classification",
            "algorithm": "Random Forest Classifier",
            "target_column": "target",
            "feature_columns": ["col1", "col2"],
            "test_split_ratio": 0.2
        }).encode("utf-8")
        req = urllib.request.Request("http://localhost:8002/api/v1/config/validate", data=val_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as res:
            val_res = json.loads(res.read().decode())
            results["Portion 2 /validate"] = ("SUCCESS", f"is_valid: {val_res.get('is_valid')}")
    except Exception as e:
        results["Portion 2 /validate"] = ("FAILED", str(e))

    # 4. Test Portion 3 Supported Models Catalog API
    try:
        with urllib.request.urlopen("http://localhost:8003/api/v1/models/supported", timeout=3) as res:
            models = json.loads(res.read().decode())
            results["Portion 3 /supported"] = ("SUCCESS", f"Categories: {list(models.keys())}")
    except Exception as e:
        results["Portion 3 /supported"] = ("FAILED", str(e))

    # 5. Test Portion 3 Execute API Skeleton
    try:
        exec_payload = json.dumps({
            "dataset_id": "test-dataset-01",
            "task_type": "classification",
            "algorithm": "Random Forest Classifier",
            "target_column": "species",
            "feature_columns": ["sepal_length", "sepal_width"],
            "test_split_ratio": 0.2
        }).encode("utf-8")
        req = urllib.request.Request("http://localhost:8003/api/v1/models/execute", data=exec_payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as res:
            exec_res = json.loads(res.read().decode())
            results["Portion 3 /execute"] = ("SUCCESS", f"Execution ID: {exec_res.get('execution_id')}")
    except Exception as e:
        results["Portion 3 /execute"] = ("FAILED", str(e))

    # 6. Test Portion 2 Pipeline Dispatch (Calls P1 & P3 via Mock Client)
    try:
        disp_payload = json.dumps({
            "config": {
                "dataset_id": "test-dataset-01",
                "task_type": "classification",
                "algorithm": "Random Forest Classifier",
                "target_column": "target",
                "feature_columns": ["col1", "col2"],
                "test_split_ratio": 0.2
            }
        }).encode("utf-8")
        req = urllib.request.Request("http://localhost:8002/api/v1/pipeline/dispatch", data=disp_payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as res:
            disp_res = json.loads(res.read().decode())
            results["Portion 2 /dispatch"] = ("SUCCESS", f"Status: {disp_res.get('status')}")
    except Exception as e:
        results["Portion 2 /dispatch"] = ("FAILED", str(e))

    # 7. Test Portion 3 Model Comparison Dashboard API (FR-12)
    try:
        with urllib.request.urlopen("http://localhost:8003/api/v1/models/comparison/test-dataset-01", timeout=3) as res:
            comp_res = json.loads(res.read().decode())
            results["Portion 3 /comparison"] = ("SUCCESS", f"Total runs: {comp_res.get('total_runs')}")
    except Exception as e:
        results["Portion 3 /comparison"] = ("FAILED", str(e))

finally:
    p1.terminate()
    p2.terminate()
    p3.terminate()
    subprocess.call("taskkill /F /IM portion1_data_ingestion_preprocessing.exe /T 2>nul", shell=True)
    subprocess.call("taskkill /F /IM portion2_task_routing_configuration.exe /T 2>nul", shell=True)
    subprocess.call("taskkill /F /IM portion3_model_execution_evaluation.exe /T 2>nul", shell=True)

print("\n================== LIVE API TEST REPORT ==================")
all_passed = True
for k, v in results.items():
    print(f"[{v[0]}] {k} -> {v[1]}")
    if v[0] != "SUCCESS":
        all_passed = False

if all_passed:
    print("\n>>> ALL API SKELETONS ARE FULLY OPERATIONAL AND VERIFIED! <<<")
