# ML Lab — High-Performance Web-Based Machine Learning Platform

**Production-Grade Unified Machine Learning Pipeline with WebGPU Acceleration**  
*Backend: Rust (Axum, Tokio, Serde, SmartCore) | Frontend: React 18, TypeScript, WebGPU, Vite*

---

## 1. System Overview

ML Lab is a high-performance web platform designed to ingest datasets, preprocess and impute missing values, configure and route machine learning tasks, train supervised and unsupervised models with hardware acceleration, and compare model metrics on a real-time leaderboard.

### Architecture Highlights:
- **Zero-Latency Unified In-Memory Pipeline:** All data processing, configuration validation, and model execution occur directly in memory inside a single cohesive Rust process (`backend/`), eliminating unnecessary inter-service network hops.
- **Client GPU & VRAM Acceleration (WebGPU):** Directly utilizes the user's browser GPU (NVIDIA, AMD, Intel, Apple Silicon) and dedicated VRAM using WGSL compute shaders for instant distance matrix evaluations, clustering, and matrix operations.
- **Warm Parchment Aesthetic UI:** Handcrafted editorial aesthetic with tactile parchment texture background, serif typography, decorative framing brackets, and clean responsive action cards matching the exact design specification.
- **Session Memory Security (SEC-1 / SEC-3):** Pure in-memory session isolation ensuring zero persistent data is retained upon session purge.

---

## 2. Directory Structure

```
├── backend/                       # Production Unified Rust Backend & ML Engine
│   ├── Cargo.toml                 # Serde, Axum, Tokio, SmartCore, CSV, Tracing
│   ├── src/
│   │   ├── main.rs               # Server entry point (port 8000 + compat listeners)
│   │   ├── lib.rs                # Library module exports
│   │   ├── state.rs              # Thread-safe in-memory session state (RAM)
│   │   ├── api/                  # Unified REST API controllers
│   │   │   ├── datasets.rs       # Ingestion, validation, preview, preprocessing, schema
│   │   │   ├── routing.rs        # Task routing, config validation, pipeline dispatch
│   │   │   ├── models.rs         # Model training, evaluation, metrics
│   │   │   ├── comparison.rs     # Multi-model leaderboard & ranking
│   │   │   └── hardware.rs       # GPU/VRAM hardware diagnostics & benchmarks
│   │   ├── core/                 # Compute & algorithmic core
│   │   │   ├── validator.rs      # CSV and schema validation
│   │   │   ├── preprocessing.rs  # Missing value imputation (Mean, Median, Mode, Drop)
│   │   │   ├── router_validation.rs # Parameter & split boundary validation
│   │   │   └── ml/               # Machine Learning & PCA algorithms
│   │   │       ├── algorithms.rs # Classifiers, Regressors, K-Means
│   │   │       ├── metrics.rs    # Confusion matrix, MAE/RMSE/R², Silhouette, Davies-Bouldin
│   │   │       └── hardware_accel.rs # Hardware probe & acceleration diagnostics
│   │   └── models/               # Strongly-typed Serde DTOs & Error contracts
│   └── tests/                    # 108 Comprehensive Unit and Integration Tests
│
├── frontend/                      # Modern React + TypeScript + WebGPU Client
│   ├── index.html                # Typography (Playfair Display & Inter), textured canvas
│   ├── vite.config.mjs           # Lightning-fast Vite bundler (~240ms builds)
│   └── src/
│       ├── index.tsx             # Client bootstrap
│       ├── App.tsx               # Main stateful application controller
│       ├── components/
│       │   ├── Navbar.tsx        # Stacked orange diamonds logo & user profile avatar
│       │   ├── Hero.tsx          # Decorative corner brackets, serif title with brush underline
│       │   ├── GpuAccelerationPanel.tsx # WebGPU status, VRAM monitor, shader benchmark
│       │   ├── DatasetIngestionView.tsx # Data upload, preview, imputation
│       │   ├── TaskConfigurationView.tsx# Task selection, feature/target configuration
│       │   └── EvaluationDashboardView.tsx # Execution metrics, heatmap, 2D PCA, leaderboard
│       ├── services/
│       │   ├── api.ts            # Consolidated API client
│       │   └── webgpu.ts         # Browser WebGPU / VRAM execution engine (WGSL compute shaders)
│       └── styles/
│           └── theme.css         # Warm parchment color, tactile paper fiber texture
│
├── documents/                     # Consolidated project documentation
│   ├── Team13_Consolidated_Documentation.pdf # All 4 project specifications in one 41-page PDF
│   ├── Functional_Requirements_Team13.pdf
│   ├── SRS_Team_13.pdf
│   ├── Team13_SASD.pdf
│   └── test_plan_team_13.pdf
│
├── docs/                          # Architecture & API specifications
│   ├── API_SPECIFICATION.md
│   ├── ARCHITECTURE_AND_RESPONSIBILITY_MATRIX.md
│   └── TECH_STACK_AUDIT_REPORT.md
│
├── Cargo.toml                     # Root workspace Cargo manifest
├── package.json                   # Unified development and build scripts
└── vite.config.mjs                # Bundler configuration
```

---

## 3. Quick Start & Execution

### Prerequisites
- **Rust:** `1.80+` (Cargo)
- **Node.js:** `v18+` (npm)

### Run Backend
```bash
# Starts the high-performance unified server on http://localhost:8000
npm run start:backend
```

### Run Frontend
```bash
# Starts the Vite development server on http://localhost:3000
npm run dev
```

### Run Full Test Suite
```bash
# Runs all 108 backend tests and TypeScript typecheck
npm run test:all
```

### Build for Production
```bash
# Compiles optimized frontend bundle (dist/)
npm run build
```

---

## 4. Documentation
All project specifications (Functional Requirements, SRS, System Architecture, and Test Plan) are merged into a single consolidated document located at:
- [`documents/Team13_Consolidated_Documentation.pdf`](file:///c:/B.Tech/Fifth_sem/SE/ite1/documents/Team13_Consolidated_Documentation.pdf)
