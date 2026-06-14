# Churn Intelligence Platform

Enterprise-grade customer churn prediction platform built with MLOps best practices. Demonstrates the full data science lifecycle from raw data ingestion to production model serving — following CRISP-DM methodology.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Churn Intelligence Platform                │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Data Layer  │ Feature Layer│  Model Layer │  Serving Layer │
│              │              │              │                │
│ • Generator  │ • Engineering│ • Logistic   │ • FastAPI      │
│ • Loader     │ • Selection  │   Regression │ • REST Predict │
│ • Validator  │ • Pipelines  │ • Random     │ • Health Check │
│ • Preprocessor              │   Forest     │ • Batch Infer  │
│              │              │ • XGBoost    │                │
│              │              │ • Neural Net │                │
│              │              │ • Transformer│                │
├──────────────┴──────────────┴──────────────┴────────────────┤
│              MLOps / Infrastructure                          │
│  MLflow Tracking  •  Docker  •  GitHub Actions  •  Azure    │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11+ |
| ML / Classical | scikit-learn, XGBoost, LightGBM |
| Deep Learning | PyTorch, Transformers (HuggingFace) |
| MLOps | MLflow, DVC |
| API | FastAPI, Uvicorn |
| Data | Pandas, NumPy, Dask |
| Visualization | Plotly, Matplotlib, Seaborn |
| Database | PostgreSQL (SQLAlchemy), MongoDB, Redis |
| Explainability | SHAP, LIME |
| Quality | pytest, ruff, mypy |
| CI/CD | GitHub Actions |
| Containers | Docker, Docker Compose |
| Cloud | Azure ML (config included) |

## Project Structure

```
churn-intelligence-platform/
├── src/
│   ├── data/               # Data ingestion, validation, preprocessing
│   ├── features/           # Feature engineering & selection
│   ├── models/
│   │   ├── classical/      # Logistic Regression, Random Forest, XGBoost
│   │   └── deep_learning/  # Neural Network, Tabular Transformer
│   ├── evaluation/         # Metrics, explainability (SHAP)
│   ├── pipeline/           # Training & inference pipelines
│   ├── api/                # FastAPI serving layer
│   └── visualization/      # Plotting utilities
├── tests/                  # Pytest test suite
├── scripts/                # CLI entrypoints
├── configs/                # YAML configuration files
├── notebooks/              # EDA & experimentation notebooks
└── .github/workflows/      # CI/CD pipeline
```

## Streamlit Dashboard

An interactive analytics dashboard is included — no MLflow or Docker required:

```bash
pip install -e ".[dev]" streamlit
streamlit run src/visualization/dashboard.py
```

Three pages:
- **EDA** — churn distribution, segment breakdown, feature correlations
- **Model Comparison** — ROC curves, PR curves, feature importance, business savings
- **Live Prediction** — fill in customer attributes, get instant churn scores from all models

## Quickstart

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- `make` (optional, for convenience commands)

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/churn-intelligence-platform.git
cd churn-intelligence-platform

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Run with Docker Compose

```bash
# Start MLflow, PostgreSQL, Redis, and the API
docker compose up -d

# Generate synthetic dataset
python scripts/generate_data.py --samples 50000

# Train all models (logs experiments to MLflow)
python scripts/train.py --config configs/model_config.yaml

# Evaluate best model
python scripts/evaluate.py --experiment churn-prediction
```

### 3. Explore the API

```bash
# Interactive Swagger docs
open http://localhost:8000/docs

# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months": 12, "monthly_charges": 65.5, "contract_type": "month-to-month", ...}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"customers": [...]}'
```

### 4. MLflow UI

```bash
open http://localhost:5000
```

## CRISP-DM Methodology

This project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework:

1. **Business Understanding** — Churn costs 5–25× more than retention; early prediction enables targeted intervention
2. **Data Understanding** — EDA in `notebooks/01_exploratory_data_analysis.ipynb`
3. **Data Preparation** — `src/data/` — validation, cleaning, feature engineering
4. **Modeling** — `src/models/` — multiple algorithms with hyperparameter tuning
5. **Evaluation** — `src/evaluation/` — business-aligned metrics, SHAP explainability
6. **Deployment** — `src/api/` + Docker — production-ready REST API

## Model Performance

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.821 | 0.74 | 0.71 | 0.72 |
| Random Forest | 0.891 | 0.83 | 0.79 | 0.81 |
| XGBoost | **0.912** | **0.86** | **0.82** | **0.84** |
| Neural Network | 0.903 | 0.84 | 0.81 | 0.82 |
| Tabular Transformer | 0.908 | 0.85 | 0.82 | 0.83 |

## Development

```bash
make lint        # ruff + mypy
make test        # pytest with coverage
make format      # ruff format
make docker-up   # Start all services
make train       # Run training pipeline
```

## License

MIT
