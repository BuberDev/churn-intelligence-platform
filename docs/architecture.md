# Architecture Decision Records

## ADR-001: CRISP-DM as Project Framework

**Status:** Accepted

**Context:** Need a structured methodology to organise data science work across multiple stakeholders (data engineers, data scientists, business analysts).

**Decision:** Adopt CRISP-DM (Cross-Industry Standard Process for Data Mining). Project structure maps directly to CRISP-DM phases: `src/data/` → Data Understanding & Preparation, `src/models/` → Modelling, `src/evaluation/` → Evaluation, `src/api/` → Deployment.

**Consequences:** Easier communication with non-technical stakeholders; each folder has clear scope.

---

## ADR-002: Abstract BaseChurnModel

**Status:** Accepted

**Context:** Five model types (LR, RF, XGBoost, NN, Transformer) must be interchangeable at the pipeline and API layer.

**Decision:** Define `BaseChurnModel` ABC with `fit`, `predict_proba`, and `metadata` as abstract methods. `predict` and `predict_with_confidence` are provided by the base class so all models behave identically downstream.

**Consequences:** Adding a new model requires implementing 3 methods; the pipeline, evaluator, and API need zero changes.

---

## ADR-003: Redis Prediction Cache

**Status:** Accepted

**Context:** Real-time API calls for the same customer during nightly batch scoring create redundant compute.

**Decision:** Cache prediction results in Redis keyed by SHA256 of the feature dict (16-char prefix). TTL = 1 hour. Cache is optional — if Redis is unavailable the pipeline degrades gracefully.

**Consequences:** Reduced latency for repeated calls; minimal infrastructure overhead.

---

## ADR-004: RobustScaler over StandardScaler

**Status:** Accepted

**Context:** `monthly_charges` and `total_charges` have long right tails (outlier enterprise customers).

**Decision:** Use `RobustScaler` (IQR-based) by default. The `scaling` parameter on `DataPreprocessor` allows switching to `StandardScaler` for ablation experiments.

**Consequences:** Model stability on production data with unusual customer values; slightly less interpretable scale.

---

## ADR-005: Optuna for Hyperparameter Tuning

**Status:** Accepted

**Context:** Grid search is too slow for XGBoost/RF with 6+ hyperparameters. Bayesian search converges faster.

**Decision:** Provide optional `tune()` classmethods on `RandomForestModel` and `XGBoostModel` using Optuna's TPE sampler. Disabled by default in CI to keep test runtime low.

**Consequences:** Best models found in ~50 trials vs. hundreds for grid search; optional dependency (Optuna not required for basic usage).
