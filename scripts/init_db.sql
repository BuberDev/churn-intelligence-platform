-- Initialize databases for Docker Compose setup
CREATE DATABASE IF NOT EXISTS mlflow_db;

-- Predictions table for audit log
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     VARCHAR(20),
    model_name      VARCHAR(100) NOT NULL,
    model_version   VARCHAR(20),
    churn_probability FLOAT NOT NULL,
    predicted_churned BOOLEAN NOT NULL,
    confidence      VARCHAR(10),
    requested_at    TIMESTAMPTZ DEFAULT NOW(),
    features        JSONB
);

CREATE INDEX IF NOT EXISTS idx_predictions_customer ON predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_requested_at ON predictions(requested_at);

-- Model registry (simplified — MLflow handles the full registry)
CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    version         VARCHAR(20) NOT NULL,
    mlflow_run_id   VARCHAR(50),
    roc_auc         FLOAT,
    registered_at   TIMESTAMPTZ DEFAULT NOW(),
    is_production   BOOLEAN DEFAULT FALSE,
    UNIQUE(name, version)
);
