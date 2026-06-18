# RestaurantBookings

## Project Architecture

```
configs/
  etl.yaml                   # ETL data source (Kaggle) + per-stage directories
  raw_data_schema.yaml       # Raw data schema
  regression.yaml            # Model hyperparameter config
  xgboost.yaml
data/
  raw/                       # Original datasets
  interim/                   # Cleaned & joined intermediate data
  processed/
    features_ready.csv       # All models share this as starting point
src/
  etl/                       # Domain knowledge feature engineering (run in numbered order)
    step0_run_download.py      # Download raw data from Kaggle
    step1_run_cleaning.py
    step2_run_integration.py
    step3_run_features.py
    _config.py                 # Loads configs/etl.yaml (paths resolved to project root)
    _utils.py                  # Shared helpers (cleaning / integration / features)
  preprocessing/             # Model-specific encoding & scaling
    common.py                # Shared: train/test split, missing values, scaling
    regression.py            # One-hot encoding, drop baseline, standardize
    tree.py                  # Label encoding or keep integer, no scaling
  models/                    # Training, evaluation & MLflow logging
    train_regression.py
    train_xgboost.py
notebooks/                   # Exploratory analysis & experiments
mlruns/                      # MLflow tracking data (auto-generated)
```

## Data Flow

```
Kaggle → etl/step0..3 → features_ready.csv → preprocessing/{regression,tree}.py → models/train_*.py → MLflow
```

## Core Principles

- **ETL**: domain knowledge features only, no model-specific encoding
- **Preprocessing**: model-specific transforms (one-hot for regression, label/int for tree-based)
- **Training**: each model script handles its own preprocessing call and MLflow logging
- Run ETL in order: `step0_run_download` → `step1_run_cleaning` → `step2_run_integration` → `step3_run_features` (e.g. `uv run python -m src.etl.step3_run_features`)
