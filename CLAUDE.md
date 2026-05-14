# RestaurantBookings

## Project Architecture

```
configs/
  raw_data_schema.yaml       # Raw data schema
  regression.yaml            # Model hyperparameter config
  xgboost.yaml
data/
  raw/                       # Original datasets
  interim/                   # Cleaned & joined intermediate data
  processed/
    features_ready.csv       # All models share this as starting point
src/
  etl/                       # Domain knowledge feature engineering
    run_cleaning.py
    run_integration.py
    run_features.py
    _cleaning_utils.py
    _integration_utils.py
    _features_utils.py
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
features_ready.csv → preprocessing/{regression,tree}.py → models/train_*.py → MLflow
```

## Core Principles

- **ETL**: domain knowledge features only, no model-specific encoding
- **Preprocessing**: model-specific transforms (one-hot for regression, label/int for tree-based)
- **Training**: each model script handles its own preprocessing call and MLflow logging
- Run ETL with: `uv run python -m src.etl.run_features`
