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
  common/                    # Shared modeling infra (used by exploration & xgboost)
    model_utils.py           # MLflow logging, CV, Optuna, plotting, run discovery
  exploration/               # FROZEN: early multi-model comparison study (see its README)
    preprocessing/           # per-model encoding (common / regression / tree)
    train_regression.py      # superseded by src/xgboost/ — do not extend
    train_random_forest.py
    train_xgboost.py
    inference.py
  xgboost/                   # SOURCE OF TRUTH: the chosen, productionized model
    preprocessing.py         # self-contained encoding & scaling
    train.py                 # training + MLflow logging
    inference.py             # batch / request inference
notebooks/                   # Exploratory analysis & experiments
mlruns/                      # MLflow tracking data (auto-generated)
```

## Data Flow

```
Kaggle → etl/step0..3 → features_ready.csv → xgboost/{preprocessing,train}.py → MLflow
```
(`src/exploration/` is the frozen comparison study; `src/xgboost/` is the live model.)

## Core Principles

- **ETL**: domain knowledge features only, no model-specific encoding
- **Preprocessing**: model-specific transforms, owned by each model package
- **Training**: each model script handles its own preprocessing call and MLflow logging
- **xgboost is source of truth**; `exploration/` is frozen — extend the model in `src/xgboost/`
- Shared modeling helpers live in `src/common/model_utils.py`
- Run ETL in order: `step0_run_download` → `step1_run_cleaning` → `step2_run_integration` → `step3_run_features` (e.g. `uv run python -m src.etl.step3_run_features`)
