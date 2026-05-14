import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import false, true
from xgboost import XGBClassifier

from src.models._utils import (
    build_optuna_objective,
    load_config,
    log_evaluation_artifacts,
    resolve_granularity,
    run_cv,
    save_pipeline_artifacts,
    tolerance_accuracy,
)
from src.preprocessing.common import fill_missing, load_features, split_data
from src.preprocessing.tree import preprocess_for_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "feature_engineering":{
        "external_cols":{}, #從外部資料源整合加入的欄位
        "engineered_cols":{}, # 從原始欄位與外部欄位衍生的特徵欄位
        "target_cols":{} # 目標欄位
    },
    "model_params":{
        "target": {
            "external":""
        },
        "features":{
            "original_cols": {
                "original":[],
                "external":[]
            },
            "engineered_cols":[],
        }
    },
    "estimator_params":{
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        #early_stopping_rounds: 30
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "mlogloss" # multi-class log loss for classification, rmse for regression
    },
    "evaluation_params":{
        "test_size": 0.2,
        "split_strategy": "random", # random | temporal (random is default for tabular data, temporal is for time series)
        "cv_folds": 5, # 0 = holdout only
        "log_confusion_matrix": True,
        "log_feature_importance": True,
        "tolerance_slots": 1    
    },
    "tuning_params":{
        "activate":false,
        "n_trials": 50,
        "search_space":{
            "n_estimators": { "type": "categorical", "choices": [100, 300, 500, 1000] },
            "max_depth": { "type": "int", "low": 3, "high": 12 },
            "learning_rate": { "type": "float", "low": 0.01, "high": 0.3, "log": True },
            "min_child_weight": { "type": "int", "low": 1, "high": 10 }
        }
    }
}

def half_hour_slot_class(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """convert datatime to class label for half-hour slot prediction."""
    df[column] = df[column].dt.hour * 2 + (df[column].dt.minute >= 30).astype(int)
    return df

def data_extract(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    """Extract target and features from the dataframe."""
    return df[[target] + features]

def data_preprocess(df: pd.DataFrame, target: str, features: list)