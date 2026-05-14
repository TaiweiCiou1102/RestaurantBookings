"""Azure ML scoring script for the XGBoost reservation-hour classifier.

This module provides the required `init()` and `run()` entry points for
Azure Machine Learning online endpoints. It loads an XGBoost model and 
associated Scikit-Learn encoders to predict reservation time slots.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier

# ─── Module-Level Constants ───────────────────────────────────────────────────

# Fallback to current directory if AZUREML_MODEL_DIR is not set (e.g., local testing).
_MODEL_DIR = Path(os.getenv("AZUREML_MODEL_DIR", Path(__file__).parent.resolve()))

_MODEL_FILENAME = "model.ubj"
_X_ENCODER_FILENAME = "encoders.joblib"
_Y_ENCODER_FILENAME = "target_encoder.joblib"

# Must match exactly what was used during training.
_CATEGORICAL_COLS = frozenset([
    "dining_purpose",
    "booked_by_gender",
    "account_gender",
    "city_code",
    "city_area_code",
    "member_city_code",
])

# Columns to safely ignore if present in the inference payload.
_COLS_TO_DROP = frozenset([
    "has_google_id",
    "has_yahoo_id",
    "has_weibo_id",
])

# ─── Global State ─────────────────────────────────────────────────────────────

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# Global variables required by Azure ML's init/run architecture.
_MODEL: Optional[XGBClassifier] = None
_X_ENCODER: Optional[Dict[str, Any]] = None
_Y_ENCODER: Optional[LabelEncoder] = None


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _parse_input_json(raw_data: str) -> pd.DataFrame:
    """Parses the Azure ML input JSON string into a Pandas DataFrame.

    Args:
        raw_data: A JSON string containing the request payload.

    Returns:
        A pandas DataFrame constructed from the parsed JSON.

    Raises:
        KeyError: If 'input_data', 'columns', or 'data' are missing.
        json.JSONDecodeError: If the input is not valid JSON.
    """
    body = json.loads(raw_data)["input_data"]
    return pd.DataFrame(
        data=body["data"],
        columns=body["columns"],
        index=body.get("index")
    )


def _convert_class_to_time_range(class_label: int) -> str:
    """Converts a class label to a human-readable time range string.

    Args:
        class_label: The integer class label from the model prediction.

    Returns:
        A string representing the time range.
        - class < 24  -> hour granularity (e.g., 19 -> "19:00-19:59")
        - class >= 24 -> half-hour slot   (e.g., 39 -> "19:30-19:59")
    """
    if class_label >= 24:
        hour, half = divmod(class_label, 2)
        minute = half * 30
        return f"{hour:02d}:{minute:02d}-{hour:02d}:{minute + 29:02d}"
    return f"{class_label:02d}:00-{class_label:02d}:59"


def _build_output_json(proba_df: pd.DataFrame) -> str:
    """Serializes the probability DataFrame to an AML-compliant JSON string.

    Args:
        proba_df: DataFrame containing prediction probabilities where 
            columns are class labels and rows are instances.

    Returns:
        A JSON string formatted for Azure ML endpoint response.
    """
    time_columns = [_convert_class_to_time_range(int(c)) for c in proba_df.columns]
    payload = {
        "predictions": {
            "columns": time_columns,
            "index": proba_df.index.tolist(),
            "data": [[round(val, 6) for val in row] for row in proba_df.values.tolist()],
        }
    }
    return json.dumps(payload, ensure_ascii=False)


def _encode_features(features_df: pd.DataFrame, encoders: Dict[str, Any]) -> pd.DataFrame:
    """Applies fitted encoders to categorical columns.

    Args:
        features_df: The input features DataFrame.
        encoders: A dictionary mapping column names to fitted scikit-learn encoders.

    Returns:
        A new DataFrame with categorical columns encoded. Unseen categories 
        are mapped to -1 or appropriately handled.
    """
    encoded_df = features_df.copy()
    
    for col_name, encoder in encoders.items():
        if col_name not in encoded_df.columns:
            continue
            
        # Ensure column is treated as string and handle missing values
        encoded_df[col_name] = encoded_df[col_name].fillna("Unknown").astype(str)
        
        if isinstance(encoder, OrdinalEncoder):
            # Flatten the 2D array returned by OrdinalEncoder to a 1D Series
            encoded_df[col_name] = encoder.transform(encoded_df[[col_name]])[:, 0]
        else:
            # Fallback for LabelEncoder: map unseen labels to -1
            known_classes = set(encoder.classes_)
            encoded_df[col_name] = encoded_df[col_name].apply(
                lambda val: int(encoder.transform([val])[0]) if val in known_classes else -1
            )
            
    return encoded_df


# ─── Azure ML Entry Points ────────────────────────────────────────────────────

def init() -> None:
    """Loads the model and encoders once at container startup.

    This function initializes the global state required for inference.
    
    Raises:
        FileNotFoundError: If any of the required artifact files are missing.
        RuntimeError: If model loading fails.
    """
    global _MODEL, _X_ENCODER, _Y_ENCODER

    model_path = _MODEL_DIR / _MODEL_FILENAME
    x_encoder_path = _MODEL_DIR / _X_ENCODER_FILENAME
    y_encoder_path = _MODEL_DIR / _Y_ENCODER_FILENAME

    try:
        _logger.info("Loading model from %s", model_path)
        _MODEL = XGBClassifier()
        _MODEL.load_model(model_path)

        _logger.info("Loading feature encoder from %s", x_encoder_path)
        _X_ENCODER = joblib.load(x_encoder_path)

        _logger.info("Loading target encoder from %s", y_encoder_path)
        _Y_ENCODER = joblib.load(y_encoder_path)

    except Exception as e:
        _logger.exception("Failed to initialize model or encoders.")
        raise RuntimeError(f"Initialization failed: {e}") from e

    _logger.info("Initialization complete successfully.")


def run(raw_data: str) -> str:
    """Processes a single inference request.

    Args:
        raw_data: A JSON string conforming to the AML expected input schema.

    Returns:
        A JSON string containing the predicted probabilities.

    Raises:
        RuntimeError: If global dependencies are not initialized, or if 
            the model's output shape mismatches the target encoder's classes.
    """
    if _MODEL is None or _X_ENCODER is None or _Y_ENCODER is None:
        raise RuntimeError("Global model or encoders are not initialized. Check init() logs.")

    # 1. Parse and clean input data
    df = _parse_input_json(raw_data)
    
    columns_to_drop = [col for col in _COLS_TO_DROP if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    # 2. Encode features
    x_features = _encode_features(df, _X_ENCODER)

    # 3. Drop unexpected object-dtype columns to prevent XGBoost errors
    object_cols = x_features.select_dtypes(include="object").columns.tolist()
    if object_cols:
        _logger.warning("Dropping unexpected object-dtype columns: %s", object_cols)
        x_features = x_features.drop(columns=object_cols)

    # 4. Predict probabilities
    probabilities = _MODEL.predict_proba(x_features)
    num_predicted_classes = probabilities.shape[1]

    # 5. Validate alignment between Model and Target Encoder
    if hasattr(_MODEL, "classes_") and len(_MODEL.classes_) == num_predicted_classes:
        hour_labels = [int(h) for h in _MODEL.classes_]
    elif hasattr(_Y_ENCODER, "classes_") and len(_Y_ENCODER.classes_) == num_predicted_classes:
        hour_labels = [int(h) for h in _Y_ENCODER.classes_]
    else:
        # FAIL-FAST: Instead of silently passing bad data, we raise a clear error.
        # This prevents downstream systems from processing mismatched schema data.
        encoder_len = len(_Y_ENCODER.classes_) if hasattr(_Y_ENCODER, "classes_") else "Unknown"
        error_msg = (
            f"Artifact mismatch error! The XGBoost model predicted {num_predicted_classes} classes, "
            f"but the target encoder (_Y_ENCODER) contains {encoder_len} classes. "
            "Please verify that the model and encoders were generated from the same training run."
        )
        _logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 6. Format output
    proba_df = pd.DataFrame(probabilities, columns=hour_labels, index=df.index)
    return _build_output_json(proba_df)