"""Inference script for XGBoost classifier (JSON I/O).

Prediction granularity is fixed at **half-hour** (e.g. 11, 11.5, 12, 12.5 …).

Usage:
    uv run python -m src.xgboost.inference --input request.json
    uv run python -m src.xgboost.inference --input request.json --output predictions.json

Input JSON format:
    {
      "input_data": {
        "columns": [0, 1, 2, ...],       # numeric indices or column name strings
        "index":   [0, 1],
        "data":    [[...], [...]]
      }
    }

Output JSON format:
    {
      "predictions": {
        "columns": ["prediction"],
        "index":   [0, 1],
        "data":    [[12.0], [18.5]]
      }
    }
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from src.common.model_utils import find_nested_runs, init_tracking, load_config, resolve_granularity
from src.xgboost.preprocessing import NON_FEATURE_COLS, fill_missing, apply_saved_encoders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/xgboost.yaml"

# Column order matching features_ready.csv (minus NON_FEATURE_COLS).
# Used to map numeric indices → column names when JSON columns are integers.
FEATURE_COLUMNS = [
    "party_size", "dining_purpose", "booked_by_gender", "prepay_satisfied",
    "hotel_restaurant", "family_friendly",
    "accepts_credit_card", "has_parking", "outdoor_seating", "has_wifi",
    "wheelchair_accessible", "lat", "lng", "is_vip_member", "account_gender",
    "lead_time", "age", "booking_hour",
    "reservation_hour", "reservation_seconds", "reservation_half_hour",
    "weekday", "avg_price", "is_holiday_vicinity", "days_from_payday",
    "popular_hour", "dinner_ratio", "restaurant_density",
]


# ------------------------------------------------------------------
#  JSON I/O
# ------------------------------------------------------------------

def parse_json_input(path: str) -> pd.DataFrame:
    """Parse input JSON into a DataFrame.

    Accepts ``columns`` as either string names or integer indices.
    Integer indices are mapped to FEATURE_COLUMNS.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    body = payload["input_data"]
    columns = body["columns"]
    index = body.get("index")
    data = body["data"]

    # Map numeric indices to column names
    if columns and isinstance(columns[0], int):
        columns = [FEATURE_COLUMNS[i] for i in columns]

    return pd.DataFrame(data=data, columns=columns, index=index)


def build_json_output(predictions: pd.Series) -> dict:
    """Build output JSON from a prediction Series."""
    return {
        "predictions": {
            "columns": ["prediction"],
            "index": predictions.index.tolist(),
            "data": [[v] for v in predictions.tolist()],
        }
    }


# ------------------------------------------------------------------
#  Routing: split rows into subsets by reservation_hour
# ------------------------------------------------------------------

def route_to_subsets(
    df: pd.DataFrame,
    subsets: Dict[str, dict],
) -> Dict[str, pd.DataFrame]:
    """Split *df* into subsets based on reservation_hour ranges.

    Args:
        df: Full input DataFrame (must contain ``reservation_hour``).
        subsets: Subset definitions from config, e.g.
            ``{"午餐 Lunch (11-14h)": {"hour_min": 11, "hour_max": 14}, ...}``

    Returns:
        Dict mapping subset name → filtered DataFrame (preserving original index).
        Rows that fall outside all defined ranges are collected under a
        special ``"_unmatched"`` key (empty DataFrame if all rows matched).
    """
    matched_indices: list[int] = []
    result: Dict[str, pd.DataFrame] = {}

    for name, rng in subsets.items():
        mask = df["reservation_hour"].between(rng["hour_min"], rng["hour_max"])
        subset_df = df.loc[mask]
        result[name] = subset_df
        matched_indices.extend(subset_df.index.tolist())
        logger.info("Route '%s': %d rows", name, len(subset_df))

    unmatched = df.loc[~df.index.isin(matched_indices)]
    if len(unmatched) > 0:
        logger.warning("%d rows did not match any subset — skipped", len(unmatched))
        result["_unmatched"] = unmatched

    return result


# ------------------------------------------------------------------
#  Single-subset inference (knows nothing about routing)
# ------------------------------------------------------------------

def load_subset_artifacts(
    run_id: str,
) -> Tuple[object, Dict[str, OrdinalEncoder], LabelEncoder]:
    """Load model, feature encoders, and target encoder from an MLflow run.

    Returns:
        (xgb_model, feature_encoders, target_encoder)
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    artifact_dir = client.download_artifacts(run_id, "")
    artifact_path = Path(artifact_dir)

    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

    encoders: Dict[str, OrdinalEncoder] = joblib.load(artifact_path / "encoders.joblib")
    target_enc: LabelEncoder = joblib.load(artifact_path / "target_encoder.joblib")

    return model, encoders, target_enc


def predict_subset(
    df: pd.DataFrame,
    run_id: str,
    target: str,
    leakage: List[str],
) -> pd.Series:
    """Run inference on a single subset DataFrame.

    Args:
        df: Feature DataFrame for one subset (already routed).
        run_id: MLflow run ID of the nested (subset) run.
        target: Target column name (dropped if present).
        leakage: Leakage columns to drop.

    Returns:
        pd.Series of predicted labels (original scale), indexed like *df*.
    """
    model, encoders, target_enc = load_subset_artifacts(run_id)

    # Drop columns that the model should not see
    drop_cols = [c for c in NON_FEATURE_COLS + leakage + [target] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Apply saved encoders (transform only, no fitting)
    X_proc = apply_saved_encoders(X, encoders)

    # Predict and map back to original labels
    y_enc = model.predict(X_proc)
    y_labels = target_enc.inverse_transform(y_enc)

    return pd.Series(y_labels, index=df.index, name="prediction")


# ------------------------------------------------------------------
#  Orchestrator
# ------------------------------------------------------------------

GRANULARITY = "half_hour"


def run_inference(
    df: pd.DataFrame,
    config_path: str,
    parent_run_id: str | None = None,
) -> pd.Series:
    """End-to-end inference: route → predict per subset → merge.

    Granularity is fixed at half-hour.

    Returns:
        Series of predictions, indexed like *df*.
        Rows that cannot be routed will have NaN.
    """
    init_tracking()  # metadata lives in sqlite:///mlflow.db, not the ./mlruns file store
    config = load_config(config_path)
    target, leakage = resolve_granularity(config, GRANULARITY)

    df = fill_missing(df, strategy="median")

    # --- Discover subset models --------------------------------------------
    nested_runs = find_nested_runs(
        "xgboost", parent_run_id,
        parent_run_name_contains=f"xgb_{GRANULARITY}",
    )

    # --- Route -------------------------------------------------------------
    routed = route_to_subsets(df, config["subsets"])

    # --- Predict per subset ------------------------------------------------
    predictions: list[pd.Series] = []

    for subset_name, subset_df in routed.items():
        if subset_name == "_unmatched" or len(subset_df) == 0:
            continue

        if subset_name not in nested_runs:
            logger.warning(
                "No model found for subset '%s' — %d rows skipped",
                subset_name, len(subset_df),
            )
            continue

        run_id = nested_runs[subset_name]
        logger.info("Predicting '%s' (%d rows) with run %s", subset_name, len(subset_df), run_id)
        preds = predict_subset(subset_df, run_id, target, leakage)
        predictions.append(preds)

    # --- Merge results -----------------------------------------------------
    all_preds = pd.concat(predictions).sort_index() if predictions else pd.Series(dtype=float)

    n_missing = len(df) - len(all_preds)
    if n_missing > 0:
        logger.warning("%d rows have no prediction (unmatched or missing model)", n_missing)

    return all_preds


# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost inference (三分法, half-hour granularity)")
    parser.add_argument(
        "--input", required=True,
        help="Path to input JSON file.",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="MLflow parent run ID. If omitted, uses the latest finished run.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save output JSON. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    # --- Input -------------------------------------------------------------
    df = parse_json_input(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    # --- Inference ---------------------------------------------------------
    predictions = run_inference(
        df,
        config_path=args.config,
        parent_run_id=args.run_id,
    )

    # --- Output ------------------------------------------------------------
    output = build_json_output(predictions)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("Saved predictions to %s", args.output)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
