"""Azure ML scoring script for the three-subset XGBoost reservation model.

The productionized model is **three sub-models** (lunch / afternoon / dinner),
each predicting a half-hour reservation slot within its meal period. This script
loads all three from a single registered model folder and routes each input row
to the right sub-model by ``reservation_hour``.

Azure ML entry points:
    init()  — load every sub-model + encoders once at container startup
    run()   — route rows, predict per subset, merge, return JSON

The model folder (``AZUREML_MODEL_DIR``) is produced by ``export_bundle.py`` and
has this layout::

    model_bundle/
    ├── lunch/      { model.ubj, encoders.joblib, target_encoder.joblib }
    ├── afternoon/  { ... }
    ├── dinner/     { ... }
    └── routing.json

Input JSON (Azure ML standard format)::

    {
      "input_data": {
        "columns": ["reservation_hour", "party_size", "dining_purpose", ...],
        "index":   [0, 1],
        "data":    [[19, 2, "生日慶祝", ...], ...]
      }
    }

``reservation_hour`` is required for routing (which meal period). It is used only
to pick the sub-model and is dropped before prediction (it is a leakage feature).

Output JSON::

    {
      "predictions": {
        "columns": ["reservation_half_hour", "time_range", "subset", "probability"],
        "index":   [0, 1],
        "data":    [[39, "19:30-19:59", "dinner", 0.42], ...]
      }
    }
Rows whose ``reservation_hour`` falls outside every meal period get a null
prediction with subset ``"_unmatched"``.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier

# ─── Paths ─────────────────────────────────────────────────────────────────
# AZUREML_MODEL_DIR points at the registered model folder in the container.
# Fall back to the local bundle for offline testing.
_MODEL_ROOT = Path(os.getenv("AZUREML_MODEL_DIR", Path(__file__).parent.resolve()))

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# ─── Global state (populated by init) ───────────────────────────────────────
_ROUTING: Optional[dict] = None
# {subset_key: (model, encoders, target_encoder)}
_SUBSETS: Dict[str, Tuple[XGBClassifier, Dict[str, Any], LabelEncoder]] = {}


# ─── Helpers ────────────────────────────────────────────────────────────────

def _find_bundle_dir(root: Path) -> Path:
    """Locate the folder containing routing.json under *root* (handles the extra
    nesting Azure ML adds, e.g. AZUREML_MODEL_DIR/<model_name>/model_bundle)."""
    if (root / "routing.json").exists():
        return root
    matches = list(root.rglob("routing.json"))
    if not matches:
        raise FileNotFoundError(f"routing.json not found under {root}")
    return matches[0].parent


def _clean_str(series: pd.Series) -> pd.Series:
    """Normalise to clean strings, filling missing with 'Unknown' (matches training)."""
    return series.astype("object").fillna("Unknown").astype(str)


def _apply_saved_encoders(df: pd.DataFrame, encoders: Dict[str, Any]) -> pd.DataFrame:
    """Transform categorical columns with the artifacts saved at training time.

    Supports CategoricalDtype (native categorical, current), OrdinalEncoder and
    LabelEncoder (legacy). Mirrors src/xgboost/preprocessing.apply_saved_encoders.
    """
    out = df.copy()
    for col, enc in encoders.items():
        if col not in out.columns:
            continue
        out[col] = _clean_str(out[col])
        if isinstance(enc, pd.CategoricalDtype):
            out[col] = out[col].astype(enc)
        elif isinstance(enc, OrdinalEncoder):
            out[col] = enc.transform(out[[col]])[:, 0]
        else:  # LabelEncoder: unseen -> -1
            known = set(enc.classes_)
            out[col] = out[col].apply(lambda v: int(enc.transform([v])[0]) if v in known else -1)
    return out


def _slot_to_time_range(slot: int) -> str:
    """Half-hour slot index -> readable range, e.g. 39 -> '19:30-19:59'."""
    hour, half = divmod(int(slot), 2)
    minute = half * 30
    return f"{hour:02d}:{minute:02d}-{hour:02d}:{minute + 29:02d}"


def _parse_input(raw: str) -> pd.DataFrame:
    body = json.loads(raw)["input_data"]
    return pd.DataFrame(data=body["data"], columns=body["columns"], index=body.get("index"))


# ─── Azure ML entry points ──────────────────────────────────────────────────

def init() -> None:
    """Load routing metadata and every sub-model once at startup."""
    global _ROUTING, _SUBSETS

    bundle = _find_bundle_dir(_MODEL_ROOT)
    _logger.info("Loading model bundle from %s", bundle)

    with open(bundle / "routing.json", encoding="utf-8") as f:
        _ROUTING = json.load(f)

    _SUBSETS = {}
    for key in _ROUTING["subsets"]:
        sub_dir = bundle / key
        model = XGBClassifier(enable_categorical=True)
        model.load_model(sub_dir / "model.ubj")
        encoders = joblib.load(sub_dir / "encoders.joblib")
        target_enc = joblib.load(sub_dir / "target_encoder.joblib")
        _SUBSETS[key] = (model, encoders, target_enc)
        _logger.info("Loaded sub-model '%s'", key)

    _logger.info("Initialization complete (%d sub-models).", len(_SUBSETS))


def _predict_subset(rows: pd.DataFrame, key: str) -> pd.DataFrame:
    """Predict slot + confidence for the rows routed to one sub-model."""
    model, encoders, target_enc = _SUBSETS[key]
    drop_cols = (
        _ROUTING["non_feature_cols"] + _ROUTING["leakage"] + [_ROUTING["target"]]
    )
    X = rows.drop(columns=[c for c in drop_cols if c in rows.columns], errors="ignore")
    X = _apply_saved_encoders(X, encoders)
    # Align to the exact training feature order; unseen/missing cols -> NaN (missing).
    if _ROUTING.get("feature_columns"):
        X = X.reindex(columns=_ROUTING["feature_columns"])

    proba = model.predict_proba(X)
    enc_pred = proba.argmax(axis=1)
    slots = target_enc.inverse_transform(enc_pred)
    conf = proba.max(axis=1)

    return pd.DataFrame(
        {
            "reservation_half_hour": [int(s) for s in slots],
            "time_range": [_slot_to_time_range(s) for s in slots],
            "subset": key,
            "probability": [round(float(c), 6) for c in conf],
        },
        index=rows.index,
    )


def run(raw_data: str) -> str:
    """Route each row to its meal-period sub-model, predict, and merge results."""
    if not _SUBSETS or _ROUTING is None:
        raise RuntimeError("Model not initialized. Check init() logs.")

    df = _parse_input(raw_data)

    route_col = _ROUTING["routing_column"]
    if route_col not in df.columns:
        raise ValueError(
            f"Input must include '{route_col}' to route rows to the correct "
            f"meal-period sub-model."
        )

    hours = pd.to_numeric(df[route_col], errors="coerce")
    parts = []
    matched = pd.Series(False, index=df.index)

    for key, meta in _ROUTING["subsets"].items():
        mask = hours.between(meta["hour_min"], meta["hour_max"])
        if mask.any():
            parts.append(_predict_subset(df.loc[mask], key))
            matched |= mask

    # Rows that matched no meal period -> null prediction.
    if (~matched).any():
        unmatched = df.loc[~matched]
        _logger.warning("%d row(s) outside all meal periods -> unmatched", len(unmatched))
        parts.append(
            pd.DataFrame(
                {
                    "reservation_half_hour": None,
                    "time_range": None,
                    "subset": "_unmatched",
                    "probability": None,
                },
                index=unmatched.index,
            )
        )

    result = pd.concat(parts).reindex(df.index)

    payload = {
        "predictions": {
            "columns": ["reservation_half_hour", "time_range", "subset", "probability"],
            "index": result.index.tolist(),
            "data": result.where(pd.notnull(result), None).values.tolist(),
        }
    }
    return json.dumps(payload, ensure_ascii=False)
