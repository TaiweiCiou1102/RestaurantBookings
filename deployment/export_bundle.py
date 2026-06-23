"""Export the trained three-subset XGBoost model into a self-contained bundle.

Pulls the latest `xgb_half_hour` parent run's three subset models (lunch /
afternoon / dinner) from MLflow and writes them — together with their feature
encoders, target encoders, and routing metadata — into ``deployment/model_bundle/``
as plain files. The Azure ML container has no MLflow/sqlite, so the scoring
script reads only from this folder.

Run from the project root:
    uv run python deployment/export_bundle.py

Output layout:
    deployment/model_bundle/
    ├── lunch/      { model.ubj, encoders.joblib, target_encoder.joblib }
    ├── afternoon/  { ... }
    ├── dinner/     { ... }
    └── routing.json
"""

import json
import logging
import shutil
import sys
from pathlib import Path

import joblib

# Allow `import src.*` when run as a script (sys.path[0] would be deployment/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.model_utils import (
    find_nested_runs,
    init_tracking,
    load_config,
    resolve_granularity,
)
from src.xgboost.inference import load_subset_artifacts
from src.xgboost.preprocessing import CATEGORICAL_COLS, NON_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/xgboost.yaml"
GRANULARITY = "half_hour"
BUNDLE_DIR = Path("deployment/model_bundle")


def _subset_key(display_name: str) -> str:
    """Map a config subset display name to a stable English folder key."""
    low = display_name.lower()
    for key in ("lunch", "afternoon", "dinner"):
        if key in low:
            return key
    raise ValueError(f"Cannot derive a subset key from '{display_name}'")


def main() -> None:
    init_tracking()
    config = load_config(CONFIG_PATH)
    target, leakage = resolve_granularity(config, GRANULARITY)
    subsets_cfg = config["subsets"]

    subset_runs = find_nested_runs("xgboost", parent_run_name_contains=f"xgb_{GRANULARITY}")

    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True)

    routing: dict = {
        "target": target,
        "leakage": leakage,
        "non_feature_cols": NON_FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "routing_column": "reservation_hour",
        "subsets": {},
    }
    feature_columns: list | None = None

    for display_name, rng in subsets_cfg.items():
        if display_name not in subset_runs:
            logger.warning("No run found for subset '%s' — skipped", display_name)
            continue

        key = _subset_key(display_name)
        run_id = subset_runs[display_name]
        model, encoders, target_enc = load_subset_artifacts(run_id)

        out = BUNDLE_DIR / key
        out.mkdir()
        model.save_model(out / "model.ubj")
        joblib.dump(encoders, out / "encoders.joblib")
        joblib.dump(target_enc, out / "target_encoder.joblib")

        if feature_columns is None and hasattr(model, "feature_names_in_"):
            feature_columns = list(model.feature_names_in_)

        routing["subsets"][key] = {
            "display_name": display_name,
            "hour_min": int(rng["hour_min"]),
            "hour_max": int(rng["hour_max"]),
            "run_id": run_id,
            "n_classes": int(len(target_enc.classes_)),
        }
        logger.info("Exported '%s' -> %s (%d classes)", display_name, key, len(target_enc.classes_))

    routing["feature_columns"] = feature_columns

    with open(BUNDLE_DIR / "routing.json", "w", encoding="utf-8") as f:
        json.dump(routing, f, ensure_ascii=False, indent=2)

    logger.info("Bundle written to %s (subsets: %s)", BUNDLE_DIR, list(routing["subsets"]))


if __name__ == "__main__":
    main()
