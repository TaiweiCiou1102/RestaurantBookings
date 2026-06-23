"""Shared utilities for model training scripts."""

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CLI usage
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  MLflow setup
# ---------------------------------------------------------------------------

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def init_tracking(uri: str = DEFAULT_TRACKING_URI) -> None:
    """Point MLflow at the project's tracking backend.

    Run metadata lives in ``mlflow.db`` (SQLite); artifacts live under
    ``./mlruns``. Without this call, ``MlflowClient()`` defaults to the
    ``./mlruns`` file store, whose per-run ``meta.yaml`` is absent here, so
    no runs are discoverable even though the artifacts exist on disk. Call
    once (from the project root) before any MLflow read/write.
    """
    mlflow.set_tracking_uri(uri)
    logger.info("MLflow tracking URI: %s", uri)


# ---------------------------------------------------------------------------
#  Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    import yaml

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return config


def resolve_granularity(config: dict, granularity: str) -> Tuple[str, List[str]]:
    """Return (target_col, leakage_cols) for the chosen granularity.

    Raises:
        KeyError: If the granularity is not defined in the config.
    """
    gran = config["granularity"][granularity]
    return gran["target"], gran["leakage"]


# ---------------------------------------------------------------------------
#  Evaluation metrics
# ---------------------------------------------------------------------------

def tolerance_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: int = 1,
) -> float:
    """Accuracy allowing ±tolerance slots of error."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred)) <= tolerance))


# ---------------------------------------------------------------------------
#  Plotting helpers (return Figure, never call plt.show)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
) -> plt.Figure:
    """Plot a normalised confusion-matrix heatmap and return the Figure."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))
    sns.heatmap(cm_norm, annot=len(labels) <= 20, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Normalized)")
    ax.tick_params(axis="both", labelsize=max(6, 10 - len(labels) // 5))
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    title: str = "Feature Importance",
) -> plt.Figure:
    """Plot a horizontal bar chart of top-N feature importances."""
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh(range(len(imp_df)), imp_df["importance"], color="steelblue")
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df["feature"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
#  MLflow artifact logging
# ---------------------------------------------------------------------------

def log_evaluation_artifacts(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model: Any,
    feature_names: List[str],
    eval_cfg: dict,
    labels: Optional[List] = None,
) -> None:
    """Log confusion matrix, classification report, and feature importance to MLflow."""
    # Classification report (always logged)
    report = classification_report(y_test, y_pred, zero_division=0)
    mlflow.log_text(report, "classification_report.txt")

    # Confusion matrix
    if eval_cfg.get("log_confusion_matrix", False):
        fig = plot_confusion_matrix(y_test, y_pred, labels=labels)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

    # Feature importance
    if eval_cfg.get("log_feature_importance", False) and hasattr(model, "feature_importances_"):
        fig = plot_feature_importance(model.feature_importances_, feature_names)
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)


def save_pipeline_artifacts(
    encoders: Dict[str, Any],
    target_encoder: Any = None,
) -> None:
    """Serialise preprocessing encoders and log them as MLflow artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        enc_path = Path(tmpdir) / "encoders.joblib"
        joblib.dump(encoders, enc_path)
        mlflow.log_artifact(str(enc_path))

        if target_encoder is not None:
            te_path = Path(tmpdir) / "target_encoder.joblib"
            joblib.dump(target_encoder, te_path)
            mlflow.log_artifact(str(te_path))


# ---------------------------------------------------------------------------
#  MLflow run discovery
# ---------------------------------------------------------------------------

def find_nested_runs(
    experiment_name: str,
    parent_run_id: str | None = None,
    parent_run_name_contains: str | None = None,
) -> Dict[str, str]:
    """Return {run_name: run_id} for nested runs under a parent run.

    If *parent_run_id* is None, the latest finished parent run in the
    experiment is used automatically.  When *parent_run_name_contains* is
    given (e.g. ``"hour"``), only parent runs whose name contains that
    substring are considered.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    if parent_run_id is None:
        all_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )
        # Identify parent runs: those whose run_id appears as a parentRunId.
        child_parent_ids = {
            r.data.tags["mlflow.parentRunId"]
            for r in all_runs
            if "mlflow.parentRunId" in r.data.tags
        }
        parent_runs = [
            r for r in all_runs
            if r.info.run_id in child_parent_ids
               and r.info.status == "FINISHED"
        ]
        if parent_run_name_contains:
            parent_runs = [
                r for r in parent_runs
                if parent_run_name_contains in (r.info.run_name or "")
            ]
        if not parent_runs:
            raise ValueError(
                f"No finished parent run found in experiment '{experiment_name}'"
                + (f" matching '{parent_run_name_contains}'" if parent_run_name_contains else "")
            )
        parent_run_id = parent_runs[0].info.run_id
        logger.info("Auto-selected parent run: %s (%s)", parent_runs[0].info.run_name, parent_run_id)

    nested = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )
    mapping = {r.info.run_name: r.info.run_id for r in nested}
    logger.info("Found %d nested runs: %s", len(mapping), list(mapping.keys()))
    return mapping


# ---------------------------------------------------------------------------
#  Cross-validation
# ---------------------------------------------------------------------------

def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_cls: type,
    model_params: dict,
    cv_folds: int,
    preprocess_fn: Callable,
) -> Dict[str, float]:
    """Run StratifiedKFold CV, preprocessing independently per fold.

    Returns:
        Dict with acc_mean, acc_std, f1_mean, f1_std.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    acc_scores: List[float] = []
    f1_scores: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_tr_proc, X_val_proc, _ = preprocess_fn(X_tr, X_val)

        mdl = model_cls(**model_params)
        mdl.fit(X_tr_proc, y_tr)
        y_pr = mdl.predict(X_val_proc)

        acc_scores.append(accuracy_score(y_val, y_pr))
        f1_scores.append(f1_score(y_val, y_pr, average="weighted", zero_division=0))
        logger.debug("CV fold %d: acc=%.4f, f1=%.4f", fold, acc_scores[-1], f1_scores[-1])

    return {
        "cv_acc_mean": float(np.mean(acc_scores)),
        "cv_acc_std": float(np.std(acc_scores)),
        "cv_f1_mean": float(np.mean(f1_scores)),
        "cv_f1_std": float(np.std(f1_scores)),
    }


# ---------------------------------------------------------------------------
#  Optuna helpers
# ---------------------------------------------------------------------------

def suggest_params(trial: Any, search_space: dict) -> dict:
    """Translate YAML search-space spec into Optuna trial suggestions."""
    params: dict = {}
    for name, spec in search_space.items():
        stype = spec["type"]
        if stype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif stype == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif stype == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False),
            )
        else:
            raise ValueError(f"Unknown search-space type '{stype}' for param '{name}'")
    return params


def build_optuna_objective(
    X: pd.DataFrame,
    y: pd.Series,
    search_space: dict,
    fixed_params: dict,
    model_cls: type,
    cv_folds: int,
    preprocess_fn: Callable,
) -> Callable:
    """Return an Optuna objective that maximises CV accuracy."""

    def objective(trial: Any) -> float:
        tuned = suggest_params(trial, search_space)
        merged = {**fixed_params, **tuned}

        cv_result = run_cv(X, y, model_cls, merged, cv_folds, preprocess_fn)
        return cv_result["cv_acc_mean"]

    return objective
