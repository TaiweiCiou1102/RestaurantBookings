"""Training script for Random Forest classifier with MLflow tracking.

Usage:
    uv run python -m src.models.train_random_forest --granularity hour
    uv run python -m src.models.train_random_forest --granularity half_hour --tune
    uv run python -m src.models.train_random_forest --granularity hour --config configs/rf_v2.yaml
"""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

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

DEFAULT_CONFIG = "configs/random_forest.yaml"


# ------------------------------------------------------------------
#  Core training logic (one subset)
# ------------------------------------------------------------------

def train_and_evaluate(
    df_subset: pd.DataFrame,
    subset_name: str,
    target: str,
    leakage: list[str],
    config: dict,
    tune: bool,
) -> dict:
    """Train a Random Forest on *df_subset* and log everything to MLflow.

    Returns a result dict for the comparison summary.
    """
    eval_cfg = config["evaluation"]
    model_params = {k: v for k, v in config["model_params"].items()}

    # --- Drop leakage --------------------------------------------------
    df_clean = df_subset.drop(columns=leakage, errors="ignore")

    # --- Split ----------------------------------------------------------
    is_temporal = eval_cfg.get("split_strategy", "random") == "temporal"
    if is_temporal:
        df_clean = df_clean.sort_values("booking_time")

    X_train, X_test, y_train, y_test = split_data(
        df_clean,
        target_col=target,
        test_size=eval_cfg.get("test_size", 0.2),
        shuffle=not is_temporal,
    )

    # --- Preprocess -----------------------------------------------------
    X_train_proc, X_test_proc, encoders = preprocess_for_tree(X_train, X_test)

    # --- Optuna tuning (optional) ---------------------------------------
    fixed_keys = {"random_state", "n_jobs"}
    if tune and "tuning" in config:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        fixed = {k: v for k, v in model_params.items() if k in fixed_keys}
        objective = build_optuna_objective(
            X_train_proc, y_train,
            search_space=config["tuning"]["search_space"],
            fixed_params=fixed,
            model_cls=RandomForestClassifier,
            cv_folds=max(eval_cfg.get("cv_folds", 3), 3),
            preprocess_fn=preprocess_for_tree,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config["tuning"]["n_trials"])
        model_params = {**fixed, **study.best_params}
        mlflow.log_param("optuna_best_trial", study.best_trial.number)
        mlflow.log_metric("optuna_best_cv_acc", study.best_value)
        logger.info("Optuna best params: %s (cv_acc=%.4f)", study.best_params, study.best_value)

    # --- Train final model ----------------------------------------------
    model = RandomForestClassifier(**model_params)
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)

    # --- Metrics --------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    tol = eval_cfg.get("tolerance_slots", 1)
    tol_acc = tolerance_accuracy(y_test, y_pred, tolerance=tol)

    mlflow.log_params({
        **{k: v for k, v in model_params.items()},
        "subset": subset_name,
        "n_samples": len(df_subset),
        "n_classes": y_train.nunique(),
    })
    mlflow.log_metrics({
        "accuracy": acc,
        "f1_weighted": f1,
        f"tolerance_{tol}_accuracy": tol_acc,
    })

    logger.info(
        "[%s] Accuracy=%.4f  F1=%.4f  Tol±%d=%.4f",
        subset_name, acc, f1, tol, tol_acc,
    )

    # --- Cross-validation (skip if tuning already did CV) ---------------
    cv_folds = eval_cfg.get("cv_folds", 0)
    if cv_folds > 0 and not tune:
        cv_result = run_cv(
            X_train_proc, y_train,
            RandomForestClassifier, model_params,
            cv_folds, preprocess_for_tree,
        )
        mlflow.log_metrics(cv_result)
        logger.info(
            "[%s] CV acc=%.4f±%.4f  f1=%.4f±%.4f",
            subset_name,
            cv_result["cv_acc_mean"], cv_result["cv_acc_std"],
            cv_result["cv_f1_mean"], cv_result["cv_f1_std"],
        )

    # --- Artifacts ------------------------------------------------------
    log_evaluation_artifacts(
        y_test, y_pred, model,
        feature_names=list(X_train_proc.columns),
        eval_cfg=eval_cfg,
    )

    # --- Pipeline serialisation -----------------------------------------
    mlflow.sklearn.log_model(model, "model")
    save_pipeline_artifacts(encoders)

    return {
        "subset": subset_name,
        "n_samples": len(df_subset),
        "accuracy": acc,
        "f1_weighted": f1,
        f"tol_{tol}_acc": tol_acc,
    }


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest (三分法)")
    parser.add_argument(
        "--granularity", required=True, choices=["hour", "half_hour"],
        help="Prediction granularity: 'hour' or 'half_hour'.",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable Optuna hyper-parameter search.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    target, leakage = resolve_granularity(config, args.granularity)

    # --- Load data ------------------------------------------------------
    df = load_features()
    df = fill_missing(df, strategy="drop")

    # --- MLflow parent run ----------------------------------------------
    mlflow.set_experiment("random_forest")

    with mlflow.start_run(run_name=f"rf_{args.granularity}"):
        mlflow.log_params({
            "granularity": args.granularity,
            "target": target,
            "config_path": args.config,
        })
        mlflow.log_artifact(args.config)

        results: list[dict] = []

        for subset_name, subset_range in config["subsets"].items():
            df_subset = df[
                df["reservation_hour"].between(
                    subset_range["hour_min"], subset_range["hour_max"],
                )
            ]
            logger.info(
                "Subset '%s': %d rows", subset_name, len(df_subset),
            )

            with mlflow.start_run(run_name=subset_name, nested=True):
                result = train_and_evaluate(
                    df_subset, subset_name, target, leakage, config, args.tune,
                )
                results.append(result)

        # --- Comparison summary -----------------------------------------
        print("\n" + "=" * 70)
        print("  COMPARISON SUMMARY — Random Forest")
        print("=" * 70)
        summary = pd.DataFrame(results)
        print(summary.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
