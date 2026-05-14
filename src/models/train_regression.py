"""Training script for linear regression model with MLflow tracking."""

import logging

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.preprocessing.common import fill_missing, load_features, split_data
from src.preprocessing.regression import preprocess_for_regression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COL = "party_size"  # TODO: update to actual target column


def main() -> None:
    df = load_features()
    df = fill_missing(df)

    X_train, X_test, y_train, y_test = split_data(df, target_col=TARGET_COL)

    X_train_processed, X_test_processed, scaler = preprocess_for_regression(
        X_train, X_test
    )

    with mlflow.start_run(run_name="linear_regression"):
        model = LinearRegression()
        model.fit(X_train_processed, y_train)

        y_pred = model.predict(X_test_processed)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params({"model": "LinearRegression", "target": TARGET_COL})
        mlflow.log_metrics({"rmse": rmse, "r2": r2})
        mlflow.sklearn.log_model(model, "model")

        logger.info("RMSE=%.4f, R2=%.4f", rmse, r2)


if __name__ == "__main__":
    main()
