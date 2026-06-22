# exploration/ — 模型探索階段（已凍結）

這個資料夾是專案初期「一次比較多個模型」的探索研究，**已凍結、不再積極維護**。
保留它是為了重現當初的模型比較結果。

## 內容
- `preprocessing/` — 探索階段各模型的前處理
  - `common.py` — 共用：train/test 切分、缺值處理
  - `regression.py` — one-hot、drop baseline、標準化（線性模型用）
  - `tree.py` — label / ordinal encoding（樹模型用）
- `train_regression.py` — Logistic / linear regression baseline
- `train_random_forest.py` — Random Forest
- `train_xgboost.py` — XGBoost（探索版）
- `inference.py` — 探索階段的推論腳本

## ⚠️ Source of truth 是 `src/xgboost/`
比較結束後選定 XGBoost，正式、持續維護的版本在 [`src/xgboost/`](../xgboost/)。
要訓練 / 推論請用那邊，**不要改這裡**。

共用的建模工具（MLflow logging、CV、Optuna、繪圖等）已抽到 [`src/common/model_utils.py`](../common/model_utils.py)，
本資料夾與 `src/xgboost/` 都從那裡 import。
