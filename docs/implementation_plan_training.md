# 實作計畫：三分法分類建模（RF / XGBoost）

## 設計原則

- 每次執行固定訓練 **3 個模型**（午餐、下午茶、晚餐）
- 顆粒度只有 **`hour`** 和 **`half_hour`** 兩種，由 CLI 參數 `--granularity` 決定
- 任務類型一律為**分類**
- MLflow 採 **parent + 3 nested child runs** 結構

---

## 檔案清單

| 檔案 | 動作 | 用途 |
|------|------|------|
| `pyproject.toml` | **修改** | 加入 `optuna` 依賴 |
| `src/preprocessing/common.py` | **修改** | `split_data` 新增 `shuffle` 參數支援 temporal split |
| `configs/random_forest.yaml` | **新增** | RF 超參數 + 顆粒度 + 子集 + 評估 + 調參設定 |
| `configs/xgboost.yaml` | **新增** | XGBoost 同上（多 `learning_rate`、`early_stopping_rounds`） |
| `src/models/_utils.py` | **新增** | 兩個訓練腳本共用的輔助函式 |
| `src/models/train_random_forest.py` | **新增** | RF 訓練腳本 |
| `src/models/train_xgboost.py` | **改寫** | XGBoost 訓練腳本 |

---

## 1. `pyproject.toml`（修改）

新增一行依賴：

```toml
"optuna>=4.0.0",
```

---

## 2. `src/preprocessing/common.py`（修改）

`split_data` 新增 `shuffle` 參數：

```python
def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,          # <- 新增
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
```

- `shuffle=True`：現行行為（隨機切分）
- `shuffle=False`：依序切分（呼叫端需先按 `booking_time` 排序）
- 傳遞 `shuffle` 給底層 `train_test_split`，不改變其他邏輯

---

## 3. `configs/random_forest.yaml`（新增）

```yaml
model_params:
  n_estimators: 200
  max_depth: 15
  min_samples_leaf: 5
  random_state: 42
  n_jobs: -1

granularity:
  hour:
    target: reservation_hour
    leakage: [reservation_seconds, reservation_half_hour]
  half_hour:
    target: reservation_half_hour
    leakage: [reservation_hour, reservation_seconds]

subsets:
  午餐 Lunch (11-14h):       { hour_min: 11, hour_max: 14 }
  下午茶 Afternoon (15-17h):  { hour_min: 15, hour_max: 17 }
  晚餐 Dinner (18-23h):      { hour_min: 18, hour_max: 23 }

evaluation:
  test_size: 0.2
  split_strategy: random        # random | temporal
  cv_folds: 5                   # 0 = 只做 holdout
  log_confusion_matrix: true
  log_feature_importance: true
  tolerance_slots: 1            # +-1 slot 容許準確率

# tuning:                       # 取消註解即啟動 Optuna
#   n_trials: 50
#   search_space:
#     n_estimators: { type: categorical, choices: [100, 200, 300, 500] }
#     max_depth:    { type: int, low: 4, high: 20 }
#     min_samples_leaf: { type: int, low: 2, high: 20 }
```

## 4. `configs/xgboost.yaml`（新增）

```yaml
model_params:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.1
  early_stopping_rounds: 30     # <- XGBoost 獨有
  random_state: 42
  n_jobs: -1

granularity:
  hour:
    target: reservation_hour
    leakage: [reservation_seconds, reservation_half_hour]
  half_hour:
    target: reservation_half_hour
    leakage: [reservation_hour, reservation_seconds]

subsets:
  午餐 Lunch (11-14h):       { hour_min: 11, hour_max: 14 }
  下午茶 Afternoon (15-17h):  { hour_min: 15, hour_max: 17 }
  晚餐 Dinner (18-23h):      { hour_min: 18, hour_max: 23 }

evaluation:
  test_size: 0.2
  split_strategy: random
  cv_folds: 5
  log_confusion_matrix: true
  log_feature_importance: true
  tolerance_slots: 1

# tuning:
#   n_trials: 50
#   search_space:
#     n_estimators:     { type: categorical, choices: [100, 300, 500, 1000] }
#     max_depth:        { type: int, low: 3, high: 12 }
#     learning_rate:    { type: float, low: 0.01, high: 0.3, log: true }
#     min_child_weight: { type: int, low: 1, high: 10 }
```

---

## 5. `src/models/_utils.py`（新增）

兩個訓練腳本共用的輔助函式：

| 函式 | 輸入 | 輸出 | 做什麼 |
|------|------|------|--------|
| `load_config(path)` | YAML 路徑 | `dict` | 讀取並回傳設定檔 |
| `resolve_granularity(config, gran)` | config dict, `"hour"` / `"half_hour"` | `(target, leakage)` | 從 config 取出對應的 target 欄位與 leakage 清單 |
| `tolerance_accuracy(y_true, y_pred, tol)` | 真實值、預測值、容許 slot 數 | `float` | `np.mean(np.abs(y_true - y_pred) <= tol)` |
| `plot_confusion_matrix(y_true, y_pred, labels)` | — | `matplotlib.Figure` | 畫正規化混淆矩陣，回傳 fig 物件供 MLflow 存檔 |
| `plot_feature_importance(importances, feature_names, top_n=20)` | — | `matplotlib.Figure` | 畫 top-N 特徵重要性橫條圖 |
| `log_evaluation_artifacts(y_test, y_pred, model, feature_names, eval_cfg)` | — | `None` | 依 `eval_cfg` flags 呼叫 plot 函式 + `classification_report`，統一寫入 MLflow |
| `run_cv(X, y, model_cls, model_params, cv_folds, preprocess_fn)` | — | `dict` | StratifiedKFold CV，每 fold 獨立做 `preprocess_fn` -> fit -> score；回傳 `{acc_mean, acc_std, f1_mean, f1_std}` |
| `build_optuna_objective(...)` | X, y, search_space, fixed_params, model_cls, cv_folds, preprocess_fn | `callable` | 回傳 Optuna `objective` 函式，內部用 CV 評估每組超參數 |
| `suggest_params(trial, search_space)` | Optuna trial, YAML search_space | `dict` | 依 `type` 欄位呼叫 `trial.suggest_categorical / suggest_int / suggest_float` |
| `save_pipeline_artifacts(encoders, target_encoder=None)` | encoder dicts | `None` | `joblib.dump` -> `mlflow.log_artifact` 存 encoders |

---

## 6. `src/models/train_random_forest.py`（新增）

### CLI 介面

```
uv run python -m src.models.train_random_forest \
    --granularity hour \
    [--config configs/random_forest.yaml] \
    [--tune]
```

| 參數 | 必填 | 說明 |
|------|------|------|
| `--granularity` | 是 | `hour` 或 `half_hour` |
| `--config` | 否 | YAML 路徑，預設 `configs/random_forest.yaml` |
| `--tune` | 否 | 帶了就啟動 Optuna 搜尋 |

### 主流程 `main()`

```
1. parse_args()
2. load_config(args.config)
3. resolve_granularity(config, args.granularity)  ->  target, leakage
4. load_features() -> dropna()
5. mlflow.set_experiment("random_forest")
6. mlflow.start_run(run_name=f"rf_{granularity}")   <- Parent Run
   |-- log_params: granularity, target, config_path
   |-- log_artifact: YAML 設定檔
   |
   |-- for subset_name, {hour_min, hour_max} in config["subsets"]:
   |     df_subset = df[reservation_hour.between(min, max)]
   |
   |     mlflow.start_run(run_name=subset_name, nested=True)  <- Child Run
   |       -> train_and_evaluate_rf(...)
   |
   +-- 印出 3 個子集的指標比較表
```

### `train_and_evaluate_rf()` 詳細流程

```
輸入: df_subset, target, leakage, config, args

 1. 移除洩漏欄位
    df_clean = df_subset.drop(columns=leakage)

 2. 切分資料
    if split_strategy == "temporal":
        df_clean = df_clean.sort_values("booking_time")
        shuffle = False
    else:
        shuffle = True
    X_train, X_test, y_train, y_test = split_data(
        df_clean, target, test_size, shuffle=shuffle
    )

 3. 預處理
    X_train_proc, X_test_proc, encoders = preprocess_for_tree(X_train, X_test)

 4. 超參數決定
    if args.tune and "tuning" in config:
        objective = build_optuna_objective(
            X_train_proc, y_train,
            search_space=config["tuning"]["search_space"],
            fixed_params={random_state, n_jobs},
            model_cls=RandomForestClassifier,
            cv_folds=config["evaluation"]["cv_folds"],
            preprocess_fn=preprocess_for_tree,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config["tuning"]["n_trials"])
        best_params = {**fixed_params, **study.best_params}
        mlflow.log_params({"optuna_best_trial": study.best_trial.number})
    else:
        best_params = config["model_params"]

 5. 訓練最終模型
    model = RandomForestClassifier(**best_params)
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)

 6. 評估指標
    accuracy, f1_weighted, tolerance_accuracy
    mlflow.log_params({**best_params, subset, n_samples, n_classes})
    mlflow.log_metrics({accuracy, f1_weighted, tolerance_accuracy})

 7. Cross-validation（若 cv_folds > 0 且非 tune 模式）
    cv_result = run_cv(X_train_proc, y_train, RF, best_params, cv_folds, preprocess_for_tree)
    mlflow.log_metrics({cv_acc_mean, cv_acc_std, cv_f1_mean, cv_f1_std})

 8. Artifacts
    log_evaluation_artifacts(y_test, y_pred, model, feature_names, eval_cfg)
      -> confusion_matrix.png
      -> classification_report.txt
      -> feature_importance.png

 9. Pipeline 序列化
    mlflow.sklearn.log_model(model, "model")
    save_pipeline_artifacts(encoders)
      -> encoders.joblib

 10. 回傳結果 dict 給 parent run 做比較表
```

---

## 7. `src/models/train_xgboost.py`（改寫）

結構與 RF 相同，以下只列**差異**：

| 步驟 | 差異 |
|------|------|
| **3. 預處理** | 額外對 target 做 `LabelEncoder`（XGBoost 要求 0..n-1 連續標籤） |
| **4. 超參數** | search_space 多了 `learning_rate: {type: float, log: true}` |
| **5. 訓練** | 若 config 有 `early_stopping_rounds`：從 X_train 再切 15% 作 `eval_set`，超過 N rounds 無改善自動停止；MLflow 記錄 `best_iteration` |
| **6. 評估** | `tolerance_accuracy` 要在 `inverse_transform` 回原始 slot 值後再算，確保量測的是真實 slot 距離 |
| **9. 序列化** | 用 `mlflow.xgboost.log_model`；額外存 `target_encoder.joblib` |

### Early Stopping 細節

```python
if "early_stopping_rounds" in model_params:
    esr = model_params.pop("early_stopping_rounds")

    # 從訓練集切出 15% 當 validation（不動 test set）
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_proc, y_train_encoded, test_size=0.15
    )

    model = XGBClassifier(
        n_estimators=1000,        # 設高上限
        early_stopping_rounds=esr,
        **other_params
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    mlflow.log_metric("best_iteration", model.best_iteration)
else:
    model = XGBClassifier(**model_params)
    model.fit(X_train_proc, y_train_encoded)
```

---

## MLflow 記錄結構

```
Experiment: "random_forest"
+-- Run: "rf_hour"                          <- Parent
    |-- params: granularity=hour, target=reservation_hour
    |-- artifact: random_forest.yaml
    |
    |-- Run: "午餐 Lunch (11-14h)"           <- Child 1
    |   |-- params: n_estimators=200, max_depth=15, ...
    |   |           subset=午餐, n_samples=12345, n_classes=4
    |   |-- metrics: accuracy, f1_weighted, tolerance_accuracy
    |   |            cv_acc_mean, cv_acc_std, cv_f1_mean, cv_f1_std
    |   |-- artifacts:
    |   |   |-- model/              (MLflow model)
    |   |   |-- encoders.joblib     (特徵 LabelEncoder)
    |   |   |-- confusion_matrix.png
    |   |   |-- classification_report.txt
    |   |   +-- feature_importance.png
    |
    |-- Run: "下午茶 Afternoon (15-17h)"      <- Child 2
    |   +-- ...同上
    |
    +-- Run: "晚餐 Dinner (18-23h)"          <- Child 3
        +-- ...同上

Experiment: "xgboost"
+-- Run: "xgb_half_hour"
    +-- ...同上結構，多了 target_encoder.joblib + best_iteration
```

---

## 使用方式

```bash
# --- 日常訓練 ---
uv run python -m src.models.train_random_forest --granularity hour
uv run python -m src.models.train_random_forest --granularity half_hour
uv run python -m src.models.train_xgboost --granularity hour
uv run python -m src.models.train_xgboost --granularity half_hour

# --- 調參 ---
# 1. 在 YAML 中取消 tuning 區塊的註解
# 2. 加上 --tune flag
uv run python -m src.models.train_xgboost --granularity hour --tune

# --- 換參數做實驗 ---
cp configs/xgboost.yaml configs/xgboost_v2.yaml
# 編輯 v2 的 model_params
uv run python -m src.models.train_xgboost --granularity hour --config configs/xgboost_v2.yaml

# --- 比較結果 ---
mlflow ui
```

---

## 資料流

```
features_ready.csv
      |
      |  load_features() + dropna()
      v
  +------------------------------------------+
  |  依 reservation_hour 篩出 3 個子集        |
  |  午餐(11-14) / 下午茶(15-17) / 晚餐(18-23) |
  +----+--------------+--------------+-------+
       v              v              v
  +---------+   +---------+   +---------+
  | drop    |   | drop    |   | drop    |
  | leakage |   | leakage |   | leakage |
  +----+----+   +----+----+   +----+----+
       v              v              v
  split_data     split_data     split_data
  (random/       (random/       (random/
   temporal)      temporal)      temporal)
       v              v              v
  preprocess     preprocess     preprocess
  _for_tree      _for_tree      _for_tree
       v              v              v
  +----------+  +----------+  +----------+
  | [--tune] |  | [--tune] |  | [--tune] |
  |  Optuna  |  |  Optuna  |  |  Optuna  |
  |  search  |  |  search  |  |  search  |
  +----+-----+  +----+-----+  +----+-----+
       v              v              v
     Train          Train          Train
   RF / XGB       RF / XGB       RF / XGB
       v              v              v
   Evaluate       Evaluate       Evaluate
   + CV           + CV           + CV
       v              v              v
  +--------------------------------------+
  |         MLflow Parent Run            |
  |  +------+  +------+  +------+       |
  |  |Child1|  |Child2|  |Child3|       |
  |  |model |  |model |  |model |       |
  |  |report|  |report|  |report|       |
  |  +------+  +------+  +------+       |
  |         比較總表                      |
  +--------------------------------------+
```
