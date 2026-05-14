# RestaurantBookings

> <!-- TODO: 一句話描述這個專案（例如：「預測餐廳訂位是否會回流的機器學習專案」） -->

---

## 專案目的

本專案透過使用訂位平台上的訂位資料，使用xgboost演算法，預測顧客最有可能的用餐時段。使用資料來自[kaggle-Predict Repeat Restaurant Bookings](https://www.kaggle.com/competitions/predict-repeat-restaurant-bookings/overview)

本專案是取得客戶資料前驗證構想的Lab，動機來自餐飲業的真實需求。我們希望能夠為客戶設計一個推薦系統，為顧客推薦用餐時間。預設情境是當顧客遇到原先想要預定的用餐時段沒有空位之後，串接系統的AIAgent可以根據這個推薦模型找出最適合的時段推薦給他。

## 說明

我們將客戶實際的入座時間，依據每半小時或每一小時切成不同用餐時段—因此模型變成一個分類預測問題，並利用各種呈現於資料中的客戶屬性預測用餐時間。

本模型開發中的主要困難，在於客戶集中分布於中餐時間與晚餐時間，若將資料一概而論進行訓練，絕大多數都預測在下午時段，即中餐與晚餐之間，與我們要精確預測客戶用餐時間的目的有別。本模型因此拆分為三個小模型，即分別對中餐、下午、晚餐時段的資料進行訓練。由於實際需求是為客人推薦用餐時段，判斷客人是用午餐、下午茶還是晚餐的任務應可事先判斷。

![全資料集訓練與三子模型比較-直方圖](./docs/distribution.png)

- 全資料模型與三分模型預測比較圖

![全資料集訓練與三子模型比較-表現](./docs/output.png)

- 從圖片中可以得知，將資料切分成三個子模型集大提升的Accuracy和F1-score。

我們的結果發現，對於下午時段(2點到6點)的預測最為準確，午餐和晚餐的預測則有待加強。模型在三個時段都高估尖峰時段的來客數，推測與不均衡樣本資料的特性有關，模型傾向將預測結果向眾數靠攏所致。

---

## 專案架構

```
RestaurantBookings/
├── configs/                     # 設定檔
│   ├── raw_data_schema.yaml     #   原始資料欄位 schema
│   ├── regression.yaml          #   Regression 超參數設定
│   ├── random_forest.yaml       #   Random Forest 超參數設定
│   └── xgboost.yaml             #   XGBoost 超參數設定
│
├── data/                        # 資料（透過 Git LFS 管理）
│   ├── raw/                     #   原始資料集
│   ├── interim/                 #   清理 / 整合後的中間資料
│   └── processed/
│       └── features_ready.csv   #   所有模型共用的起點
│
├── src/
│   ├── etl/                     # 領域知識 / 特徵工程
│   │   ├── run_cleaning.py
│   │   ├── run_integration.py
│   │   ├── run_features.py
│   │   ├── _cleaning_utils.py
│   │   ├── _integration_utils.py
│   │   └── _features_utils.py
│   │
│   ├── preprocessing/           # 模型專屬編碼 / 縮放
│   │   ├── common.py            #   共用：train/test split、缺值、標準化
│   │   ├── regression.py        #   One-hot encoding、drop baseline、standardize
│   │   └── tree.py              #   Label encoding 或保留整數、不縮放
│   │
│   └── models/                  # 訓練、評估、MLflow 紀錄
│       ├── train_regression.py
│       ├── train_random_forest.py
│       ├── train_xgboost.py
│       └── xgboost_inference.py
│
├── deployment/                  # 模型部署 (MLflow serving)
│   ├── conda.yaml               #   推論環境
│   ├── score.py                 #   推論入口
│   ├── schemas.py               #   I/O Pydantic schema
│   ├── example_input.json
│   └── test_score.py
│
├── notebooks/                   # 探索性分析與實驗
├── docs/                        # 文件與 API collection
└── mlruns/                      # MLflow tracking（本地、不入 git）
```

## 資料流

```
raw/ ──► run_cleaning ──► interim/ ──► run_integration ──► run_features ──► processed/features_ready.csv
                                                                                       │
                                                                                       ▼
                                                          preprocessing/{regression, tree}.py
                                                                                       │
                                                                                       ▼
                                                                  models/train_*.py ──► MLflow
                                                                                       │
                                                                                       ▼
                                                                            deployment/score.py
```

## 設計原則

- **ETL**：只做領域知識特徵工程，**不做** 模型專屬編碼
- **Preprocessing**：模型專屬轉換（regression 走 one-hot；tree-based 走 label / 保留整數）
- **Training**：每個模型腳本自行呼叫對應 preprocessing 並負責 MLflow logging

---

## 環境需求

- Python `>= 3.12`
- [uv](https://docs.astral.sh/uv/)（套件管理）
- [Git LFS](https://git-lfs.com/)（資料檔案需透過 LFS 拉取）

主要相依套件：`pandas`、`scikit-learn`、`xgboost`、`mlflow`、`optuna`、`pydantic`、`ydata-profiling`

## 安裝

```bash
# 1. clone 並拉取 LFS 資料
git clone https://github.com/TaiweiCiou1102/RestaurantBookings.git
cd RestaurantBookings
git lfs pull

# 2. 建立虛擬環境並安裝相依
uv sync
```

## 使用方式

### 跑 ETL 產生特徵

```bash
uv run python -m src.etl.run_cleaning
uv run python -m src.etl.run_integration
uv run python -m src.etl.run_features
```

### 訓練模型

```bash
uv run python -m src.models.train_regression
uv run python -m src.models.train_random_forest
uv run python -m src.models.train_xgboost
```

### 查看 MLflow 實驗結果

```bash
uv run mlflow ui
# 預設開在 http://127.0.0.1:5000
```

### 模型推論 / 部署

<!-- TODO: 補充部署方式
例如：
- MLflow Model Serving 啟動指令
- API endpoint 規格
- 範例請求 / 回應（可參考 docs/RestaurantBooking.postman_collection.json）
-->

---

## 實驗追蹤

本專案使用 **MLflow** 追蹤實驗。所有訓練腳本會自動記錄：

- 超參數（從 `configs/*.yaml` 讀取）
- 評估指標
- 模型 artifact
- 訓練資料指紋

預設追蹤資料寫入本地 `mlruns/`（已在 `.gitignore`，不入庫）。

---

## 資料來源

<!-- TODO: 說明資料來源
例如：
- 來自 Kaggle 競賽 / 公司內部資料倉儲
- 資料時間範圍
- 主要資料表（會員、訂單、餐廳）的關聯關係
-->

## 模型表現

<!-- TODO: 填入目前最佳模型的指標
| Model         | AUC   | F1    | Notes                     |
|---------------|-------|-------|---------------------------|
| Regression    |       |       |                           |
| Random Forest |       |       |                           |
| XGBoost       |       |       |                           |
-->

---

## 開發者

<!-- TODO: 作者 / 維護者 / 聯絡方式 -->

## License

<!-- TODO: 選擇授權（MIT / Apache-2.0 / Proprietary / 等） -->
