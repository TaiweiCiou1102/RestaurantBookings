# Deployment — 三分模型 Azure ML 線上端點

把產品化的三分模型（午餐 / 下午茶 / 晚餐）部署成**一個** Azure ML managed
online endpoint。三個子模型打包成一份 model，分流邏輯寫在 `score.py`，依請求中的
`reservation_hour` 決定每筆走哪個子模型。

## 檔案說明

### 上雲執行的核心（一個 deployment = 模型 + 環境 + 程式）

| 檔案 | 用途 |
| ---- | ---- |
| `score.py` | 推論程式。`init()` 在容器啟動時載入三個子模型；`run()` 依 `reservation_hour` 分流→各自編碼預測→合併輸出。 |
| `conda.yaml` | 環境定義。容器要安裝的套件（pandas / xgboost / scikit-learn + `azureml-inference-server-http`）。版本對齊訓練環境。 |
| `model_bundle/` | 打包好的三分模型（各子集 `model.ubj` + `encoders.joblib` + `target_encoder.joblib`，加 `routing.json`）。註冊成一個 custom model。**由 `export_bundle.py` 產生、已 gitignore**。 |
| `deployment.yml` | 部署設定：綁定「註冊的 model + 環境 + score.py + 機器規格」。 |
| `endpoint.yml` | 端點設定：名稱與驗證方式（key）。 |
| `.amlignore` | 上傳排除清單，避免把 `model_bundle/`、測試檔等當成 code 一起上傳。 |

### 產生模型 / 本地驗證

| 檔案 | 用途 |
| ---- | ---- |
| `export_bundle.py` | 從 MLflow 匯出最新 `xgb_half_hour` 三分模型成 `model_bundle/`。模型更新後重跑即可。 |
| `test_score.py` | 本地測試（免連 Azure）：模擬容器跑 `init()`/`run()`，驗證三分路由、未匹配時段、缺路由欄、未見類別等。 |
| `example_input.json` | 範例請求；本地測試與 `az ml online-endpoint invoke` 共用。 |
| `schemas.py` | 輸入/輸出的 Pydantic 結構定義（文件化 + 驗證用，不參與雲端推論）。 |

## 前置需求

- Azure ML workspace（+ 訂閱 / resource group）
- Azure CLI 與 `ml` extension v2：`az extension add -n ml`
- managed online endpoint 的 compute quota（如 `Standard_DS3_v2`）

## 部署步驟

```bash
# 0) 產生 model bundle（需先有訓練好的 MLflow run）
uv run python deployment/export_bundle.py

# 0.1) 登入並設定預設 workspace
az login
az account set --subscription <SUB_ID>
az configure --defaults workspace=<WS> group=<RG> location=<REGION>

# 1) 把整個 bundle 註冊成一個 custom model
az ml model create --name resv-3split --version 1 \
    --path deployment/model_bundle --type custom_model

# 2) 建立 endpoint
az ml online-endpoint create -f deployment/endpoint.yml

# 3) 建立 deployment（綁 model + 環境 + score.py）並導入全部流量
az ml online-deployment create -f deployment/deployment.yml --all-traffic

# 4) 測試
az ml online-endpoint invoke --name resv-3split-endpoint \
    --request-file deployment/example_input.json
```

> 更新模型時：重跑 `export_bundle.py` → 以新 `--version` 重新 `az ml model create`
> → 更新 `deployment.yml` 的 `model:` 版本 → `az ml online-deployment update`。

## 本地驗證（不需 Azure）

```bash
uv run python deployment/export_bundle.py   # 先產生 bundle
uv run python deployment/test_score.py      # 跑 init()/run() 全套測試
```

## 輸入 / 輸出格式

**輸入**（Azure ML 標準格式）：`columns` 需含 **`reservation_hour`**（路由用）與 25 個模型特徵。

```json
{
  "input_data": {
    "columns": ["reservation_hour", "party_size", "dining_purpose", "..."],
    "index":   [0],
    "data":    [[19, 11, "朋友聚餐", "..."]]
  }
}
```

**輸出**：每列回傳預測 slot、可讀時間區間、所屬子模型與信心值。

```json
{
  "predictions": {
    "columns": ["reservation_half_hour", "time_range", "subset", "probability"],
    "index":   [0],
    "data":    [[38, "19:00-19:29", "dinner", 0.2915]]
  }
}
```

## 設計重點

- **`reservation_hour` 只用於路由**（決定餐期子模型），預測前即丟棄——它對模型而言是
  leakage。符合「客人用午餐 / 下午茶 / 晚餐可事先判斷」的情境。
- 落在所有餐期之外的列（如清晨）回傳 `null` + `subset="_unmatched"`，不會報錯。
- 容器內**沒有 MLflow / sqlite**，`score.py` 只從 `model_bundle/` 讀純檔案。
