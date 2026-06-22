# 特徵調整決策紀錄

手動調整模型特徵（合併類別、刪除欄位等）的決策 log。
每筆記錄：日期、決定、理由、狀態（待實作 / 已實作）。

模型範圍：`src/xgboost/`（source of truth）。資料來源：`data/processed/features_ready.csv`。

---

## 2026-06-22 — 刪除 `city_area_code` 特徵

**決定：** 從模型特徵中移除 `city_area_code`（餐廳行政區）。

**理由：**
- 約 65% 的列沒有有效值：`Unknown` 佔 53.3%、`NaN` 佔 11.4%，真正有行政區的只有 ~35%。
- 38 種值呈長尾分佈，多數行政區 < 0.2%（如南港區、烏來區、鳳山區僅個位數筆數），對模型近乎雜訊。
- 粒度過細且彼此區別小（例：大安區 vs 士林區同為台北市行政區），缺乏穩定的預測訊號。
- 餐廳地點的地理訊號已由 `lat` / `lng` / `city_code` / `restaurant_density` 覆蓋，`city_area_code` 邊際貢獻低。

**實驗驗證（2026-06-22, half-hour, 5-fold CV, 其餘設定不變）：** 移除後三個 subset 全部持平或變好。

| Subset | acc keep→drop | CV acc keep→drop |
|---|---|---|
| 午餐 | 0.4651 → 0.4678 | 0.4557 → **0.4665** (+0.011) |
| 下午茶 | 0.8179 → **0.8228** | 0.8277 → 0.8283 |
| 晚餐 | 0.4330 → 0.4358 | 0.4219 → 0.4234 |

關鍵：`city_area_code` 在下午茶的 gain importance 高居第一（13.5%），但移除後成效反而更好——證實該 gain 是高基數類別在小樣本（13343 筆 / 38 類）上的**過擬合假象**，非真訊號。

**狀態：** ✅ 已驗證、已實作。`city_area_code` 加入 `NON_FEATURE_COLS`（訓練與推論皆排除）、自 `CATEGORICAL_COLS` 與 inference `FEATURE_COLUMNS` 移除。

---

## 2026-06-22 — 刪除地理編碼 `city_code` + `member_city_code`（保留 `lat`/`lng`）

**決定：** 從模型特徵移除 `city_code`（餐廳城市）與 `member_city_code`（會員城市）；保留 `lat`/`lng`。

**背景：** 假設 `lat`/`lng` 與 city code 類重複編碼了地理資訊。設計乾淨的 head-to-head 實驗，一次只給模型一種地理表示，並加「完全無地理」當底線，回答「模型偏好 code 還是 lat/lng」。

**實驗數據（2026-06-22, half-hour, 5-fold CV，僅差異該組地理欄位，其餘不變）：**

| 變體 | 午餐 CV | 下午茶 CV | 晚餐 CV |
|---|---|---|---|
| `all_geo`（codes + lat/lng，現狀） | 0.4665 | 0.8283 | 0.4234 |
| `codes_only`（留 codes，丟 lat/lng） | 0.4639 | 0.8262 | 0.4251 |
| `latlng_only`（留 lat/lng，丟 codes） | **0.4743** | 0.8317 | 0.4323 |
| `no_geo`（地理全丟） | 0.4740 | **0.8320** | **0.4328** |

（holdout accuracy / F1 / tol±1 三時段同向：`codes_only` 永遠墊底、`latlng_only` ≈ `no_geo` 永遠最佳。）

**理由：**
- **冗餘假設證實**：一次只給一種地理表示，效果與兩種都給相當或更好。
- **模型偏好 `lat`/`lng`**：`latlng_only` 三時段全勝 `codes_only`（午餐 +0.010、下午茶 +0.0055、晚餐 +0.0072）。連續座標用閾值分裂更有效、不過擬合；名目 code 需做類別分組，高基數帶來過擬合。
- **code 類主動拖累**：`codes_only` 三時段都**輸給 `no_geo`**（午餐 −1.0pp、下午茶 −0.6pp、晚餐 −0.8pp）——名目高基數的過擬合蓋過了微弱的地理訊號，留著比沒有還差。
- **保留 lat/lng 的理由**：`no_geo` ≈ `latlng_only`（差距 0.0003~0.0005，統計上等價），lat/lng 相對無地理近乎零加分，但為連續低基數、無過擬合風險，留作地理 fallback 無害。
- **可解釋性需求另解**：`city_code` 仍可用於**事後分析 / 報表分群**（按城市切預測結果），不必當模型特徵。

**狀態：** ✅ 已驗證、已實作。`city_code` + `member_city_code` 加入 `NON_FEATURE_COLS`（訓練與推論皆排除）、自 `CATEGORICAL_COLS` 與 inference `FEATURE_COLUMNS` 移除；`lat`/`lng` 保留。

---

## 評估中（尚未定案）

以下欄位待更細的占比分析後決定，先記錄方向：

- **`dining_purpose`（8 種）** — 重要約會(0.4%)、其他(28 筆) 極稀疏。方向：併入 `Unknown`。
- **`account_gender`** — 實質 F/M + NaN，"Unknown" 僅 3 筆。方向：併入 NaN。
