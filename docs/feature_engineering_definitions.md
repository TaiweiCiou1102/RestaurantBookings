# 特徵工程功能定義文件 (Feature Engineering Definitions)

### 1. `lead_time`
- **功能描述**：計算客人從「實際訂位」到「預定用餐」之間提前了多少天。
- **產生方式**：`reserved_for` (預定用餐時間) - `booked_at` (實際訂位時間)
- **Args**:
    - `df`: pd.DataFrame
    - `booking_time`: str (實際上打電話訂位的日期欄位，如：`booked_at`)
    - `reservation_time`: str (預定要用餐的日期欄位，如：`reserved_for`)
- **Return**: pd.DataFrame (新增 `lead_time` 欄位)

---

### 2. `calculate_age`
- **功能描述**：根據客人生日與指定的基準時間點，計算客人在該時間點的年齡。
- **產生方式**：`(ref_date - birth_date)` 並換算為整數歲數。
- **Args**:
    - `df`: pd.DataFrame
    - `birth_date_col`: str (用來計算年紀的生日欄位)
    - `ref_date_col`: str (計算年紀的基準時間點欄位，如：用餐當日或當前日期)
- **Return**: pd.DataFrame (新增 `age` 欄位)

---

### 3. `booking_hour`
- **功能描述**：將指定的時間欄位切割成小時顆粒度，並展開為多個時段欄位以利分析。
- **產生方式**：提取日期時間中的 `hour` (0-23)，並進行 One-hot Encoding。
- **Args**:
    - `df`: pd.DataFrame
    - `datetime_col`: str (要用來轉換的日期時間格式欄位)
- **Return**: pd.DataFrame (新增 `hour_0` 至 `hour_23` 等多個欄位)

---

### 4. `weekday`
- **功能描述**：將「日期時間」欄位轉換為對應的星期幾。
- **產生方式**：將日期時間物件轉換為星期名稱（Monday-Sunday）或數值（0-6）。
- **Args**:
    - `df`: pd.DataFrame
    - `datetime_col`: str (要用來轉換的日期時間格式欄位)
- **Return**: pd.DataFrame (新增 `weekday` 欄位)

---

### 5. `average_price`
- **功能描述**：計算餐廳或套餐的平均價位。
- **產生方式**：`(PRICE1 + PRICE2) / 2`
- **Args**:
    - `df`: pd.DataFrame
    - `price1_col`: str (價格區間下限，通常為 `PRICE1`)
    - `price2_col`: str (價格區間上限，通常為 `PRICE2`)
- **Return**: pd.DataFrame (新增 `avg_price` 欄位)

---

### 6. `is_holiday_vicinity`
- **功能描述**：判斷用餐日期是否落在國定假日的「附近」範圍內（前後 7 天）。
- **產生方式**：檢查 `datetime_col` 與 `holiday_list` 中任一假日的差距是否介於 -7 到 +7 天之間。
- **Args**:
    - `df`: pd.DataFrame
    - `datetime_col`: str (要用來判斷的日期時間格式欄位)
    - `holiday_list`: list (國定假日日期清單)
- **Return**: pd.DataFrame (新增 `is_holiday_vicinity` 布林值欄位)

---

### 7. `days_from_payday`
- **功能描述**：計算指定日期距離發薪日（每月固定日期）的天數。
- **產生方式**：計算 `datetime_col` 與最近的發薪日期（需考慮跨月邏輯）之間的絕對天數差。
- **Args**:
    - `df`: pd.DataFrame
    - `datetime_col`: str (要用來計算的日期時間格式欄位)
    - `payday`: int (定義的發薪日期，例如：5 或 25)
- **Return**: pd.DataFrame (新增 `days_from_payday` 欄位)

---

### 8. `restaurant_popular_hour`
- **功能描述**：計算每間餐廳過去最常被預訂用餐的時段（例如最熱門的小時，或是晚餐時段佔比），用來量化該餐廳的「主打時段」屬性。
- **產生方式**：依據 `restaurant_col` 進行分組 (groupby)，計算 `datetime_col` 中各個小時的訂單數量，找出特定餐廳的時段眾數 (Mode) 或特定時段比例，再 mapping 回原始表格。
- **Args**:
    - `df`: pd.DataFrame
    - `restaurant_col`: str (餐廳 ID 欄位，如 `restaurant_id`)
    - `datetime_col`: str (用來計算熱門時段的實際用餐時間欄位，如 `datetime`)
- **Return**: pd.DataFrame (新增 `popular_hour` 或特定時段佔比欄位，如 `dinner_ratio`)

---

### 9. `area_restaurant_density`
- **功能描述**：計算每個縣市或行政區的餐廳總數，用以衡量該地區的「餐飲競爭激烈程度」與「顧客選擇多樣性」。
- **產生方式**：依據 `area_col` 進行分組 (groupby)，計算該區域內不重複的 `restaurant_col` 數量，並將該數值 Mapping 回原預測資料表中。
- **Args**:
    - `df`: pd.DataFrame
    - `area_col`: str (區域劃分欄位，如 `city` 或 `cityarea`)
    - `restaurant_col`: str (計算不重複數量的餐廳 ID 欄位，如 `restaurant_id`)
- **Return**: pd.DataFrame (新增 `restaurant_density` 欄位)

---

### 10. `weather_condition`
- **功能描述**：將當日的天氣狀況（如下雨與否、氣溫）結合到訂單中，用以分析天氣好壞對客人訂單行為（如是否取消、提前多久訂位）的影響。
- **產生方式**：將外部整理好的天氣資料表，透過「用餐純日期」與「餐廳所在縣市」，與原始訂單資料進行 `merge` (Left Join)。
- **Args**:
    - `df`: pd.DataFrame (包含訂單的原始資料表)
    - `weather_df`: pd.DataFrame (包含天氣資訊的外部輔助資料表)
    - `datetime_col`: str (用來對齊天氣的用餐時間欄位，會先萃取出純日期)
    - `city_col`: str (用來對應各地區天氣的餐廳縣市欄位)
- **Return**: pd.DataFrame (新增 `is_raining`, `avg_temperature` 等天氣關聯欄位)
