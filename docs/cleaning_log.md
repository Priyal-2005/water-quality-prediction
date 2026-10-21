# Data Cleaning Log & Audit Trail (Section 5.3)

- **Owner:** Sarthak (Data Lead)
- **Raw Source:** `data/raw/Ireland_dataset.csv`
- **Processed Output:** `data/processed/clean.csv`
- **Split Indices Output:** `data/processed/split_indices.json`
- **Raw Input Dimensions:** 235,019 rows × 14 columns
- **Clean Output Dimensions:** 235,019 rows × 13 columns *(adjustable if exact duplicate rows are dropped during execution)*

---

## 1. Cleaning & Domain Validation Log Table

| Issue found | How many rows/values | Action | Why |
| :--- | :--- | :--- | :--- |
| **Schema & column header formatting** (spaces, mixed case, unit suffixes in names) | 14 columns | Standardized to `snake_case` via `clean_column_names()`; mapped `Area` to `site_id` | Ensures consistent programmatic access across EDA, feature engineering, and modeling pipelines |
| **Date format standardization** (`Date` stored as `object` string `dd-mm-yyyy`) | 235,019 | Converted strings to `datetime64[ns]` (`dayfirst=True`) | Enables chronological verification and seasonal/cyclical feature extraction in Section 5.4 |
| **Left-censored detection limits check** (e.g., `<0.02` text strings in numeric columns) | 0 (All 8 columns verified `float64`) | Automated parser active in `src/data.py`: substitutes `0.5 * limit` and creates `{col}_bdl_flag` if encountered | Standard environmental chemistry convention for left-censored instrument detection limits |
| **Unit consistency audit** (`mg/L` vs `µg/L`, `°C` vs `°F`) | 235,019 rows across 8 parameters | Verified homogeneous SI units (`mg/L`, `°C`, `pH units`); no scaling conversion required | Prevents multi-scale distortion across monitoring stations and time periods |
| **Exact duplicate rows check** (`df.duplicated()`) | * Verified at runtime (0 or exact duplicate count)* | Removed exact duplicate rows and reset index | Identical repeated records artificially inflate sample weights and risk train/test leakage |
| **`ph` physical bounds check `[0.0, 14.0]`** | *Verified at runtime* | Set out-of-range values to `NaN`; retained rows | Values outside 0–14 violate the physical logarithmic pH scale (sensor calibration error) |
| **`ammonia_mg_l` physical bounds check `[0.0, 100.0]`** | *Verified at runtime* | Set out-of-range values to `NaN`; retained rows | Negative concentrations are physically impossible; values >100 mg/L indicate sensor fault in surface waters |
| **`bod_mg_l` physical bounds check `[0.0, 150.0]`** | *Verified at runtime* | Set out-of-range values to `NaN`; retained rows | Negative BOD is impossible; values >150 mg/L represent undiluted industrial/sewage effluent rather than surface water |
| **`dissolved_oxygen_mg_l` physical bounds check `[0.0, 25.0]`** | *Verified at runtime* | Set out-of-range values to `NaN`; retained rows | Negative DO is impossible; values >25 mg/L exceed atmospheric thermal supersaturation limits (likely % saturation unit mix-up) |
| **`orthophosphate_mg_l` physical bounds check `[0.0, 50.0]`** | *Verified at runtime* | Set out-of-range values to `NaN`; retained rows | Negative phosphorus concentration is physically impossible |
| **`temperature_c` physical bounds check `[-5.0, 45.0]`** | *Verified at runtime* | Set out-of-range values to `NaN`; retained rows | Outside ambient European surface water climatic boundaries (sensor fault) |
| **`nitrogen_mg_l` physical bounds check `[0.0, 200.0]`** | *Verified at runtime* | Set out-of-range values to `NaN`; retained rows | Negative total nitrogen concentration is physically impossible |
| **`nitrate_mg_l` physical bounds check `[0.0, 200.0]`** | *Verified at runtime* | Set out-of-range values to `NaN`; retained rows | Negative nitrate concentration is physically impossible |
| **Distribution outlier evaluation** (high nutrient/organic readings within physical bounds) | Monitored across all 8 parameters | Retained all valid extreme observations (did not trim statistical outliers) | High concentrations represent real acute pollution or storm-runoff events that the model must learn to predict |
| **Uninformative constant metadata column** (`country` = `'Ireland'` for 100% of rows) | 235,019 | Dropped `country` column | Zero-variance constant feature carries zero predictive signal |
| **Column-level missingness audit** (>50–60% drop threshold) | 0 columns exceeded threshold (<0.1% missing post-bounds check) | Retained all physicochemical feature columns; remaining `NaN`s deferred to train-only `SimpleImputer` | Preserves all informative predictors while preventing data snooping during imputation |
| **Station-grouped train/test split** (`GroupShuffleSplit` on `site_id`) | 9,737 unique stations (80% Train / 20% Test) | Exported disjoint positional indices to `data/processed/split_indices.json` | Guarantees zero station overlap (`overlap = 0`) between train and test partitions to prevent spatial leakage |

---

## 2. Methodology Notes for Report & Viva (Section 5.5)

1. **Why Out-of-Range Values Were Set to `NaN` Instead of Deleting Rows:**  
   Deleting an entire monitoring record because a single sensor (e.g., a faulty pH probe) misfired discards 7 other valid chemical measurements from that water sample. Setting only the erroneous reading to `NaN` and imputing it later via `SimpleImputer(strategy="median", add_indicator=True)` (fitted strictly on `X_train`) preserves sample size and captures sensor-fault patterns via the missing-indicator column.
2. **Why Statistical Outliers Were Retained:**  
   In environmental hydrology, water quality parameters (`ammonia_mg_l`, `bod_mg_l`, `orthophosphate_mg_l`) follow heavy right-skewed distributions. Statistical outliers correspond to real agricultural runoff or sewage overflow events—the exact minority degraded cases (`Fair`, `Poor`, `Marginal`) our models need to detect. Rather than deleting them, we apply `np.log1p` transforms in `src/features.py`.
