# Target Definition & Leakage Mitigation Agreement (Section 5.2)

- **Project:** Surface Water Quality Modeling & Prediction
- **Data Lead:** Sarthak
- **Modeling Lead:** Priyal
- **Dataset:** Ireland Regional Subset (`data/raw/Ireland_dataset.csv`)
- **Total Records:** 235,019 observations across 14 raw columns
- **Date:** Wednesday, 21 October 2026

---

## 1. Target Variable Specifications & Distributions

The selected dataset provides both continuous and categorical representations of the Canadian Council of Ministers of the Environment (CCME) Water Quality Index (WQI).

### 1.1 Continuous Target (Regression): `CCME_Values`
The continuous index scores range on a standard scale from 0 to 100, where higher values denote superior water quality.
- **Count:** 235,019
- **Mean:** 98.11
- **Standard Deviation:** 6.41
- **Min:** 37.19
- **25th Percentile:** 100.00
- **Median (50th Percentile):** 100.00
- **75th Percentile:** 100.00
- **Max:** 100.00
- **Distribution Insight:** Over 75% of all historical monitoring observations in Ireland exhibit an upper-bound score of 100.00 (pristine water meeting all statutory thresholds). Models evaluated with standard Mean Squared Error (MSE) risk predicting near 100 everywhere.

### 1.2 Categorical Target (Classification): `CCME_WQI`
The regulatory quality tier derived from the underlying CCME score:
- **Excellent:** 84.24% (197,980 samples)
- **Good:** 14.07% (33,067 samples)
- **Fair:** 0.69% (1,622 samples)
- **Poor:** 0.61% (1,434 samples)
- **Marginal:** 0.39% (916 samples)

### 1.3 Modeling Guidance for Priyal & Anusha
- **Severe Class Imbalance:** Clean classes (`Excellent` + `Good`) account for 98.31% of data. The three degraded tiers (`Fair`, `Marginal`, `Poor`) comprise only 1.69% combined.
- **Evaluation Metric Constraint:** Raw classification accuracy is an invalid benchmark (a naive majority classifier achieves 84.24% accuracy with 0% recall on contaminated water). Models must be evaluated using **Macro F1-Score**, **Balanced Accuracy**, and **Precision-Recall AUC (PR-AUC)**.
- **Recommended Classification Formulations:**
  1. *Binary Reformulation (Primary):* Group classes into `Pristine / Excellent` (84.24%) vs. `Impacted / Below-Excellent` (15.76%). This provides a viable operational classification boundary that can be easily balanced via `class_weight='balanced'`.
  2. *Multi-Class Alternative:* 5-tier classification utilizing cost-sensitive learning or synthetic minority oversampling (SMOTE) on train folds only.

---

## 2. Zero-Leakage Parameter Partitioning Plan

### The Core Problem (Section 5.2 Protocol)
Because `CCME_Values` is calculated using a deterministic environmental index formula derived from physicochemical concentrations, passing all 8 parameters into feature matrix $X$ would result in complete mathematical formula leakage. Models would merely perform algebraic curve-fitting rather than learning genuine environmental relationships.

To satisfy Section 5.2 (Option B), the 8 monitored parameters are partitioned into strictly non-overlapping subsets:

| Column Name | Assigned Pipeline Role | Justification |
| :--- | :--- | :--- |
| `CCME_Values` | **Target ($y_{\text{reg}}$)** | Continuous regression target (0–100). |
| `CCME_WQI` | **Target ($y_{\text{clf}}$)** | Categorical classification target. |
| `Biochemical Oxygen Demand (mg/l)` | **EXCLUDED (Dropped from $X$)** | Core driver of CCME organic violation scores; omitted to prevent direct equation inversion. |
| `Dissolved Oxygen (mg/l)` | **EXCLUDED (Dropped from $X$)** | Primary aquatic survival metric; omitted to prevent direct equation inversion. |
| `Nitrate (mg/l)` | **EXCLUDED (Dropped from $X$)** | Key nutrient violation metric; omitted to prevent direct equation inversion. |
| `pH (ph units)` | **Feature ($X$)** | Acid-base balance; standard ambient physicochemical input. |
| `Temperature (cel)` | **Feature ($X$)** | Thermal regime and seasonal physical proxy. |
| `Ammonia (mg/l)` | **Feature ($X$)** | Agricultural runoff and waste indicator. |
| `Orthophosphate (mg/l)` | **Feature ($X$)** | Bioavailable phosphorus nutrient proxy. |
| `Nitrogen (mg/l)` | **Feature ($X$)** | Total nitrogen organic/inorganic nutrient balance. |
| `Waterbody Type` | **Feature ($X$)** | Environmental context (River, Lake, Transitional, Coastal); one-hot encoded. |
| `Date` | **Temporal Feature Generator** | Used in `src/features.py` to derive cyclical features (`month_sin`, `month_cos`) and season. |
| `Area` | **Grouping Key ONLY** | Contains 9,737 unique station IDs; reserved exclusively for `GroupShuffleSplit` across folds. Strictly excluded from $X$. |
| `Country` | **Metadata (Dropped)** | Uniform constant (`Ireland`), dropped during data cleaning. |

---

## 3. Mandatory Leakage Acknowledgment Statement (For Final Report)

> *"To prevent direct algebraic target leakage from the Canadian Council of Ministers of the Environment (CCME) Water Quality Index calculation, parameters heavily driving the target formulation—specifically Biochemical Oxygen Demand (BOD), Dissolved Oxygen (DO), and Nitrate—are strictly excluded from the predictive feature matrix $X$. Models predict overall water quality index scores strictly from secondary physicochemical proxies (pH, Temperature, Ammonia, Orthophosphate, Nitrogen), waterbody classification, and seasonal indicators derived from sampling dates. While direct mathematical formula leakage is eliminated, residual indirect correlation remains due to natural hydro-chemical coupling across environmental variables (e.g., oxygen solubility inversely scaling with water temperature)."*

---

## 4. Train-Test Split Protocol (Scheduled for Thursday 22 Oct)

- **Strategy:** Station-level grouped validation (`GroupShuffleSplit`).
- **Grouping Column:** `Area` (9,737 unique monitoring stations).
- **Split Ratio:** 80% Train, 20% Test (`test_size=0.20`, `random_state=42`).
- **Guarantee:** No monitoring station appearing in the training set will appear in the test set, ensuring models are evaluated on spatial generalizability to unseen geographical sites.
- **Output Artifacts:** Stored in `data/processed/clean.csv` and `data/processed/split_indices.json`.
