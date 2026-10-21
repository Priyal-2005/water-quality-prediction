# Dataset Candidate Comparison Table (Section 5.1)

| Evaluation Metric | Candidate 1: Plan A (Ireland Regional Subset) [SELECTED] | Candidate 2: Plan B (Cork Harbour WQI) [BACKUP] | Candidate 3: Kaggle Water Potability [REJECTED BASELINE] |
| :--- | :--- | :--- | :--- |
| **Source & Link** | [Figshare: Comprehensive Surface Water Quality (1940–2023)](https://doi.org/10.6084/m9.figshare.27800394) | [Figshare: Cork Harbour WQI Forecasting Dataset](https://doi.org/10.6084/m9.figshare.28184252) | [Kaggle: Water Potability (adityakadiwal)](https://www.kaggle.com/datasets/adityakadiwal/water-potability) |
| **Licence** | Creative Commons Attribution 4.0 International (CC BY 4.0) | Creative Commons Attribution 4.0 International (CC BY 4.0) | CC0: Public Domain |
| **Rows** | **235,019** | 7,790 | 3,276 |
| **Columns** | **14** | 16 | 10 |
| **Target Available? (Standard)** | **Yes:** `CCME_Values` (Continuous Canadian Council WQI index) & `CCME_WQI` (Categorical rating) | **Yes:** `WQI Value` (Continuous index) & `Label` (Categorical rating: Fair, Good, etc.) | **Yes:** `Potability` (Binary classification: 0 = Not Potable, 1 = Potable) |
| **% Missing per Column** | **0.0% missing across all 14 columns** (`Country`: 0.0%, `Area`: 0.0%, `Waterbody Type`: 0.0%, `Date`: 0.0%, `Ammonia`: 0.0%, `BOD`: 0.0%, `DO`: 0.0%, `Orthophosphate`: 0.0%, `pH`: 0.0%, `Temperature`: 0.0%, `Nitrogen`: 0.0%, `Nitrate`: 0.0%, `CCME_Values`: 0.0%, `CCME_WQI`: 0.0%) | **0.0% missing across all 16 columns** (`WaterbodyName`, `Years`, `SampleDate`, and all 11 physicochemical parameters: 0.0%) | `pH`: 14.99%, `Sulfate`: 23.84%, `Trihalomethanes`: 4.95%, all remaining 6 features: 0.0% |
| **Has Site/Station ID?** | **Yes:** `Area` (9,737 unique monitoring locations) and `Waterbody Type` (4 categories: River, Lake, Transitional, Coastal) | **Yes:** `WaterbodyName` (monitoring station identifier codes, e.g., `LEE (CORK)_020`) | **No:** Lacks station, sensor, or geographical site identifiers |
| **Has Date?** | **Yes:** `Date` column formatted as `dd-mm-yyyy` (spans historical multi-year records) | **Yes:** `Years` (numeric year) and `SampleDate` (month name) spanning 2007–2023 | **No:** Completely absent; records are static unanchored snapshots |
| **Units Documented?** | **Yes:** Documented in column names (`mg/l`, `cel`, `ph units`) and corresponding *Scientific Data* publication | **Yes:** Explicitly included in column headers (`as CaCO3`, `as N`, `mg/L`, `cel`, `µS/cm`) | **Yes:** Documented on Kaggle overview (mg/L, µS/cm, NTU) |
| **Leakage Risk Analysis** | **Medium:** The target (`CCME_Values`) is mathematically calculated from physical parameters. To prevent formula leakage, target-defining parameters (BOD, DO, Nitrate) are strictly omitted from feature inputs $X$. | **High:** The target `WQI Value` is directly derived from the 11 monitored parameters; predicting it from all 11 would allow trivial equation reconstruction. | **Low:** Target was independently measured/assigned, but dataset cannot be split appropriately. |
| **Final Recommendation & Verdict** | **SELECTED (Plan A):** Meets all project complexity criteria. High row count (235k), 0% missingness, complete parameter coverage, and 9,737 station IDs for leak-free `GroupShuffleSplit`. | **BACKUP (Plan B):** Strong secondary choice, but single estuarine harbor limits spatial generalizability compared to nationwide Irish monitoring. | **REJECTED:** Lacks site and temporal metadata, making station-based or temporal splitting impossible. Fails complexity standards. |

---

### Four Key Questions Evaluation for Plan A (Recommendation Verification)

1. **Is there a site or station ID?**  
   * **Yes.** The dataset provides `Area` (containing 9,737 unique monitoring stations across Ireland) and `Waterbody Type` (River, Lake, Transitional, Coastal). This enables a true station-separated validation split via `sklearn.model_selection.GroupShuffleSplit`.
2. **Is there a date?**  
   * **Yes.** The `Date` column provides timestamps in `dd-mm-yyyy` format across multiple years, enabling seasonal feature extraction and temporal alignment.
3. **Which of the 8 parameters are present and how much is missing?**  
   * **All 8 parameters are present:** Ammonia (mg/l), Biochemical Oxygen Demand (mg/l), Dissolved Oxygen (mg/l), Orthophosphate (mg/l), pH (ph units), Temperature (cel), Nitrogen (mg/l), and Nitrate (mg/l).
   * **Missingness:** 0.0% missing values across all parameters in the Ireland regional subset.
4. **Is there any WQI or class column?**  
   * **Yes.** Both `CCME_Values` (continuous target index) and `CCME_WQI` (discrete quality classes: Excellent, Good, Fair, Marginal, Poor) are provided.
