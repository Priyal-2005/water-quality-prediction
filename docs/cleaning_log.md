# Data cleaning log

- **Raw input:** `data/raw/Combined_dataset.csv` (England, USA, Ireland, Canada)
- **Output:** `data/processed/final_dataset.csv`
- **Code:** `build_final_dataset()` in `src/data.py`, run from `notebooks/02_cleaning.ipynb`
- Generated automatically, so every number below is from the real run (seed 42).

| Issue found | How many rows/values | Action | Why |
| :--- | :--- | :--- | :--- |
| Rows loaded for the chosen countries | 2,781,980 {'England': 2129198, 'USA': 413814, 'Ireland': 235019, 'Canada': 3949} | Kept countries ['England', 'USA', 'Ireland', 'Canada'] from Combined_dataset.csv | Other countries are not used (China has a single site, so no site split is possible). |
| Unparseable dates (dd-mm-yyyy) | 0 | None found | Date is needed for the time split and seasonal features. |
| Non-surface water types (effluent, sewage, drainage) | 635,532 {'Effluent': 601550, 'Sewage': 23777, 'Drainage': 10205} | Dropped rows | The task is surface-water quality; effluent/sewage has a different chemistry and was ~28% of England. |
| Records before 2000 | 243089 | Dropped rows | England only has data from 2000; older USA/Ireland records used different methods and are sparse. |
| Exact duplicate rows across all columns | 7277 | Removed duplicate rows and reset the index | Exact duplicates can over-weight repeated observations and complicate validation. |
| ph physical bounds check [0.0, 14.0] | 503 | Set out-of-range values to NaN; retained rows | Outside the physical pH scale; instrument error |
| temperature_c physical bounds check [-5.0, 40.0] | 222 | Set out-of-range values to NaN; retained rows | Surface water outside -5..40 C is a unit error (e.g. Fahrenheit) or a fault |
| dissolved_oxygen_mg_l physical bounds check [0.0, 25.0] | 0 | No out-of-range values found | Above ~25 mg/L is not physically possible in surface water |
| bod_mg_l physical bounds check [0.0, 150.0] | 3564 | Set out-of-range values to NaN; retained rows | Above 150 mg/L looks like effluent, not surface water |
| ammonia_mg_l physical bounds check [0.0, 100.0] | 647 | Set out-of-range values to NaN; retained rows | Negative or extreme concentrations are errors |
| orthophosphate_mg_l physical bounds check [0.0, 50.0] | 198 | Set out-of-range values to NaN; retained rows | Negative or extreme concentrations are errors |
| nitrogen_mg_l physical bounds check [0.0, 200.0] | 0 | No out-of-range values found | Negative or extreme concentrations are errors |
| nitrate_mg_l physical bounds check [0.0, 200.0] | 0 | No out-of-range values found | Negative or extreme concentrations are errors |
| ammonia_mg_l: repeated placeholder value (filled in by the dataset authors) | 728,057 (38.4% of rows) | Set to NaN (imputed later, train-only, with a missing-indicator) | A value repeated for 5%+ of a country/water-type group, 2.5x more than any other (checked up to 3 times), is not a measurement. |
| bod_mg_l: repeated placeholder value (filled in by the dataset authors) | 1,117,647 (58.9% of rows) | Set to NaN (imputed later, train-only, with a missing-indicator) | A value repeated for 5%+ of a country/water-type group, 2.5x more than any other (checked up to 3 times), is not a measurement. |
| dissolved_oxygen_mg_l: repeated placeholder value (filled in by the dataset authors) | 663,570 (35.0% of rows) | Set to NaN (imputed later, train-only, with a missing-indicator) | A value repeated for 5%+ of a country/water-type group, 2.5x more than any other (checked up to 3 times), is not a measurement. |
| orthophosphate_mg_l: repeated placeholder value (filled in by the dataset authors) | 259,239 (13.7% of rows) | Set to NaN (imputed later, train-only, with a missing-indicator) | A value repeated for 5%+ of a country/water-type group, 2.5x more than any other (checked up to 3 times), is not a measurement. |
| ph: repeated placeholder value (filled in by the dataset authors) | 313,104 (16.5% of rows) | Set to NaN (imputed later, train-only, with a missing-indicator) | A value repeated for 5%+ of a country/water-type group, 2.5x more than any other (checked up to 3 times), is not a measurement. |
| temperature_c: repeated placeholder value (filled in by the dataset authors) | 200,807 (10.6% of rows) | Set to NaN (imputed later, train-only, with a missing-indicator) | A value repeated for 5%+ of a country/water-type group, 2.5x more than any other (checked up to 3 times), is not a measurement. |
| nitrogen_mg_l: repeated placeholder value (filled in by the dataset authors) | 267,856 (14.1% of rows) | Set to NaN (imputed later, train-only, with a missing-indicator) | A value repeated for 5%+ of a country/water-type group, 2.5x more than any other (checked up to 3 times), is not a measurement. |
| nitrate_mg_l: repeated placeholder value (filled in by the dataset authors) | 590,747 (31.2% of rows) | Set to NaN (imputed later, train-only, with a missing-indicator) | A value repeated for 5%+ of a country/water-type group, 2.5x more than any other (checked up to 3 times), is not a measurement. |
| Rows with fewer than 5 of 8 genuine parameters or fewer than 2 of 3 genuine sensor readings | 474,227 {'England': 243583, 'USA': 158866, 'Ireland': 71403, 'Canada': 375} | Dropped rows | Mostly filled-in rows: their CCME target is computed from constants, so they teach the model nothing real. This removes most USA rows. |
| Site identifiers | 41295 | Prefixed with country so ids are unique across countries | Needed so the site split cannot mix two stations that share a name. |
| File size (GitHub limit) | 1,421,855 -> 259,774 rows, 9,940 sites | Random sample of whole sites per country, caps {'England': 180000, 'Ireland': 70000, 'USA': None, 'Canada': None}, seed 42 | A full file would be hundreds of MB; whole-site sampling keeps each station's history intact. |
| Station split (main): train / val / test | 181,747 / 26,054 / 51,973 rows; 6,583 / 1,136 / 2,221 sites | Stations shuffled (seed 42) and dealt out until 70% / 10% / 20% of rows are reached, stored in the `split` column | No station is in more than one set. Validation is for tuning and early stopping; the test set is touched once. |
| Time split (secondary): train / val / test | 185,094 / 21,499 / 53,181 rows | Train up to 2015, validation 2016-2017, test 2018+, stored in `split_time` | Checks that the model still works on later years. |
| Output | 259,774 rows x 19 columns, 34.7 MB | Saved final_dataset.csv | The single processed dataset used by every later step. |

## Filled-in values found for the model inputs

Values repeated for a large share of a country/water-type group (and far more common than any other value) are placeholders, not measurements. They were set to NaN. Same pattern was found and removed for the five held-out parameters (counts in the table above).

| Parameter | Country | Repeated value | Rows set to NaN | Water types |
| :--- | :--- | ---: | ---: | :--- |
| dissolved_oxygen_mg_l | Canada | 9.824 | 505 | Lake, River |
| dissolved_oxygen_mg_l | Canada | 11.6841 | 184 | Lake, River |
| dissolved_oxygen_mg_l | Canada | 10.89 | 1 | Coastal |
| dissolved_oxygen_mg_l | England | 10.2 | 360,401 | Estuarine, Lake, Marine, River, Sea Water |
| dissolved_oxygen_mg_l | England | 11.6841 | 112,516 | Estuarine, Lake, Marine, River, Sea Water |
| dissolved_oxygen_mg_l | Ireland | 11.6841 | 88,371 | Coastal, Lake, River, Transitional |
| dissolved_oxygen_mg_l | Ireland | 4.9725 | 630 | Coastal, Transitional |
| dissolved_oxygen_mg_l | USA | 9.87 | 92,424 | Lake, Marine, River |
| dissolved_oxygen_mg_l | USA | 11.6841 | 8,538 | Lake, Marine, River |
| ph | Canada | 7.79 | 995 | Lake, River |
| ph | Canada | 6.1517 | 1 | Coastal |
| ph | England | 7.78 | 190,725 | Lake, Marine, River |
| ph | USA | 7.8 | 121,383 | Lake, Marine, River |
| temperature_c | Canada | 12 | 264 | Lake, River, Transitional |
| temperature_c | Canada | 8.35 | 1 | Coastal |
| temperature_c | England | 11.46 | 108,213 | Lake, Marine, River, Sea Water |
| temperature_c | Ireland | 11.15 | 11,433 | Coastal, River, Transitional |
| temperature_c | USA | 11.1 | 80,896 | Lake, Marine, River |
