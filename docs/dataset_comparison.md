# Dataset comparison

Every candidate we examined, with the verdict. Final choice: **candidate 2, the dataset linked in the project brief**, using England, Ireland, USA and Canada.

| # | Dataset | Rows | Columns | Site id | Date | Ready target | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Cork Harbour WQI (figshare) | 7,790 | 16 | no (unique name per row) | year + month name | yes (`WQI Value`) | **Backup.** Learnable (R² 0.70, time split) but small and one location |
| 2 | **Comprehensive Surface Water Quality 1940–2023 (figshare), linked in the brief** | 2,827,977 | 14 | yes (~61,700 sites) | yes | yes (`CCME_Values`) | **Selected.** Needs real cleaning and a leakage-safe feature set (see below) |
| 3 | Kaggle Water Potability | 3,276 | 10 | no | no | yes (binary) | Rejected: no site/date, so no fair split |
| 4 | UNEP GEMStat (Zenodo) | 80 files, ~0.7–1.1M rows each | 13 (long format) | yes | yes | no | Rejected: no WQI, heavy wrangling |
| 5 | Kaggle "Water Quality" | 7,999 | 21 | no | no | yes (binary) | Rejected: unclear origin, no site/date, corrupt rows |

## Why candidate 2, and what it needed

- It is the dataset named in the brief, so no complexity argument is needed. It has site ids and full dates, which allow the station split and the time split the brief asks for.
- **Problems found, and how they were fixed (`notebooks/02_cleaning.ipynb`, `docs/cleaning_log.md`):**
  - About 30% (29.9%) of England is effluent, sewage or drainage water: dropped.
  - Many values are filled-in placeholders (one repeated value per country and water type): set to NaN.
  - Duplicate rows, impossible temperatures and pH: removed or set to NaN.
  - Records before 2000 (different methods, sparse): dropped.
  - Rows with fewer than 5 of 8 real parameters (most of the USA): dropped.
  - The target is computed from 8 parameters, so models only see dissolved oxygen, pH and temperature plus context (`docs/target_definition.md`).

## Countries in the file

| Country | Raw rows | In `final_dataset.csv` | Notes |
|---|---|---|---|
| England | 2,129,198 | 180,082 (4,413 sites) | Core of the dataset; 29.9% was effluent/sewage |
| Ireland | 235,019 | 70,282 (3,691 sites) | 82% of target is at the maximum of 100; about a third of dissolved oxygen was a filler |
| USA | 413,814 | 6,810 (94 sites) | Mostly filled-in values (BOD 100% one value), so few usable rows |
| Canada | 3,949 | 2,600 (1,742 sites) | Small |
| China | 45,997 | 0 | Single monitoring site, so no site split is possible |

Whole sites were sampled (not single rows) so the file stays under GitHub's size limit (34.8 MB).
