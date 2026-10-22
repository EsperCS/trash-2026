# Data Audit Report


## train.csv — Basic Info

- **Rows:** 4,000
- **Columns:** 3
- **Column names:** `Id`, `text`, `label`

| Column | Dtype | Nulls | Null % |
|--------|-------|-------|--------|
| `Id` | int64 | 0 | 0.00% |
| `text` | str | 0 | 0.00% |
| `label` | str | 0 | 0.00% |


## train.csv — Duplicate Analysis

- **Exact duplicate rows:** 0
- **Duplicate `text` values:** 0
- **Near-duplicate `text` (lowered+stripped):** 0


## train.csv — Class Distribution

| Label | Count | Proportion |
|-------|-------|------------|
| `benign` | 2,401 | 0.6002 |
| `jailbreak` | 1,599 | 0.3997 |


## train.csv — Text Length Statistics

**Char length**

- mean: 711.7
- std:  821.5
- min:  11
- 25%:  130
- 50%:  616
- 75%:  918
- 95%:  1814
- max:  11977

**Word count**

- mean: 113.3
- std:  127.4
- min:  2
- 25%:  23
- 50%:  97
- 75%:  145
- 95%:  282
- max:  1939


## test.csv — Basic Info

- **Rows:** 1,000
- **Columns:** 2
- **Column names:** `Id`, `text`

| Column | Dtype | Nulls | Null % |
|--------|-------|-------|--------|
| `Id` | int64 | 0 | 0.00% |
| `text` | str | 0 | 0.00% |


## test.csv — Duplicate Analysis

- **Exact duplicate rows:** 0
- **Duplicate `text` values:** 0
- **Near-duplicate `text` (lowered+stripped):** 0


## test.csv — Class Distribution

_No label column detected._


## test.csv — Text Length Statistics

**Char length**

- mean: 709.7
- std:  791.1
- min:  24
- 25%:  150
- 50%:  606
- 75%:  908
- 95%:  1895
- max:  6766

**Word count**

- mean: 113.2
- std:  124.5
- min:  4
- 25%:  26
- 50%:  96
- 75%:  141
- 95%:  300
- max:  1046
