# TuneLab Data Directory

This directory stores the Text-to-SQL benchmark datasets, schemas, and experimental splits.

---

## 1. Verified Facts from Official Spider v1.0 Release
The following properties are directly verified from the published benchmark paper (Yu et al., EMNLP 2018) and Yale LILY Lab release:
- **Benchmark**: Spider v1.0
- **Authors**: Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, Dragomir Radev (Yale University).
- **Paper**: EMNLP 2018 ("Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL")
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Official URL**: https://yale-lily.github.io/spider
- **Published Benchmark Scale**: 10,181 natural language questions and 5,693 unique complex SQL queries across 200 databases spanning 138 domains.
- **Public Archive (`spider.zip`) File Manifest**:
  - `train_spider.json`: 7,000 question-SQL pairs on Spider databases.
  - `train_others.json`: 1,659 question-SQL pairs adapted from prior datasets (Restaurants, GeoQuery, Scholar, Academic, IMDB, Yelp).
  - `dev.json`: 1,034 question-SQL pairs on 20 unseen evaluation databases.
  - `tables.json`: Structural metadata (column names, types, primary keys, foreign keys).
  - `database/`: Subdirectories holding `.sqlite` files for each public database.
- **Cross-Database Split Rule**: The 20 databases in `dev.json` have zero overlap with databases in `train_spider.json` or `train_others.json`.

---

## 2. Assumptions Requiring Verification Upon Ingestion
These items cannot be treated as verified until the actual dataset archive is unzipped and inspected locally:
1. **Training Set Composition**: Whether the baseline training set should be `train_spider.json` alone (7,000 samples) or combined with `train_others.json` (8,659 samples) must be explicitly configured and reported.
2. **SQLite DDL Foreign Keys**: Whether table constraints (e.g. FOREIGN KEY clauses) exist inside the SQLite files or reside purely in `tables.json`.
3. **Database File Integrity**: Verifying that all SQLite database files load cleanly in Python's standard `sqlite3` without table corruption or dialect incompatibility.
4. **Hardness Breakdown**: The exact proportion of easy, medium, hard, and extra-hard questions across active splits.

---

## 3. Directory Layout
- `data/raw/`: Target location for extracted `spider/` folder (`database/`, `tables.json`, `train_spider.json`, `dev.json`).
- `data/processed/`: Standardized question-SQL pairs and serialized schema representations.
- `data/splits/`: Controlled subset splits (1%, 5%, 10%, 25%, 50%, 100%) and noise-corrupted splits.

---

## 4. Offline Environment Handling
The current development environment is air-gapped without external network access, so the ~100MB `spider.zip` archive cannot be downloaded dynamically. 

* **Policy**: Do not fabricate fake Spider datasets or invent synthetic statistics.
* **Testing**: Pipeline validation and unit tests will run against self-contained, human-verified fixtures strictly formatted to Spider's specifications. Full research runs are marked `PENDING` until the actual archive is mounted.
