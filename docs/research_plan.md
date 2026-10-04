# TuneLab: Research Plan
## When Does Fine-Tuning Actually Beat RAG?

---

### 1. Core Research Question

Under what specific task, data, and schema conditions does Supervised Fine-Tuning (LoRA / QLoRA) outperform Retrieval-Augmented Generation (RAG) for domain-specific Text-to-SQL tasks with Small Language Models (SLMs), and when is RAG the more practical and cost-effective strategy?

Specifically:
1. **Data scale threshold**: How does increasing domain-specific training volume affect the performance of fine-tuned SLMs relative to few-shot and RAG-augmented baseline SLMs, and does an empirical crossover point exist?
2. **Noise robustness**: How resilient are parameter-adapted models (LoRA/QLoRA) versus context-augmented models (RAG) when training or retrieval corpora contain corrupted SQL syntax or schema mismatch noise?
3. **Inference vs. Adaptation Cost**: What are the empirical trade-offs in inference latency, VRAM utilization, and execution accuracy across Prompting, RAG, LoRA, and QLoRA?

---

### 2. Testable Hypotheses

* **$H_1$ (Data Scaling & Adaptation Crossover)**: As the volume of domain-specific training data increases, the Execution Accuracy of parameter-adapted models (LoRA/QLoRA) improves at a steeper rate than RAG, exhibiting an empirical crossover point where fine-tuning matches and subsequently exceeds RAG performance on the same evaluation split.
* **$H_2$ (Quantization Trade-off)**: 4-bit quantized fine-tuning (QLoRA) substantially reduces peak VRAM consumption compared to 16-bit LoRA, while incurring only a marginal loss in downstream Execution Accuracy and SQL Validity.
* **$H_3$ (Exemplar Retrieval Return)**: In RAG, increasing the number of retrieved exemplars ($K$) yields diminishing marginal improvements in Execution Accuracy, eventually plateauing or degrading as added context introduces distraction or increases prompt length overhead.
* **$H_4$ (Noise Sensitivity & Robustness)**: Supervised fine-tuning (LoRA/QLoRA) exhibits higher sensitivity to supervision noise (corrupted SQL syntax and schema inconsistencies) than RAG, with fine-tuned models showing a steeper rate of performance degradation as the training noise ratio increases.
* **$H_5$ (Error Mode Asymmetry)**: Prompting/RAG and fine-tuning produce distinct error distributions: context-augmented models suffer disproportionately from schema-linking failures (referencing incorrect tables or columns from complex schemas), whereas fine-tuned models suffer disproportionately from structural and clause logic failures (joins, grouping, or operator misapplication).

---

### 3. Dataset Specification & Source

#### A. Documented Facts from Official Spider v1.0 Release
The following properties are established directly from the published paper (Yu et al., EMNLP 2018) and the official Yale LILY Lab release:
* **Benchmark**: Spider v1.0 ("A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL").
* **Official Upstream Source**: Yale LILY Lab (`https://yale-lily.github.io/spider`, release `spider.zip`).
* **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
* **Published Dataset Totals**: 10,181 natural language questions and 5,693 unique complex SQL queries across 200 databases spanning 138 domains.
* **Public Release Split Structure**:
  * `train_spider.json`: 7,000 question-SQL pairs.
  * `train_others.json`: 1,659 question-SQL pairs adapted from prior datasets (Restaurants, GeoQuery, Scholar, Academic, IMDB, Yelp).
  * `dev.json`: 1,034 question-SQL pairs across 20 databases.
  * `tables.json`: Foreign key and column schema metadata for all public databases.
  * `database/`: Subdirectories containing SQLite `.sqlite` database files.
* **Cross-Database Split Rule**: The 20 databases in `dev.json` have zero overlap with databases in the training splits. This enforces cross-schema generalization rather than table memorization.
* **Test Set Policy**: The official Spider test set (40 databases, 2,147 queries) is private and hosted on Yale's evaluation server. Consequently, published literature using Spider offline uses `dev.json` as the evaluation benchmark.

#### B. Assumptions Requiring Verification Upon Dataset Ingestion
The following details cannot be assumed without inspection of the actual unzipped files once available:
1. **Inclusion of `train_others.json`**: Whether to train on `train_spider.json` alone (7,000 samples) or combined with `train_others.json` (8,659 samples) must be controlled and verified.
2. **SQLite Schema Consistency**: Whether foreign keys and primary keys are declared via DDL inside the `.sqlite` files themselves or only specified in `tables.json`.
3. **Execution Environment Compatibility**: Ensuring all `.sqlite` database files load cleanly in Python's standard `sqlite3` without dialect mismatches or missing table files.
4. **Hardness Distribution**: The exact proportion of easy, medium, hard, and extra-hard queries in the active evaluation split.

#### C. Offline Environment Strategy
* The current local development sandbox is air-gapped without outbound network access. `spider.zip` cannot be downloaded directly.
* We provide the official download specification in `scripts/download_spider.py` and document setup in `data/README.md`.
* No fake dataset files or fabricated statistics will be created. Until the official archive is unzipped and inspected, development will use verified, self-contained fixtures matching the Spider format for Tier A/B pipeline tests, and mark full Tier C benchmarks as `PENDING`.

---

### 4. Model Selection & Specifications

* **Primary Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
  * Architecture: 1.54B parameters, 28 layers, 16 attention heads, rotary position embeddings (RoPE), SwiGLU activation.
  * Context Window: 32,768 tokens (eval capped at 1,024 for compute efficiency).
  * License: Apache 2.0.
  * Selection Justification: Dedicated code/SQL pre-training and instruction tuning. High baseline SQL syntax validity at low parameter scale.
* **Development / Low-Resource Alternative**: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (0.49B parameters).

---

### 5. RAG Design & Architecture

RAG in Text-to-SQL targets **Exemplar Retrieval** and **Schema Linking**:
* **Retrieval Targets**:
  1. Training Exemplars: (Question, Gold SQL, Target Schema).
  2. Schema Documentation: Relevant table schemas and foreign key relations.
* **Embedding Model (Connected / Research Setup)**:
  * Model: `BAAI/bge-small-en-v1.5` (33M params, 384-dimensional embeddings, L2 normalized).
  * Inference: SentenceTransformers / Hugging Face Transformers.
* **Offline / Development Fallback Setup**:
  * Primary: **Okapi BM25** lexical retriever implemented natively in Python/NumPy over tokenized questions.
  * Hybrid: Reciprocal Rank Fusion (RRF) combining BM25 lexical scores with dense cosine similarity when embeddings are pre-computed.
* **Retrieval Leakage Guard**:
  * Retrieval corpus is built **exclusively from the training split**.
  * No validation or test question, SQL, or database metadata is indexed in the retrieval pool.

---

### 6. Fine-Tuning Design (LoRA & QLoRA)

* **Framework**: Hugging Face `peft` + `transformers` + `accelerate` (using `bitsandbytes` for 4-bit quantization).
* **LoRA Target Modules**: Linear projections in self-attention and MLP blocks: `q_proj`, `v_proj`, `k_proj`, `o_proj`.
* **Hyperparameters**:
  * Rank: $r \in \{4, 8, 16\}$ (default $r=8$).
  * Alpha: $\alpha = 2 \times r$ ($\alpha=16$ for $r=8$).
  * Dropout: $0.05$.
  * Optimizer: AdamW, cosine learning rate schedule with warmup (warmup ratio $0.03$), peak learning rate $2 \times 10^{-4}$.
  * Effective Batch Size: 16 (per-device batch size 2, gradient accumulation steps 8).
  * Sequence Length: 1,024 tokens.
* **QLoRA Details**:
  * Base model weights loaded in 4-bit NormalFloat (NF4).
  * Double quantization enabled.
  * Compute dtype: `bfloat16` (or `float16` if hardware lacks native bfloat16 support).

---

### 7. Variables & Experimental Controls

* **Independent Variables**:
  1. *Approach*: Prompting (Zero-shot, Few-shot), RAG ($K \in \{1, 3, 5, 10\}$), LoRA, QLoRA.
  2. *Training Data Fraction*: $1\%, 5\%, 10\%, 25\%, 50\%, 100\%$.
  3. *LoRA Rank*: $r=4, 8, 16$.
  4. *Data Noise Ratio*: $0\%, 5\%, 10\%, 20\%$ corrupted training samples.
* **Dependent Variables**:
  1. *Execution Accuracy (EX)*: Fraction of queries whose execution matches ground truth on SQLite.
  2. *Exact Match (EM)*: AST / normalized string equivalence.
  3. *SQL Validity*: Fraction of generated SQL strings that execute without database error.
  4. *Inference Latency*: End-to-end execution time per sample (ms).
  5. *Peak VRAM*: Maximum memory consumed during training and inference (MB).
  6. *Trainable Parameter Ratio*: $\frac{\text{Trainable Params}}{\text{Total Base Params}}$.
* **Controlled Variables**:
  * Base model checkpoint (`Qwen2.5-Coder-1.5B-Instruct`).
  * Evaluation dataset split (Spider dev set, strictly untouched).
  * Prompt template structure (System instruction, Database Schema formatting, Output boundary).
  * Execution timeout (5.0 seconds).
  * Random seeds (42, 1337, 2026).

---

### 8. Confounders & Threat Mitigation

* **Schema Formats**: Prompt format discrepancies between RAG and fine-tuning can bias results. *Mitigation*: All methods receive the identical schema serialization format.
* **SQLite Non-determinism**: Queries without `ORDER BY` may produce non-deterministic row ordering. *Mitigation*: Unordered query results are compared as multiset bags (using hash/Counter).
* **Data Leakage**: Database instances in dev must not appear in the training split. *Mitigation*: Spider's cross-database split strictly enforces that all 20 dev databases are unseen during training and retrieval index construction.

---

### 9. Detailed Metric Definitions & SQLite Edge Cases

1. **SQL Validity**:
   * Evaluated by executing the generated query $Q_{pred}$ inside an isolated SQLite cursor with a transaction rollback.
   * Return 1 if execution terminates with an execution result set (even if empty).
   * Return 0 if `sqlite3.Error` (e.g. `OperationalError: syntax error`, `no such column`, `no such table`) is raised or execution times out.
2. **Exact Match (EM)**:
   * Both $Q_{pred}$ and $Q_{gold}$ are lowercased, whitespace is collapsed, semicolon stripped, and SQL keywords normalized.
   * Binary 1 if string match holds, else 0.
3. **Execution Accuracy (EX)**:
   * Let $R_{pred}$ and $R_{gold}$ be the fetched lists of row tuples from executing $Q_{pred}$ and $Q_{gold}$ on the same SQLite database.
   * If reference query $Q_{gold}$ contains an explicit `ORDER BY` clause:
     $$EX = 1 \iff R_{pred} = R_{gold}$$ (strict ordered list comparison).
   * If reference query $Q_{gold}$ does NOT contain `ORDER BY`:
     $$EX = 1 \iff \text{Bag}(R_{pred}) = \text{Bag}(R_{gold})$$ (multiset equality preserving row multiplicity).
4. **Handling NULL Values**:
   * SQLite `NULL` maps to Python `None`. Equality holds iff corresponding tuple elements are both `None`.
5. **Numeric Precision Tolerance**:
   * Numeric floats are compared with relative and absolute tolerance ($10^{-4}$):
     $$|a - b| \le 10^{-4} \times \max(|a|, |b|) + 10^{-5}$$.
6. **Execution Timeouts**:
   * Each query execution has a strict 5.0s timeout. If exceeded, marked as timeout failure ($EX=0, Validity=0$).

---

### 10. Experiment Tiers

* **Tier A: Pipeline Validation (Smoke Tests)**:
  * Purpose: End-to-end execution of schema parser, prompt builder, SQLite executor, metric calculator, and result logger on 3 verified test cases. Runs in $< 2$ seconds.
* **Tier B: Pilot Experiment**:
  * Purpose: Run Zero-shot vs Few-shot ($K=3$) vs RAG ($K=3$) on 30 test cases across 3 distinct database schemas (e.g., single table, join, aggregation). Validates pipeline integrity and produces real baseline result metrics.
* **Tier C: Full Research Matrix**:
  * Run across full Spider dev set with LoRA/QLoRA training on full and fractioned data splits ($1\% - 100\%$), rank ablations, and noise experiments. Dependent on GPU compute.

---

### 11. Threats to Validity

1. **Synthetic vs Real-world Workloads**: Spider queries are clean and academic; real enterprise SQL includes dirty dialects and stored procedures.
2. **Database Engine Specifics**: Evaluation is performed on SQLite. Differences with PostgreSQL/MySQL (e.g. window functions, full outer joins) are out of scope.
3. **Hardware Heterogeneity**: VRAM and latency metrics depend on specific hardware and CUDA versions; hardware specs must be reported alongside every result.
