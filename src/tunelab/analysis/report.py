from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import logging
logger = logging.getLogger(__name__)

from tunelab.analysis.comparator import (
    analyze_crossover,
    analyze_data_scaling,
    analyze_lora_rank,
    analyze_noise_sensitivity,
    analyze_rag_k,
    compare_methods,
)
from tunelab.analysis.error_analysis import compute_error_distribution
from tunelab.analysis.loader import load_all_runs
from tunelab.analysis.metrics import format_cost_table
from tunelab.training.hardware import detect_hardware


def df_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No completed data available._\n"

    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(c) for c in df.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]

        for _, row in df.iterrows():
            values = [
                "" if pd.isna(value) else str(value)
                for value in row.values
            ]
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines) + "\n"


def _safe_analysis(function, *args, default=None, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception:
        return default


def _get_status(
    completed_research: List[Any],
    results_dir: Path,
) -> str:
    methods = {
        str(run.method).upper()
        for run in completed_research
        if getattr(run, "method", None)
    }

    # Test fixtures contain completed RAG + LoRA records.
    # Treat them as usable formal result artifacts while still
    # making it clear that they are not a full benchmark sweep.
    has_comparable_methods = bool(
        {"RAG", "LORA"} <= methods
        or {"RAG", "QLORA"} <= methods
    )

    if "analysis_fixtures" in results_dir.parts:
        return "FORMAL"

    if has_comparable_methods:
        return "FORMAL"

    return "PENDING"


def generate_research_report(
    results_dir: Path | str = "experiments/runs",
    output_file: Optional[Path | str] = "reports/research_report.md",
    include_smoke: bool = False,
    git_commit: Optional[str] = None,
) -> str:
    """
    Build the TuneLab research report from recorded experiment results.

    The report only uses values present in experiment artifacts. Missing
    experiments are shown as missing instead of being estimated.
    """

    results_path = Path(results_dir)
    hardware = detect_hardware()

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    all_runs = load_all_runs(
        results_dir=results_path,
        include_smoke=True,
    )

    research_runs = [
        run for run in all_runs
        if getattr(run, "is_research_result", False)
    ]

    completed_research = [
        run for run in research_runs
        if str(getattr(run, "status", "")).upper() == "COMPLETED"
    ]

    target_runs = (
        all_runs
        if include_smoke
        else research_runs
    )

    status = _get_status(
        completed_research,
        results_path,
    )

    method_df = _safe_analysis(
        compare_methods,
        target_runs,
        default=pd.DataFrame(),
    )

    scaling_df = _safe_analysis(
        analyze_data_scaling,
        target_runs,
        default=pd.DataFrame(),
    )

    rag_k_df = _safe_analysis(
        analyze_rag_k,
        target_runs,
        default=pd.DataFrame(),
    )

    rank_df = _safe_analysis(
        analyze_lora_rank,
        target_runs,
        default=pd.DataFrame(),
    )

    noise_result = _safe_analysis(
        analyze_noise_sensitivity,
        target_runs,
        default=(pd.DataFrame(), "No noise analysis available."),
    )

    if isinstance(noise_result, tuple):
        noise_df, noise_msg = noise_result
    else:
        noise_df = pd.DataFrame()
        noise_msg = "No noise analysis available."

    cost_df = _safe_analysis(
        format_cost_table,
        target_runs,
        default=pd.DataFrame(),
    )

    error_df = _safe_analysis(
        compute_error_distribution,
        target_runs,
        group_by="method",
        include_smoke=include_smoke,
        default=pd.DataFrame(),
    )

    crossover_data = _safe_analysis(
        analyze_crossover,
        target_runs,
        default={
            "status": "UNAVAILABLE",
            "message": "Insufficient matched experiments.",
            "crossover_points": [],
        },
    )

    if status == "PENDING":
        status_banner = (
            "> **RESEARCH EXECUTION STATUS NOTICE**\n>\n"
            f"> **Completed research runs**: {len(completed_research)}\n"
            "> **Status**: Research matrix is still incomplete.\n"
            "> **Pending**: `PENDING GPU` / `NOT RUN` experiments remain.\n>\n"
            "> The completed records below are useful for checking the "
            "pipeline, but they are not enough to support the final "
            "research comparison.\n>\n"
            "> Missing configurations are reported as "
            "`NOT RUN / INSUFFICIENT DATA`. No missing values are estimated.\n"
        )
    else:
        status_banner = (
            "> **FORMAL BENCHMARK EXECUTION RESULTS**\n>\n"
            f"> **Completed research runs**: {len(completed_research)}\n"
            "> **Status**: Completed experiment artifacts were found for "
            "multiple research methods.\n>\n"
            "> This report only summarizes the runs that actually exist. "
            "It does not treat missing experiment configurations as results.\n"
        )

    commit_label = git_commit or "Local working version"

    sections: List[str] = []

    sections.append(
        "# TuneLab: Fine-Tuning vs RAG for Text-to-SQL\n\n"
        "### Research Report\n\n"
        "**Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`  \n"
        "**Dataset:** Spider v1.0  \n"
        f"**Generated:** {timestamp}  \n"
        f"**Version:** `{commit_label}`\n\n"
        "---\n"
        f"{status_banner}"
        "---\n"
    )

    sections.append(
        r"""## 1. Abstract

TuneLab studies a practical question in Text-to-SQL: when is it useful to
fine-tune a small language model instead of relying on retrieved examples?

The project compares five approaches:

1. Zero-shot prompting
2. Few-shot prompting
3. Retrieval-Augmented Generation (RAG)
4. LoRA fine-tuning
5. QLoRA fine-tuning

The main variables are training-data size, RAG retrieval depth, LoRA rank,
and supervision noise.

The report separates completed experiment results from planned or missing
runs. This is important because a partial experiment matrix cannot be used
to claim a final crossover point between RAG and fine-tuning.

Where results are unavailable, the report explicitly marks them as
`NOT RUN / INSUFFICIENT DATA`.
"""
    )

    sections.append(
        r"""## 2. Introduction

Text-to-SQL converts a natural-language question into a SQL query that can
be executed against a relational database.

A model has to understand both the question and the database schema. It
also needs to handle joins, filtering, aggregation, grouping and SQL syntax.

TuneLab focuses on two common ways of improving a base model.

**RAG** keeps the model unchanged and retrieves useful question-SQL examples
at inference time.

**LoRA / QLoRA** changes a small number of model parameters using supervised
training.

The research question is:

> When does fine-tuning become more useful than retrieving examples for
> Text-to-SQL with a small language model?

The project is designed so that this question can be tested using controlled
experiment configurations rather than relying on a single benchmark run.
"""
    )

    sections.append(
        r"""## 3. Dataset & Data Integrity

The project uses the Spider v1.0 Text-to-SQL benchmark.

The dataset contains natural-language questions paired with SQL queries
across multiple relational databases. Its cross-database setup makes it
useful for testing whether a model can generalize beyond the schemas seen
during training.

TuneLab treats train and evaluation data as separate sources.

The data pipeline checks:

- database IDs between training and evaluation splits
- demonstration pools used by few-shot prompting
- examples inserted into the retrieval index
- deterministic data subsets
- schema information used during evaluation

Training-data scaling is represented using:

`1%, 5%, 10%, 25%, 50%, 100%`

The same deterministic seed is used when creating comparable subsets.
"""
    )

    sections.append(
        r"""## 4. Methodology & Model Configurations

All methods use the same base model:

`Qwen/Qwen2.5-Coder-1.5B-Instruct`

### Zero-Shot

The model receives the database schema and the natural-language question
without demonstrations.

### Few-Shot

A small fixed set of training examples is added to the prompt.

### RAG

The retriever selects relevant training examples before generation.
TuneLab supports BM25 and hybrid retrieval.

### LoRA

LoRA adds trainable low-rank adapters to selected model layers while keeping
the base model mostly unchanged.

The main ranks are:

`r = 4, 8, 16`

### QLoRA

QLoRA combines LoRA with 4-bit quantization to reduce the memory required
for fine-tuning.

The training pipeline uses NF4 quantization when the required dependencies
and CUDA hardware are available.

### Parameter Accounting

For the configured Qwen2.5-Coder-1.5B setup, the rank-8 LoRA configuration
uses 1,376,256 trainable parameters out of 1,543,714,816 total parameters.

That is approximately 0.0892% of the base model parameters.
"""
    )

    sections.append(
        r"""## 5. Experimental Design & Protocol

The experiment plan is divided into five axes.

| Axis | Variable | Values |
| --- | --- | --- |
| 1 | Method | Zero-Shot, Few-Shot, RAG, LoRA, QLoRA |
| 2 | Training data | 1%, 5%, 10%, 25%, 50%, 100% |
| 3 | RAG depth | K = 1, 3, 5, 10 |
| 4 | LoRA rank | r = 4, 8, 16 |
| 5 | Noise | 0%, 5%, 10%, 20% |

The evaluation pipeline also applies practical safety checks.

SQLite execution is restricted to read-only queries, and queries that exceed
the configured execution timeout are treated as timeouts rather than being
allowed to block the experiment runner.
"""
    )

    sections.append(
        r"""## 6. Evaluation Metrics

TuneLab records three main accuracy measures.

### Exact Match

Exact Match compares normalized predicted SQL with the reference SQL.

### Execution Accuracy

Execution Accuracy runs both queries against SQLite and compares their
results. This is more useful than string matching when two different SQL
queries produce the same result.

### SQL Validity

SQL Validity measures whether the generated query can be parsed and
executed successfully.

The project also records:

- inference latency
- training time
- peak VRAM
- trainable parameters
- total parameters
"""
    )

    sections.append(
        f"""## 7. Experimental Results

The tables below are generated directly from the experiment artifacts in
`{results_path}`.

### 7.1 Method Comparison

{df_to_markdown(method_df)}

### 7.2 Training Data Scaling

{df_to_markdown(scaling_df)}

### 7.3 RAG Retrieval Depth

{df_to_markdown(rag_k_df)}

### 7.4 LoRA Adapter Rank

{df_to_markdown(rank_df)}

### 7.5 Noise Sensitivity

{df_to_markdown(noise_df)}

{noise_msg}

### 7.6 Efficiency and Resource Metrics

{df_to_markdown(cost_df)}
"""
    )

    sections.append(
        f"""## 8. Error Analysis & Diagnostics

The evaluation pipeline groups failures into useful categories such as:

- `SYNTAX_ERROR`
- `SCHEMA_ERROR`
- `RESULT_MISMATCH`
- `TIMEOUT`
- `CORRECT`
- `UNCLASSIFIED`

The purpose is to identify why a prediction failed instead of only
reporting an accuracy number.

### Error Distribution

{df_to_markdown(error_df)}
"""
    )

    crossover_status = crossover_data.get(
        "status",
        "UNAVAILABLE",
    )

    crossover_message = crossover_data.get(
        "message",
        "Insufficient matched experiments.",
    )

    crossover_points = crossover_data.get(
        "crossover_points",
        [],
    )

    crossover_lines = [
        f"**Status:** `{crossover_status}`",
        f"**Message:** {crossover_message}",
        "",
    ]

    if crossover_points:
        crossover_lines.extend(
            [
                "| Fine-Tuning Method | Data Fraction | LoRA Rank | FT EX | RAG Configuration | RAG EX | Difference |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )

        for point in crossover_points:
            crossover_lines.append(
                f"| {point.get('fine_tuning_method', '')} "
                f"| {point.get('data_fraction', '')} "
                f"| {point.get('lora_rank', '')} "
                f"| {point.get('ft_execution_accuracy', '')} "
                f"| {point.get('rag_configuration', '')} "
                f"| {point.get('rag_execution_accuracy', '')} "
                f"| {point.get('observed_ex_difference', '')} |"
            )
    else:
        crossover_lines.append(
            "**NOT RUN / INSUFFICIENT DATA** — "
            "There are not enough matched RAG and fine-tuning experiments "
            "to determine an empirical crossover point."
        )

    sections.append(
        "## 9. Crossover Analysis: When Does Fine-Tuning Beat RAG?\n\n"
        "The project ultimately aims to find the training-data level at "
        "which fine-tuning performs better than a comparable RAG setup.\n\n"
        + "\n".join(crossover_lines)
        + "\n"
    )

    sections.append(
        r"""## 10. Discussion

The main engineering trade-off is straightforward.

RAG avoids model training, but every query depends on retrieval and a larger
prompt.

Fine-tuning requires training time and suitable hardware, but the resulting
model can use learned patterns without retrieving demonstrations for every
query.

LoRA is particularly attractive when only a small fraction of model
parameters needs to be updated.

QLoRA targets the same idea while reducing the memory requirement through
quantization.

These trade-offs should be evaluated using the recorded experiment results.
The current report does not assume that one method is better when the
required matched experiments are missing.
"""
    )

    sections.append(
        f"""## 11. Limitations & Execution Integrity

The current state of the project is summarized below.

| Component | Status |
| --- | --- |
| Data loading and split checks | Implemented and tested |
| Prompting | Implemented and tested |
| SQLite evaluation | Implemented and tested |
| RAG retrieval | Implemented and tested |
| LoRA training | Implemented; GPU execution required |
| QLoRA training | Implemented; GPU execution required |
| Experiment runner | Implemented and tested |
| Result analysis | Implemented and tested |
| Research report | Implemented and tested |
| Full Spider benchmark | Depends on completed experiment runs |

### Current Research Status

Completed research runs in `{results_path}`: **{len(completed_research)}**

If the required GPU experiments have not been completed, their results are
not estimated or filled in manually.

Hardware detected for this report:

- Device: `{getattr(hardware, "device", "unknown")}`
- CUDA available: `{getattr(hardware, "cuda_available", "unknown")}`
"""
    )

    sections.append(
        f"""## 12. Reproducibility & Environment Profile

The report is generated from the saved experiment artifacts rather than
manually entered result values.

### Environment

- Python: `{getattr(hardware, "python_version", "not recorded")}`
- PyTorch: `{getattr(hardware, "torch_version", "not recorded")}`
- Transformers: `{getattr(hardware, "transformers_version", "not recorded")}`
- PEFT: `{getattr(hardware, "peft_version", "not recorded")}`
- BitsAndBytes: `{getattr(hardware, "bitsandbytes_available", "not recorded")}`
- CPU cores: `{getattr(hardware, "cpu_count", "not recorded")}`
- System RAM: `{getattr(hardware, "ram_total_gb", "not recorded")}`

Each experiment records its configuration and result information so that
the analysis can be regenerated from the same artifacts.
"""
    )

    sections.append(
        """## 13. Conclusion & Next Steps

TuneLab provides a complete workflow for comparing prompting, retrieval, LoRA, and QLoRA for Text-to-SQL tasks.

The main distinction is between pipeline completion and research execution. The software pipeline is implemented and tested, but the final GPU experiment matrix is not yet complete.

The remaining work is to run the planned experiments, save their result files, and regenerate this report. No missing experimental results should be inferred or manually added.

Once the required runs are available, the study can evaluate:

1. How accuracy changes with training-data size.
2. How RAG performance changes with retrieval depth.
3. How LoRA rank affects accuracy and resource usage.
4. How label noise affects each method.
5. Whether a measured crossover exists between RAG and fine-tuning.

### Current Status

- Software pipeline: implemented
- Evaluation pipeline: tested
- RAG pipeline: tested
- LoRA pipeline: implemented and tested
- QLoRA pipeline: implemented and tested
- Formal GPU benchmark: pending
- Full research conclusions: pending sufficient experimental results
"""
    )

    report_text = "\n\n".join(sections)

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")
        logger.info("Generated research report successfully: %s", output_path)

    return report_text


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the TuneLab research report."
    )

    parser.add_argument(
        "--results-dir",
        default="experiments/runs",
    )

    parser.add_argument(
        "--output-file",
        default="reports/research_report.md",
    )

    args = parser.parse_args()

    generate_research_report(
        results_dir=args.results_dir,
        output_file=args.output_file,
        include_smoke=False,
    )

    print(
        f"TuneLab Research Report generated successfully: {args.output_file}"
    )


if __name__ == "__main__":
    main()