import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tunelab.analysis.models import ExperimentRun, PerExampleResult

logger = logging.getLogger(__name__)

SUPPORTED_STATUSES = {
    "COMPLETED",
    "FAILED",
    "PENDING_GPU",
    "PENDING_DATA",
    "PENDING_MODEL",
    "DRY_RUN_ONLY",
    "SMOKE",
}


def _classify_run(run_dict: Dict[str, Any], run_id: str) -> tuple[bool, bool]:
    """Determines whether a run is a genuine RESEARCH_RESULT or a PIPELINE_TEST.
    
    A run is a research result ONLY if:
      - Status is COMPLETED
      - Dataset is a genuine research benchmark (not fixtures or test data)
      - Run ID does not denote smoke or test execution
      - Not explicitly flagged as smoke or fixture
    """
    status = run_dict.get("status", "UNKNOWN")
    dataset = str(run_dict.get("dataset", "")).lower()
    exp_id = str(run_dict.get("experiment_id", "")).upper()
    rid = str(run_id).upper()

    is_smoke = (
        status == "SMOKE"
        or rid.startswith("SMOKE")
        or exp_id.startswith("SMOKE")
        or run_dict.get("is_smoke", False)
    )
    is_test_fixture = (
        dataset in ("fixtures", "test", "mock")
        or "FIXTURE" in rid
        or rid.startswith("TEST_")
        or run_dict.get("is_fixture", False)
    )

    if status == "COMPLETED" and not is_smoke and not is_test_fixture:
        return True, False  # is_research_result, is_pipeline_test
    return False, True


def load_per_example_results(run_dir: Path | str) -> List[PerExampleResult]:
    """Loads granular per-example evaluation results from per_example.jsonl or evaluation_results.json."""
    dir_p = Path(run_dir)
    results: List[PerExampleResult] = []

    # Priority 1: per_example.jsonl
    jsonl_p = dir_p / "per_example.jsonl"
    if jsonl_p.is_file():
        with open(jsonl_p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    results.append(
                        PerExampleResult(
                            example_id=item["example_id"],
                            database_id=item.get("database_id", item.get("db_id", "")),
                            predicted_sql=item.get("predicted_sql", ""),
                            reference_sql=item.get("reference_sql", item.get("gold_sql", "")),
                            exact_match=bool(item.get("exact_match", False)),
                            execution_match=bool(item.get("execution_match", False)),
                            error_type=item.get("error_type", "result_mismatch" if not item.get("execution_match") else "correct"),
                            error_message=item.get("error_message"),
                            latency_ms=item.get("latency_ms"),
                        )
                    )
        return results

    # Priority 2: evaluation_results.json
    eval_json_p = dir_p / "evaluation_results.json"
    if eval_json_p.is_file():
        with open(eval_json_p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for item in payload.get("results", []):
            results.append(
                PerExampleResult(
                    example_id=item["example_id"],
                    database_id=item.get("database_id", item.get("db_id", "")),
                    predicted_sql=item.get("predicted_sql", ""),
                    reference_sql=item.get("reference_sql", item.get("gold_sql", "")),
                    exact_match=bool(item.get("exact_match", False)),
                    execution_match=bool(item.get("execution_match", False)),
                    error_type=item.get("error_type", "result_mismatch" if not item.get("execution_match") else "correct"),
                    error_message=item.get("error_message"),
                    latency_ms=item.get("latency_ms"),
                )
            )
    return results


def load_run_from_directory(run_dir: Path | str) -> Optional[ExperimentRun]:
    """Loads a single ExperimentRun from a directory containing summary.json."""
    dir_p = Path(run_dir)
    summary_p = dir_p / "summary.json"
    if not summary_p.is_file():
        return None

    with open(summary_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check for noise metadata file if not in summary
    noise_p = dir_p / "noise_metadata.json"
    if noise_p.is_file() and not data.get("noise_metadata"):
        with open(noise_p, "r", encoding="utf-8") as f:
            data["noise_metadata"] = json.load(f)

    run_id = data.get("run_id", dir_p.name)
    is_res, is_pipe = _classify_run(data, run_id)

    # Calculate median latency if per-example exists and run completed
    per_examples = load_per_example_results(dir_p)
    median_lat = None
    if per_examples:
        lats = [r.latency_ms for r in per_examples if r.latency_ms is not None]
        if lats:
            lats.sort()
            mid = len(lats) // 2
            median_lat = (lats[mid] if len(lats) % 2 != 0 else (lats[mid - 1] + lats[mid]) / 2.0)

    # If run did not complete, ensure evaluation metrics remain None (never 0.0)
    status = data.get("status", "UNKNOWN")
    sql_val = data.get("sql_validity") if status == "COMPLETED" else None
    em = data.get("exact_match") if status == "COMPLETED" else None
    ex = data.get("execution_accuracy") if status == "COMPLETED" else None
    mean_lat = data.get("mean_latency_ms") if status == "COMPLETED" else None

    return ExperimentRun(
        run_id=run_id,
        experiment_id=data.get("experiment_id", ""),
        method=data.get("method", "UNKNOWN"),
        model=data.get("model", "unknown"),
        dataset=data.get("dataset", "spider"),
        dataset_version=data.get("dataset_version", "1.0"),
        data_fraction=float(data.get("data_fraction", 1.0)),
        rag_k=data.get("rag_k"),
        lora_rank=data.get("lora_rank"),
        noise_level=float(data.get("noise_level", 0.0)),
        noise_type=data.get("noise_type", "none"),
        seed=int(data.get("seed", 42)),
        status=status,
        total_examples=int(data.get("total_examples", 0)),
        sql_validity=sql_val,
        exact_match=em,
        execution_accuracy=ex,
        mean_latency_ms=mean_lat,
        median_latency_ms=median_lat,
        training_time_s=data.get("training_time_s"),
        inference_time_s=data.get("inference_time_s"),
        peak_vram_mb=data.get("peak_vram_mb"),
        trainable_parameters=data.get("trainable_parameters"),
        total_parameters=data.get("total_parameters"),
        error_message=data.get("error_message"),
        noise_metadata=data.get("noise_metadata"),
        hardware=data.get("hardware", {}),
        timestamp=data.get("timestamp", ""),
        git_commit=data.get("git_commit"),
        configuration_hash=data.get("configuration_hash", ""),
        run_dir=str(dir_p),
        per_example_results=per_examples,
        is_research_result=is_res,
        is_pipeline_test=is_pipe,
    )


def load_runs_from_master(master_path: Path | str) -> List[ExperimentRun]:
    """Loads experiment run records from master_results.jsonl or master_results.csv."""
    path = Path(master_path)
    runs: List[ExperimentRun] = []
    if not path.is_file():
        return runs

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    run_id = item.get("run_id", "")
                    is_res, is_pipe = _classify_run(item, run_id)
                    status = item.get("status", "UNKNOWN")
                    runs.append(
                        ExperimentRun(
                            run_id=run_id,
                            experiment_id=item.get("experiment_id", ""),
                            method=item.get("method", "UNKNOWN"),
                            model=item.get("model", "unknown"),
                            dataset=item.get("dataset", "spider"),
                            dataset_version=item.get("dataset_version", "1.0"),
                            data_fraction=float(item.get("data_fraction", 1.0)),
                            rag_k=item.get("rag_k"),
                            lora_rank=item.get("lora_rank"),
                            noise_level=float(item.get("noise_level", 0.0)),
                            noise_type=item.get("noise_type", "none"),
                            seed=int(item.get("seed", 42)),
                            status=status,
                            total_examples=int(item.get("total_examples", 0)),
                            sql_validity=item.get("sql_validity") if status == "COMPLETED" else None,
                            exact_match=item.get("exact_match") if status == "COMPLETED" else None,
                            execution_accuracy=item.get("execution_accuracy") if status == "COMPLETED" else None,
                            mean_latency_ms=item.get("mean_latency_ms") if status == "COMPLETED" else None,
                            training_time_s=item.get("training_time_s"),
                            inference_time_s=item.get("inference_time_s"),
                            peak_vram_mb=item.get("peak_vram_mb"),
                            trainable_parameters=item.get("trainable_parameters"),
                            total_parameters=item.get("total_parameters"),
                            error_message=item.get("error_message"),
                            noise_metadata=item.get("noise_metadata"),
                            hardware=item.get("hardware", {}),
                            timestamp=item.get("timestamp", ""),
                            git_commit=item.get("git_commit"),
                            configuration_hash=item.get("configuration_hash", ""),
                            is_research_result=is_res,
                            is_pipeline_test=is_pipe,
                        )
                    )
    elif path.suffix == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                run_id = row.get("run_id", "")
                status = row.get("status", "UNKNOWN")
                is_res, is_pipe = _classify_run(row, run_id)
                # Parse numeric fields safely
                def _float_or_none(v):
                    return float(v) if v not in (None, "", "None") else None
                def _int_or_none(v):
                    return int(float(v)) if v not in (None, "", "None") else None

                noise_meta = None
                if row.get("noise_metadata"):
                    try:
                        noise_meta = json.loads(row["noise_metadata"])
                    except Exception:
                        pass

                runs.append(
                    ExperimentRun(
                        run_id=run_id,
                        experiment_id=row.get("experiment_id", ""),
                        method=row.get("method", "UNKNOWN"),
                        model=row.get("model", "unknown"),
                        dataset=row.get("dataset", "spider"),
                        dataset_version=row.get("dataset_version", "1.0"),
                        data_fraction=_float_or_none(row.get("data_fraction")) or 1.0,
                        rag_k=_int_or_none(row.get("rag_k")),
                        lora_rank=_int_or_none(row.get("lora_rank")),
                        noise_level=_float_or_none(row.get("noise_level")) or 0.0,
                        noise_type=row.get("noise_type", "none"),
                        seed=_int_or_none(row.get("seed")) or 42,
                        status=status,
                        total_examples=_int_or_none(row.get("total_examples")) or 0,
                        sql_validity=_float_or_none(row.get("sql_validity")) if status == "COMPLETED" else None,
                        exact_match=_float_or_none(row.get("exact_match")) if status == "COMPLETED" else None,
                        execution_accuracy=_float_or_none(row.get("execution_accuracy")) if status == "COMPLETED" else None,
                        mean_latency_ms=_float_or_none(row.get("mean_latency_ms")) if status == "COMPLETED" else None,
                        training_time_s=_float_or_none(row.get("training_time_s")),
                        inference_time_s=_float_or_none(row.get("inference_time_s")),
                        peak_vram_mb=_float_or_none(row.get("peak_vram_mb")),
                        trainable_parameters=_int_or_none(row.get("trainable_parameters")),
                        total_parameters=_int_or_none(row.get("total_parameters")),
                        error_message=row.get("error_message"),
                        noise_metadata=noise_meta,
                        timestamp=row.get("timestamp", ""),
                        git_commit=row.get("git_commit"),
                        configuration_hash=row.get("configuration_hash", ""),
                        is_research_result=is_res,
                        is_pipeline_test=is_pipe,
                    )
                )

    return runs


def load_all_runs(
    results_dir: Path | str = "experiments/runs",
    master_file: Optional[Path | str] = None,
    include_smoke: bool = False,
) -> List[ExperimentRun]:
    """Loads all experiment runs from directories and/or master files.
    
    If include_smoke is False (default), filters out all PIPELINE_TEST runs
    (smoke tests, dry runs, pending runs, failed runs, fixture runs).
    """
    base_dir = Path(results_dir)
    loaded_runs: Dict[str, ExperimentRun] = {}

    # 1. Scan directory folders for runs with summary.json
    if base_dir.is_dir():
        for run_dir in base_dir.iterdir():
            if run_dir.is_dir():
                run = load_run_from_directory(run_dir)
                if run:
                    loaded_runs[run.run_id] = run

    # 2. Check master_results file (either specified or inside results_dir)
    m_candidates = []
    if master_file:
        m_candidates.append(Path(master_file))
    else:
        m_candidates.extend([
            base_dir / "master_results.jsonl",
            base_dir / "master_results.csv",
            Path("experiments/results/master_results.jsonl"),
            Path("experiments/results/master_results.csv"),
        ])

    for m_path in m_candidates:
        if m_path.is_file():
            master_runs = load_runs_from_master(m_path)
            for m_run in master_runs:
                if m_run.run_id not in loaded_runs:
                    loaded_runs[m_run.run_id] = m_run
                else:
                    # Enrich existing loaded run if master has additional info
                    existing = loaded_runs[m_run.run_id]
                    if not existing.noise_metadata and m_run.noise_metadata:
                        existing.noise_metadata = m_run.noise_metadata

    runs = list(loaded_runs.values())

    if not include_smoke:
        # Exclude all non-research runs
        runs = [r for r in runs if r.is_research_result]

    return runs


def filter_by_status(runs: List[ExperimentRun], allowed_statuses: Set[str] | List[str]) -> List[ExperimentRun]:
    """Filters runs matching specified statuses."""
    allowed = set(allowed_statuses)
    return [r for r in runs if r.status in allowed]


def filter_by_method(runs: List[ExperimentRun], method: str) -> List[ExperimentRun]:
    """Filters runs for a specific method."""
    return [r for r in runs if r.method.upper() == method.upper()]


def filter_by_experiment(runs: List[ExperimentRun], experiment_id: str) -> List[ExperimentRun]:
    """Filters runs belonging to a specific experiment axis/id."""
    return [r for r in runs if r.experiment_id == experiment_id]
