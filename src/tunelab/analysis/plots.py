import logging
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_output_dir(path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_accuracy_vs_data_fraction(
    scaling_df: pd.DataFrame,
    output_path: Path | str,
) -> bool:
    """Plots Execution Accuracy vs Data Fraction for LoRA and QLoRA.
    
    Omits missing runs without interpolation or false continuity.
    """
    if scaling_df.empty:
        logger.warning("Plot skipped: scaling_df is empty.")
        return False

    valid = scaling_df[scaling_df["EX"].notna() & (scaling_df["Status"] == "COMPLETED")]
    if valid.empty:
        logger.warning("Plot skipped: No completed runs available for Execution Accuracy vs Data Fraction.")
        return False

    out_p = _ensure_output_dir(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in ["LORA", "QLORA"]:
        m_df = valid[valid["Method"] == method].sort_values(by="Data Fraction")
        if not m_df.empty:
            ax.plot(
                m_df["Data Fraction"] * 100,
                m_df["EX"],
                marker="o",
                linestyle="-",
                label=method,
            )

    ax.set_xlabel("Training Data Percentage (%)")
    ax.set_ylabel("Execution Accuracy")
    ax.set_title("Execution Accuracy vs Data Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    return True


def plot_exact_match_vs_data_fraction(
    scaling_df: pd.DataFrame,
    output_path: Path | str,
) -> bool:
    """Plots Exact Match vs Data Fraction for LoRA and QLoRA."""
    if scaling_df.empty:
        logger.warning("Plot skipped: scaling_df is empty.")
        return False

    valid = scaling_df[scaling_df["EM"].notna() & (scaling_df["Status"] == "COMPLETED")]
    if valid.empty:
        logger.warning("Plot skipped: No completed runs available for Exact Match vs Data Fraction.")
        return False

    out_p = _ensure_output_dir(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in ["LORA", "QLORA"]:
        m_df = valid[valid["Method"] == method].sort_values(by="Data Fraction")
        if not m_df.empty:
            ax.plot(
                m_df["Data Fraction"] * 100,
                m_df["EM"],
                marker="s",
                linestyle="-",
                label=method,
            )

    ax.set_xlabel("Training Data Percentage (%)")
    ax.set_ylabel("Exact Match Rate")
    ax.set_title("Exact Match vs Data Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    return True


def plot_rag_accuracy_vs_k(
    rag_k_df: pd.DataFrame,
    output_path: Path | str,
) -> bool:
    """Plots RAG Execution Accuracy and Exact Match vs Retrieval Depth K."""
    if rag_k_df.empty:
        logger.warning("Plot skipped: rag_k_df is empty.")
        return False

    valid = rag_k_df[rag_k_df["EX"].notna() & (rag_k_df["Status"] == "COMPLETED")].sort_values(by="RAG K")
    if valid.empty:
        logger.warning("Plot skipped: No completed runs available for RAG Accuracy vs K.")
        return False

    out_p = _ensure_output_dir(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(valid["RAG K"], valid["EX"], marker="o", linestyle="-", label="Execution Accuracy")
    if "EM" in valid.columns and valid["EM"].notna().any():
        ax.plot(valid["RAG K"], valid["EM"], marker="s", linestyle="--", label="Exact Match")

    ax.set_xlabel("Retrieval Depth (K)")
    ax.set_ylabel("Metric Score")
    ax.set_title("RAG Accuracy vs Retrieval Depth (K)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    return True


def plot_lora_accuracy_vs_rank(
    rank_df: pd.DataFrame,
    output_path: Path | str,
) -> bool:
    """Plots LoRA / QLoRA Execution Accuracy vs LoRA Rank r."""
    if rank_df.empty:
        logger.warning("Plot skipped: rank_df is empty.")
        return False

    valid = rank_df[rank_df["EX"].notna() & (rank_df["Status"] == "COMPLETED")]
    if valid.empty:
        logger.warning("Plot skipped: No completed runs available for Accuracy vs Rank.")
        return False

    out_p = _ensure_output_dir(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in ["LORA", "QLORA"]:
        m_df = valid[valid["Method"] == method].sort_values(by="LoRA Rank")
        if not m_df.empty:
            ax.plot(m_df["LoRA Rank"], m_df["EX"], marker="o", linestyle="-", label=method)

    ax.set_xlabel("LoRA Rank (r)")
    ax.set_ylabel("Execution Accuracy")
    ax.set_title("Fine-Tuning Accuracy vs LoRA Rank (r)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    return True


def plot_accuracy_vs_noise(
    noise_df: pd.DataFrame,
    output_path: Path | str,
) -> bool:
    """Plots Execution Accuracy vs Training Noise Rate."""
    if noise_df.empty:
        logger.warning("Plot skipped: noise_df is empty.")
        return False

    valid = noise_df[noise_df["EX"].notna() & (noise_df["Status"] == "COMPLETED")]
    if valid.empty:
        logger.warning("Plot skipped: No completed runs available for Accuracy vs Noise.")
        return False

    out_p = _ensure_output_dir(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in ["RAG", "LORA", "QLORA"]:
        m_df = valid[valid["Method"] == method].sort_values(by="Noise Rate")
        if not m_df.empty:
            ax.plot(m_df["Noise Rate"] * 100, m_df["EX"], marker="o", linestyle="-", label=method)

    ax.set_xlabel("Noise Corruption Rate (%)")
    ax.set_ylabel("Execution Accuracy")
    ax.set_title("Method Robustness: Accuracy vs Training Noise")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    return True


def plot_accuracy_vs_latency(
    runs_df: pd.DataFrame,
    output_path: Path | str,
) -> bool:
    """Plots Execution Accuracy vs Mean Inference Latency across methods."""
    if runs_df.empty:
        logger.warning("Plot skipped: runs_df is empty.")
        return False

    valid = runs_df[
        runs_df["execution_accuracy"].notna()
        & runs_df["mean_latency_ms"].notna()
        & (runs_df["status"] == "COMPLETED")
    ]
    if valid.empty:
        logger.warning("Plot skipped: No completed runs with latency data available.")
        return False

    out_p = _ensure_output_dir(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in valid["method"].unique():
        m_df = valid[valid["method"] == method]
        ax.scatter(m_df["mean_latency_ms"], m_df["execution_accuracy"], s=80, label=method)

    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("Execution Accuracy")
    ax.set_title("Execution Accuracy vs Inference Latency")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    return True


def plot_error_distribution_by_method(
    error_dist_df: pd.DataFrame,
    output_path: Path | str,
) -> bool:
    """Plots breakdown of failure categories across evaluated methods."""
    if error_dist_df.empty:
        logger.warning("Plot skipped: error_dist_df is empty.")
        return False

    # Exclude 'correct' category to focus strictly on error breakdown
    err_only = error_dist_df[error_dist_df["error_type"] != "none"]
    err_only = err_only[err_only["error_type"] != "correct"]
    if err_only.empty:
        logger.warning("Plot skipped: No error records found in distribution.")
        return False

    pivot_df = err_only.pivot(index="method", columns="error_type", values="count").fillna(0)
    out_p = _ensure_output_dir(output_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot_df.plot(kind="bar", stacked=True, ax=ax)

    ax.set_xlabel("Method")
    ax.set_ylabel("Error Count")
    ax.set_title("Error Category Distribution by Method")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    return True
