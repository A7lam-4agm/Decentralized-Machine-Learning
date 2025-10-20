import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent       # .../fl_simulation
RESULTS_DIR = BASE_DIR / "results"               # .../fl_simulation/results
REPORT_DIR = BASE_DIR / "report"                 # .../fl_simulation/report

BATCH_LOG = RESULTS_DIR / "training_results.csv"
ROUND_LOG = RESULTS_DIR / "round_metrics.csv"
DIST_CSV = RESULTS_DIR / "client_distribution.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def plot_learning_curve():
    """Plot train/test loss and accuracy vs communication rounds."""
    if not (BATCH_LOG.exists() and ROUND_LOG.exists()):
        print("⚠️ Results not found. Run FL simulation first.")
        return

    # Aggregate train metrics per round
    batch = pd.read_csv(BATCH_LOG)
    train_rounds = batch.groupby("round", as_index=False).agg(
        train_loss=("train_loss", "mean"),
        train_acc=("train_acc", "mean"),
    )

    # Test metrics
    rlog = pd.read_csv(ROUND_LOG)[["round", "test_loss", "test_acc"]]
    df = train_rounds.merge(rlog, on="round", how="left").sort_values("round")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["round"], df["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(df["round"], df["test_loss"], label="Test Loss", linewidth=2)
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["round"], df["train_acc"], linestyle="--", label="Train Acc (%)", linewidth=2)
    ax2.plot(df["round"], df["test_acc"], linestyle="--", label="Test Acc (%)", linewidth=2)
    ax2.set_ylabel("Accuracy (%)")

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="lower right")

    out_path = REPORT_DIR / "learning_curve.pdf"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Learning curve saved to {out_path}")


def plot_data_distribution():
    """Plot non-IID client × class heatmap."""
    if not DIST_CSV.exists():
        print(f"⚠️ {DIST_CSV} not found. Run FL simulation first.")
        return

    mat = pd.read_csv(DIST_CSV, index_col=0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(mat, cmap="Blues", cbar=True, annot=False)
    plt.title("Non-IID Data Distribution Across Clients")
    plt.xlabel("Class")
    plt.ylabel("Client")

    out_path = REPORT_DIR / "data_distribution.pdf"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Data distribution saved to {out_path}")


if __name__ == "__main__":
    plot_learning_curve()
    plot_data_distribution()
    print("🎉 All visualizations generated!")
