"""
plots.py — Visualisation utilities for logged pipeline results.
 
Functions:
    plot_results()           — all charts from logs.csv
    plot_phase5_comparison() — bias rate chart from phase5_report.csv
"""
 
import os
 
import matplotlib.pyplot as plt
import pandas as pd
 
LOG_FILE     = "logs.csv"
PHASE5_FILE  = "phase5_report.csv"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════
 
def _load_logs() -> pd.DataFrame | None:
    if not os.path.isfile(LOG_FILE):
        print(f"[plots] '{LOG_FILE}' not found — run some queries first.")
        return None
    df = pd.read_csv(LOG_FILE)
    if df.empty:
        print("[plots] Log file is empty.")
        return None
    return df
 
 
def _load_phase5() -> pd.DataFrame | None:
    if not os.path.isfile(PHASE5_FILE):
        print(f"[plots] '{PHASE5_FILE}' not found — run Phase 5 first.")
        return None
    df = pd.read_csv(PHASE5_FILE)
    if df.empty:
        print("[plots] Phase 5 file is empty.")
        return None
    return df
 
 
# ══════════════════════════════════════════════════════════════════════════════
# CHARTS FROM logs.csv
# ══════════════════════════════════════════════════════════════════════════════
 
def plot_evaluation_counts(df: pd.DataFrame) -> None:
    """Bar chart: Safe vs Biased response counts."""
    counts = df["evaluation"].value_counts()
    colors = [("#4CAF50" if c == "Safe" else "#F44336") for c in counts.index]
 
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Bias Evaluation Results")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()
 
 
def plot_brs_distribution(df: pd.DataFrame) -> None:
    """Histogram: distribution of BRS values (0–1)."""
    fig, ax = plt.subplots()
    df["brs"].dropna().plot(kind="hist", bins=20, ax=ax,
                            color="#2196F3", edgecolor="white")
    ax.set_title("BRS Distribution")
    ax.set_xlabel("Bias Risk Score (normalised 0–1)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()
 
 
def plot_action_counts(df: pd.DataFrame) -> None:
    """Bar chart: NORMAL / DEBIASING / UNLEARNING action frequencies."""
    counts = df["action"].value_counts()
    color_map = {
        "NORMAL":     "#4CAF50",
        "DEBIASING":  "#FF9800",
        "UNLEARNING": "#F44336",
        "FINETUNING": "#9C27B0",
    }
    bar_colors = [color_map.get(c, "#9E9E9E") for c in counts.index]
 
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax, color=bar_colors)
    ax.set_title("Action Distribution")
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# CHART FROM phase5_report.csv
# ══════════════════════════════════════════════════════════════════════════════
 
def plot_phase5_comparison() -> None:
    """
    Grouped bar chart showing Biased vs Safe counts for each phase,
    and a line overlay of bias rate % — from phase5_report.csv.
    """
    df = _load_phase5()
    if df is None:
        return
 
    phases = df["Phase"].tolist()
 
    # Parse bias rate string → float
    df["BiasRateFloat"] = (
        df["Bias Rate"].str.replace("%", "").astype(float)
    )
 
    fig, ax1 = plt.subplots(figsize=(10, 5))
 
    x     = range(len(phases))
    width = 0.35
 
    ax1.bar([i - width / 2 for i in x], df["Biased"], width,
            label="Biased", color="#F44336", alpha=0.85)
    ax1.bar([i + width / 2 for i in x], df["Safe"],   width,
            label="Safe",   color="#4CAF50", alpha=0.85)
 
    ax1.set_xlabel("Phase")
    ax1.set_ylabel("Query Count")
    ax1.set_title("Phase 5 — Bias Reduction Across Phases")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(phases, rotation=15, ha="right")
    ax1.legend(loc="upper left")
 
    # Overlay: bias rate line
    ax2 = ax1.twinx()
    ax2.plot(list(x), df["BiasRateFloat"].tolist(),
             color="#FF9800", marker="o", linewidth=2, label="Bias Rate %")
    ax2.set_ylabel("Bias Rate (%)")
    ax2.legend(loc="upper right")
 
    plt.tight_layout()
    plt.show()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════
 
def plot_results() -> None:
    """Render all logs.csv charts."""
    df = _load_logs()
    if df is None:
        return
    plot_evaluation_counts(df)
    plot_brs_distribution(df)
    plot_action_counts(df)
 
 
if __name__ == "__main__":
    plot_results()
    plot_phase5_comparison()