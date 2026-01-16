# ============================================================
# Training Analysis & Plotting
# ============================================================
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_jsonl(path):
    """Load JSONL log file into DataFrame"""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def rolling_mean(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean()


def rolling_std(x, w):
    return pd.Series(x).rolling(w, min_periods=1).std()


def plot_return_and_hits(df, window=50, out_path=None) -> bool:
    """Plot rolling mean of returns and hits. Returns True if plot was created."""
    if "return" not in df.columns or "hits" not in df.columns:
        print("[Analysis] Warning: Missing 'return' or 'hits' column, skipping")
        return False
    
    fig, ax1 = plt.subplots(figsize=(12, 4))

    r_mean = rolling_mean(df["return"], window)
    r_std = rolling_std(df["return"], window)

    x = df["ep"].values if "ep" in df.columns else np.arange(len(df))

    ax1.plot(x, r_mean, label="Return (mean)", color="blue")
    ax1.fill_between(x, r_mean - r_std, r_mean + r_std, alpha=0.3, color="blue")
    ax1.set_ylabel("Return", color="blue")
    ax1.set_xlabel("Episode")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, rolling_mean(df["hits"], window), color="green", label="Hits")
    ax2.set_ylabel("Hits", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(f"Training Progress: Return & Hits (window={window})")
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    return True


def plot_reward_decomposition(df, window=50, out_path=None) -> bool:
    """Plot reward component breakdown. Returns True if plot was created."""
    components = ["r_hit", "r_progress", "r_step", "r_near_boundary", "r_oob"]
    colors = ["green", "blue", "gray", "orange", "red"]
    
    # Only plot columns that exist
    available = [(c, col) for c, col in zip(colors, components) if col in df.columns]
    
    if not available:
        print("[Analysis] Warning: No reward components found in log, skipping reward decomposition")
        return False
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    x = df["ep"].values if "ep" in df.columns else np.arange(len(df))

    for color, col in available:
        ax.plot(x, rolling_mean(df[col], window), label=col.replace("r_", ""), color=color)

    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_ylabel("Reward contribution (per episode)")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.title(f"Reward Decomposition (window={window})")
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    return True


def plot_done_reason(df, out_path=None) -> bool:
    """Plot distribution of episode termination reasons. Returns True if plot was created."""
    if "done_reason" not in df.columns:
        print("[Analysis] Warning: 'done_reason' column not found, skipping")
        return False
    
    counts = df["done_reason"].value_counts(normalize=True)
    
    if len(counts) == 0:
        print("[Analysis] Warning: No done_reason data, skipping")
        return False

    plt.figure(figsize=(8, 5))
    # Use default colormap instead of fixed color list
    bars = counts.plot(kind="bar", colormap="tab10")
    
    # Add percentage labels
    for i, (idx, val) in enumerate(counts.items()):
        plt.text(i, val + 0.01, f"{val*100:.1f}%", ha="center", fontsize=10)
    
    plt.ylabel("Fraction of episodes")
    plt.xlabel("Termination Reason")
    plt.title("Episode Termination Reasons")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    return True


def plot_return_vs_hits(df, out_path=None) -> bool:
    """Sanity check: scatter plot of return vs hits. Returns True if plot was created."""
    if "hits" not in df.columns or "return" not in df.columns:
        print("[Analysis] Warning: Missing hits or return column, skipping scatter plot")
        return False
    
    plt.figure(figsize=(6, 5))
    plt.scatter(df["hits"], df["return"], alpha=0.3, s=10)
    plt.xlabel("Hits")
    plt.ylabel("Return")
    plt.title("Return vs Hits (Reward Sanity Check)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    return True


def plot_crash_and_steps(df, window=50, out_path=None) -> bool:
    """Plot crash rate and steps per episode. Returns True if plot was created."""
    has_crash = "crash" in df.columns
    has_steps = "steps" in df.columns
    
    # Skip entirely if both columns missing
    if not has_crash and not has_steps:
        print("[Analysis] Warning: Missing both 'crash' and 'steps' columns, skipping")
        return False
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    x = df["ep"].values if "ep" in df.columns else np.arange(len(df))
    
    # Crash rate
    ax1 = axes[0]
    if has_crash:
        ax1.plot(x, rolling_mean(df["crash"], window), color="red")
        ax1.set_ylabel("Crash Rate")
        ax1.set_xlabel("Episode")
        ax1.set_title(f"Crash Rate (window={window})")
        ax1.grid(alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
    else:
        ax1.text(0.5, 0.5, "crash column missing", ha="center", va="center", 
                 transform=ax1.transAxes, fontsize=12, color="gray")
        ax1.set_title("Crash Rate (missing)")
    
    # Steps
    ax2 = axes[1]
    if has_steps:
        ax2.plot(x, rolling_mean(df["steps"], window), color="orange")
        ax2.set_ylabel("Steps")
        ax2.set_xlabel("Episode")
        ax2.set_title(f"Steps per Episode (window={window})")
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "steps column missing", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="gray")
        ax2.set_title("Steps per Episode (missing)")
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    return True


def print_summary(df):
    """Print training summary statistics"""
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Episodes logged: {len(df)}")
    
    if "return" in df.columns:
        print(f"\nReturns:")
        print(f"  Mean: {df['return'].mean():.2f}")
        print(f"  Std:  {df['return'].std():.2f}")
        print(f"  Min:  {df['return'].min():.2f}")
        print(f"  Max:  {df['return'].max():.2f}")
        if len(df) > 100:
            print(f"  Last 100 mean: {df['return'].iloc[-100:].mean():.2f}")
    
    if "hits" in df.columns:
        print(f"\nHits:")
        print(f"  Mean: {df['hits'].mean():.2f}")
        print(f"  Max:  {df['hits'].max()}")
        if len(df) > 100:
            print(f"  Last 100 mean: {df['hits'].iloc[-100:].mean():.2f}")
    
    if "crash" in df.columns:
        print(f"\nCrash rate: {df['crash'].mean()*100:.1f}%")
        if len(df) > 100:
            print(f"  Last 100: {df['crash'].iloc[-100:].mean()*100:.1f}%")
    
    if "done_reason" in df.columns:
        print(f"\nTermination reasons:")
        for reason, count in df["done_reason"].value_counts().items():
            print(f"  {reason}: {count} ({count/len(df)*100:.1f}%)")
    
    print("=" * 50)


def run_analysis(log_path: Path, out_dir: Path, window: int = 50):
    """
    Main analysis function called by train.py
    
    Args:
        log_path: Path to JSONL training log
        out_dir: Directory to save plots
        window: Rolling window size for smoothing
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        df = load_jsonl(log_path)
    except Exception as e:
        print(f"[Analysis] Error loading log: {e}")
        return
    
    if len(df) == 0:
        print("[Analysis] Warning: Log file is empty, skipping analysis")
        return
    
    print(f"[Analysis] Loaded {len(df)} episodes from {log_path}")
    
    # Print summary
    print_summary(df)

    # Generate plots
    print(f"[Analysis] Generating plots...")
    plots_created = []
    
    # 1. Return and hits
    if plot_return_and_hits(df, window, out_path=out_dir / "rolling_mean_return_hits.png"):
        plots_created.append("rolling_mean_return_hits.png")

    # 2. Reward decomposition
    if plot_reward_decomposition(df, window, out_path=out_dir / "reward_decomposition.png"):
        plots_created.append("reward_decomposition.png")

    # 3. Done reasons
    if plot_done_reason(df, out_path=out_dir / "done_reason.png"):
        plots_created.append("done_reason.png")

    # 4. Return vs hits
    if plot_return_vs_hits(df, out_path=out_dir / "return_vs_hits.png"):
        plots_created.append("return_vs_hits.png")
    
    # 5. Crash and steps
    if plot_crash_and_steps(df, window, out_path=out_dir / "crash_and_steps.png"):
        plots_created.append("crash_and_steps.png")

    # Print created plots
    for plot_name in plots_created:
        print(f"  - {plot_name}")
    
    print(f"[Analysis] Complete! {len(plots_created)} plots saved to {out_dir}")


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze SARSA training logs")
    parser.add_argument("log_path", type=str, help="Path to JSONL log file")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: same as log file)")
    parser.add_argument("--window", type=int, default=50,
                        help="Rolling window size")
    
    args = parser.parse_args()
    
    log_path = Path(args.log_path)
    out_dir = Path(args.out_dir) if args.out_dir else log_path.parent
    
    run_analysis(log_path, out_dir, window=args.window)