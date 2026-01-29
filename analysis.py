"""
Reads JSONL produced during training and generates diagnostic plots.

Expected JSONL keys:
  ep, stage, hits, crash, return, steps,
  epsilon, alpha,
  r_hit, r_progress, r_step, r_near_boundary, r_oob
  done_reason ("crash"/"max_steps"/"success"/etc.)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# IO
# ----------------------------
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[analysis] No log file found at: {path}")
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _to_cols(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not rows:
        return {}

    keys = set()
    for r in rows:
        keys.update(r.keys())

    cols: Dict[str, List[Any]] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            cols[k].append(r.get(k, np.nan))

    out: Dict[str, np.ndarray] = {}
    for k, v in cols.items():
        if any(isinstance(x, str) for x in v if x is not np.nan):
            out[k] = np.array(v, dtype=object)
        else:
            tmp = []
            for x in v:
                if x is None:
                    tmp.append(np.nan)
                elif isinstance(x, (int, float)):
                    tmp.append(float(x))
                else:
                    tmp.append(np.nan)
            out[k] = np.array(tmp, dtype=float)
    return out


# ----------------------------
# Rolling helpers (no pandas)
# ----------------------------
def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x.copy()
    if x.size < w:
        return np.array([], dtype=float)
    k = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, k, mode="valid")


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return np.zeros_like(x)
    if x.size < w:
        return np.array([], dtype=float)
    k = np.ones(w, dtype=float) / float(w)
    mu = np.convolve(x, k, mode="valid")
    mu2 = np.convolve(x * x, k, mode="valid")
    var = np.maximum(mu2 - mu * mu, 0.0)
    return np.sqrt(var)


def _rolling_mean_std(x: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    return _rolling_mean(x, w), _rolling_std(x, w)


def _x_for_valid(n_valid: int, window: int) -> np.ndarray:
    return np.arange(n_valid) + max(int(window), 1)


def _stage_change_points(stage: np.ndarray) -> List[int]:
    if stage.size == 0:
        return []
    s = np.asarray(stage, dtype=float)
    if np.all(np.isnan(s)):
        return []

    # forward fill NaNs
    s2 = s.copy()
    last = np.nan
    for i in range(len(s2)):
        if np.isnan(s2[i]):
            s2[i] = last
        else:
            last = s2[i]

    if np.isnan(s2[0]):
        first_valid = np.where(~np.isnan(s2))[0]
        if first_valid.size == 0:
            return []
        s2[:first_valid[0]] = s2[first_valid[0]]

    return (np.where(np.diff(s2) != 0)[0] + 1).tolist()


def _add_stage_lines(ax, stage: np.ndarray) -> None:
    for cp in _stage_change_points(stage):
        ax.axvline(cp, ls="--", alpha=0.25, color="0.4")


# ----------------------------
# Plots
# ----------------------------
def _plot_return_hits(cols: Dict[str, np.ndarray], out_path: Path, window: int) -> None:
    ret = cols.get("return")
    hits = cols.get("hits")
    stage = cols.get("stage", np.array([]))

    if ret is None or hits is None:
        return

    ret_m, ret_s = _rolling_mean_std(ret, window)
    hit_m, _ = _rolling_mean_std(hits, window)
    if ret_m.size == 0:
        return

    x = _x_for_valid(ret_m.size, window)

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Return: blue (C0)
    ax1.plot(x, ret_m, color="C0", label="Return (mean)")
    ax1.fill_between(x, ret_m - ret_s, ret_m + ret_s, color="C0", alpha=0.20, label="Return ±1σ")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.grid(alpha=0.25)

    # Hits: green (C2) on twin axis
    ax2 = ax1.twinx()
    ax2.plot(x, hit_m, color="C2", label="Hits (mean)")
    ax2.set_ylabel("Hits")

    # combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    if stage.size:
        _add_stage_lines(ax1, stage)

    ax1.set_title(f"Return & Hits (rolling window={window})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_crash_steps(cols: Dict[str, np.ndarray], out_path: Path, window: int) -> None:
    crash = cols.get("crash", np.array([]))
    steps = cols.get("steps", np.array([]))
    stage = cols.get("stage", np.array([]))

    if crash.size == 0 and steps.size == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    if crash.size:
        c = _rolling_mean(crash, window)
        if c.size:
            x = _x_for_valid(c.size, window)
            axes[0].plot(x, c)
            axes[0].set_title(f"Crash rate (rolling={window})")
            axes[0].set_ylabel("Crash fraction")
            axes[0].set_xlabel("Episode")
            axes[0].grid(alpha=0.3)
            if stage.size:
                _add_stage_lines(axes[0], stage)

    if steps.size:
        s = _rolling_mean(steps, window)
        if s.size:
            x = _x_for_valid(s.size, window)
            axes[1].plot(x, s)
            axes[1].set_title(f"Steps (rolling={window})")
            axes[1].set_ylabel("Steps")
            axes[1].set_xlabel("Episode")
            axes[1].grid(alpha=0.3)
            if stage.size:
                _add_stage_lines(axes[1], stage)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_done_reason_over_time(cols: Dict[str, np.ndarray], out_path: Path, window: int) -> None:
    reason = cols.get("done_reason", np.array([]))
    stage = cols.get("stage", np.array([]))

    if reason.size == 0:
        return

    cats = ["crash", "max_steps", "success"]
    onehot = {c: np.zeros(reason.size, dtype=float) for c in cats}

    for i, r in enumerate(reason):
        if not isinstance(r, str):
            onehot["other"][i] = 1.0
            continue
        rlow = r.lower()
        if "crash" in rlow or "oob" in rlow:
            onehot["crash"][i] = 1.0
        elif "max" in rlow:
            onehot["max_steps"][i] = 1.0
        elif "success" in rlow or "all_targets" in rlow:
            onehot["success"][i] = 1.0

    fig, ax = plt.subplots(figsize=(14, 5))
    for c in cats:
        y = _rolling_mean(onehot[c], window)
        if y.size == 0:
            continue
        x = _x_for_valid(y.size, window)
        ax.plot(x, y, label=c)

    ax.set_title(f"Done reason (rolling fraction, window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=3)
    if stage.size:
        _add_stage_lines(ax, stage)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def _plot_reward_share_100pct(cols: Dict[str, np.ndarray], out_path: Path, block: int) -> None:
    # only if r_* exists
    needed = ["r_hit", "r_progress", "r_step", "r_near_boundary", "r_oob"]
    present = [k for k in needed if k in cols]
    if not present:
        print("[analysis] No r_* columns found; skip reward share plot")
        return

    labels = [k.replace("r_", "") for k in present]
    arrs = [np.abs(cols[k]).astype(float) for k in present]
    n = min(len(a) for a in arrs)
    if n < block * 2:
        print("[analysis] Not enough episodes for reward share plot")
        return

    m = n // block
    binned = np.vstack([a[:m * block].reshape(m, block).mean(axis=1) for a in arrs])
    total = binned.sum(axis=0) + 1e-12
    frac = binned / total

    x = (np.arange(m) + 1) * block
    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(m)

    for i, lab in enumerate(labels):
        ax.bar(x, frac[i], bottom=bottom, width=block * 0.9, label=lab)
        bottom += frac[i]

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Share of |total reward|")
    ax.set_title(f"Reward component share (100% stacked bars, block={block})")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper left", ncol=min(len(labels), 5))

    stage = cols.get("stage", np.array([]))
    if stage.size:
        _add_stage_lines(ax, stage)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_reward_zscore(cols: Dict[str, np.ndarray], out_path: Path, window: int) -> None:
    candidates = [("hit", "r_hit"), ("progress", "r_progress"), ("step", "r_step"),
                  ("near_boundary", "r_near_boundary"), ("oob", "r_oob")]
    avail = [(lab, k) for lab, k in candidates if k in cols]
    if not avail:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    for lab, k in avail:
        x = cols[k].astype(float)
        mu = np.nanmean(x)
        sd = np.nanstd(x) + 1e-9
        z = (x - mu) / sd
        z_m = _rolling_mean(z, window)
        if z_m.size == 0:
            continue
        t = _x_for_valid(z_m.size, window)
        ax.plot(t, z_m, label=lab)

    ax.axhline(0.0, ls="--", alpha=0.5)
    ax.set_title(f"Reward z-score (rolling mean, window={window})   z=(x-mean)/std within each component")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Z-score")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", ncol=3)

    stage = cols.get("stage", np.array([]))
    if stage.size:
        _add_stage_lines(ax, stage)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Public API (called from train.py)
# ============================================================

def run_analysis(
    log_path: Path,
    out_dir: Path,
    window: int = 50,
    share_block: int = 100,
) -> None:
    """
    Called from train.py after training.
    Saves plots into out_dir (or out_dir/plots if you prefer; here: out_dir directly).
    """
    log_path = Path(log_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(log_path)
    if not rows:
        print("[analysis] No rows loaded; no plots generated.")
        return

    cols = _to_cols(rows)

    _plot_return_hits(cols, out_dir / "rolling_mean_return_hits.png", window)
    _plot_crash_steps(cols, out_dir / "crash_and_steps.png", window)
    _plot_done_reason_over_time(cols, out_dir / "done_reason_over_time.png", window)
    _plot_reward_share_100pct(cols, out_dir / f"reward_decomp_share_100_block{share_block}.png", share_block)
    _plot_reward_zscore(cols, out_dir / "reward_decomp_zscore.png", window)

    print(f"[analysis] Saved plots to: {out_dir.resolve()}")
