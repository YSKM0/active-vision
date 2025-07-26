#!/usr/bin/env python3
from __future__ import annotations

"""
Bootstrap confidence interval (and optional p-value) for the mean or
median of the paired PSNR differences
"""

import json
from pathlib import Path
from typing import List, Dict, Callable

import numpy as np
import pandas as pd
import scipy.stats as st

# CONFIG
BASE_DIR = "/media/hanwliu/HanwenDisk/ActiveVision/tntdata_tntexp/tnt/M60/evaluation_splatfacto"
METHOD_REF = "rs"  # baseline
METHOD_TEST = "fvs"
VIEW_NUMS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
METRIC = "psnr"

STAT_FUNC = "mean"  # "mean" or "median"
N_BOOT = 10_000
CI_LEVEL = 0.95
RNG_SEED = 42


# Utilities
def generate_json_paths(base_dir: str, method: str, views: List[int]) -> List[Path]:
    prefix = method if method in ("rs", "fvs") else "vlm"
    return [
        Path(base_dir) / method / f"{prefix}_{v}" / f"{prefix}_{v}.json" for v in views
    ]


def collect_scene_psnr(
    base_dir: str, methods: List[str], view_numbers: List[int], metric: str = "psnr"
) -> pd.DataFrame:
    recs: List[Dict] = []
    for m in methods:
        for fp in generate_json_paths(base_dir, m, view_numbers):
            if not fp.exists():
                continue
            v = int(fp.stem.split("_")[-1])
            with open(fp) as f:
                res = json.load(f).get("results", {})
            if metric not in res:
                continue
            recs.append(dict(method=m, views=v, psnr=float(res[metric])))
    df = pd.DataFrame.from_records(recs)
    if df.empty:
        raise RuntimeError("No PSNR data found — check BASE_DIR & method names.")
    return df


def build_diff_series(df: pd.DataFrame, method_test: str, method_ref: str) -> pd.Series:
    piv = df.pivot(index="views", columns="method", values="psnr")
    return (piv[method_test] - piv[method_ref]).dropna().sort_index()


def bootstrap_ci(
    x: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Return (stat_orig, ci_low, ci_high) using percentile bootstrap.
    """
    n = len(x)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        stats[i] = stat_fn(sample)

    alpha = 1.0 - ci_level
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return stat_fn(x), lo, hi, stats


def bootstrap_p_value(stats: np.ndarray, stat_orig: float) -> float:
    """
    Two-sided bootstrap p-value: proportion of bootstrap stats whose
    magnitude is ≥ |stat_orig|, doubled.
    """
    extremity = np.abs(stats) >= abs(stat_orig)
    p = extremity.mean()
    return min(max(p, 0.0), 1.0)


def main():
    df = collect_scene_psnr(BASE_DIR, [METHOD_TEST, METHOD_REF], VIEW_NUMS, METRIC)
    diffs = build_diff_series(df, METHOD_TEST, METHOD_REF)
    x = diffs.values
    n = len(x)
    if n < 2:
        raise RuntimeError("Need at least two paired observations for bootstrapping.")

    print(f"\nΔ vector (TEST − REF, n = {n}):")
    print(diffs.to_string(float_format=lambda v: f"{v:.3f}"))

    # Choose statistic function
    stat_fn = np.mean if STAT_FUNC == "mean" else np.median

    rng = np.random.default_rng(RNG_SEED)
    stat_orig, ci_lo, ci_hi, boot_stats = bootstrap_ci(
        x, stat_fn, N_BOOT, CI_LEVEL, rng
    )

    print(
        f"\nBootstrap {int(CI_LEVEL*100)}% CI for {STAT_FUNC}(Δ) "
        f"based on {N_BOOT:,} resamples:"
    )
    print(f"  {STAT_FUNC}(Δ) = {stat_orig:.3f}")
    print(f"  CI = [{ci_lo:.3f}, {ci_hi:.3f}]")

    p_boot = bootstrap_p_value(boot_stats, stat_orig)
    print(f"\nTwo-sided bootstrap p-value (H₀: {STAT_FUNC}(Δ) = 0): p = {p_boot:.3e}")

    # Simple interpretation
    if ci_lo > 0 or ci_hi < 0:
        direction = "greater" if stat_orig > 0 else "less"
        print(f"→ Zero is outside the CI → significant; TEST is {direction} than REF.")
    else:
        print("→ Zero lies inside the CI → cannot rule out no difference.")


if __name__ == "__main__":
    main()
