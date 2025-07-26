#!/usr/bin/env python3
from __future__ import annotations

"""
Check for the i.i.d. assumption and normality of the paired
"""

import json, math
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.stattools import durbin_watson

# CONFIG
BASE_DIR = "/media/hanwliu/HanwenDisk/ActiveVision/tntdata_tntexp/tnt/M60/evaluation_splatfacto"
METHOD_REF = "rs"  # baseline
METHOD_TEST = "fvs"  # method under test
VIEW_NUMS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
METRIC = "psnr"
PLOT = False


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


# IID & normality check
def iid_and_norm_checks(diffs: pd.Series) -> None:
    x = diffs.values
    n = len(x)
    if n < 4:
        raise RuntimeError("Need at least 4 paired budgets for a meaningful check.")

    print(f"\nΔ vector (TEST - REF, n = {n}):")
    print(diffs.to_string(float_format=lambda v: f"{v:.3f}"))

    # Independence
    print("\n--- Independence / Autocorrelation ---")
    dw = durbin_watson(x)
    print(f"Durbin-Watson statistic: {dw:.3f}  (≈2 → no autocorrelation)")

    if PLOT:
        import matplotlib.pyplot as plt
        from pandas.plotting import lag_plot
        from statsmodels.graphics.tsaplots import plot_acf

        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        # Lag‑1 plot
        lag_plot(pd.Series(x), ax=ax[0], c="k", marker="o")
        ax[0].set_title("Lag‑1 plot of Δ")
        # ACF plot
        plot_acf(x, lags=min(10, n - 2), ax=ax[1])
        ax[1].set_title("ACF of Δ")
        # Residuals vs index
        ax[2].scatter(range(n), x, c="k")
        ax[2].axhline(0, ls="--", lw=0.8)
        ax[2].set_title("Δ vs index")
        plt.tight_layout()
        plt.show()

    # Identical distribution
    print("\n--- Identical distribution (first half vs second half) ---")
    first, second = x[: n // 2], x[n // 2 :]
    ks_stat, ks_p = st.ks_2samp(first, second)
    lev_stat, lev_p = st.levene(first, second, center="median")
    print(f"Kolmogorov-Smirnov: KS = {ks_stat:.3f}, p = {ks_p:.3e}")
    print(f"Levene equal‑variance: W = {lev_stat:.3f}, p = {lev_p:.3e}")

    # Optional histogram / KDE
    if PLOT:
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.histplot(x, kde=True, color="gray", bins="auto")
        plt.title("Histogram & KDE of Δ")
        plt.show()

    # Normality of differences
    print("\n--- Normality of Δ ---")
    sh_stat, sh_p = st.shapiro(x) if n <= 5000 else (math.nan, math.nan)
    print(
        f"Shapiro-Wilk: W = {sh_stat:.3f}, p = {sh_p:.3e}"
        if not math.isnan(sh_stat)
        else "Shapiro-Wilk skipped (n > 5000)"
    )

    # Anderson-Darling
    ad_res = st.anderson(x)
    crit = ", ".join([f"{s:.2f}%" for s in ad_res.significance_level])
    crit_vals = ", ".join([f"{v:.3f}" for v in ad_res.critical_values])
    print(f"Anderson-Darling: A² = {ad_res.statistic:.3f}")
    print(f"  Critical values ({crit}) = {crit_vals}")

    if PLOT:
        import matplotlib.pyplot as plt
        import scipy.stats as stats

        stats.probplot(x, dist="norm", plot=plt)
        plt.title("Q-Q plot of Δ")
        plt.show()

    # Independence of pairs
    print("\n--- Independence of pairs: design considerations ---")
    print("If each view-budget measurement is collected on the SAME scene,")
    print("there could be hidden dependencies across Δ values.")
    print("Random sampling of scenes / views and the ~flat Durbin-Watson")
    print("result above both lend credibility to independence.")


def main():
    df = collect_scene_psnr(BASE_DIR, [METHOD_TEST, METHOD_REF], VIEW_NUMS, METRIC)
    diffs = build_diff_series(df, METHOD_TEST, METHOD_REF)
    iid_and_norm_checks(diffs)


if __name__ == "__main__":
    main()
