#!/usr/bin/env python3
from __future__ import annotations

"""
Paired t-test (two-sided and one-sided)
Wilcoxon signed-rank (two-sided and one-sided)
Exact binomial sign test (two-sided and one-sided)
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st

# CONFIG
BASE_DIR = "/media/hanwliu/HanwenDisk/ActiveVision/tntdata_tntexp/tnt/M60/evaluation_splatfacto"
METHOD_REF = "rs"
METHOD_TEST = "fvs"
VIEW_NUMS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
METRIC = "psnr"
ALPHA = 0.05


# Utilities
def generate_json_paths(base_dir: str, method: str, views: List[int]) -> List[Path]:
    """
    For methods "rs" and "fvs" we look for:
        BASE_DIR/{method}/{method}_{v}/{method}_{v}.json
    For any other method, we look for:
        BASE_DIR/{method}/vlm_{v}/vlm_{v}.json
    """
    prefix = method if method in ("rs", "fvs") else "vlm"
    return [
        Path(base_dir) / method / f"{prefix}_{v}" / f"{prefix}_{v}.json" for v in views
    ]


def collect_scene_psnr(
    base_dir: str,
    methods: List[str],
    view_numbers: List[int],
    metric: str = "psnr",
) -> pd.DataFrame:
    """Return tidy df(method, views, psnr) for files that exist."""
    recs: List[Dict] = []
    for m in methods:
        for jp in generate_json_paths(base_dir, m, view_numbers):
            if not jp.exists():
                continue
            view = int(jp.stem.split("_")[-1])
            with open(jp) as f:
                res = json.load(f).get("results", {})
            if metric not in res:
                continue
            recs.append(dict(method=m, views=view, psnr=float(res[metric])))
    df = pd.DataFrame.from_records(recs)
    if df.empty:
        raise RuntimeError("No PSNR data found — check BASE_DIR and method names.")
    return df


def build_diff_series(
    df: pd.DataFrame,
    method_test: str,
    method_ref: str,
) -> pd.Series:
    """
    Return vector of Δ = PSNR_TEST − PSNR_REF indexed by view budget.
    Only budgets where *both* methods have data are used.
    """
    pivot = df.pivot(index="views", columns="method", values="psnr")
    required = [method_test, method_ref]
    if not set(required).issubset(pivot.columns):
        raise RuntimeError(f"Missing data for {required}")
    diffs = pivot[method_test] - pivot[method_ref]
    return diffs.dropna().sort_index()


def paired_tests(diffs: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Run paired t, Wilcoxon, and sign tests on the diff vector.
    Returns nested dict with keys 't', 'wilcoxon', 'sign'.
    """
    x = diffs.values
    n = len(x)
    if n < 2:
        raise RuntimeError("Need at least 2 paired budgets for testing.")

    results: Dict[str, Dict[str, float]] = {}

    # Paired t-test
    mean_x = np.mean(x)
    if mean_x > 0:
        p_greater = p_two / 2.0
        p_less = 1.0 - p_greater * 2.0
    else:
        p_greater = 1.0 - p_two / 2.0
        p_less = p_two / 2.0
    results["t"] = dict(
        stat=t_stat, p_two=p_two, p_greater=p_greater, p_less=p_less, df=n - 1
    )

    # Wilcoxon signed-rank
    try:
        w_stat, p_two_w = st.wilcoxon(x, zero_method="wilcox", correction=False)
    except ValueError:
        w_stat, p_two_w = math.nan, 1.0
    med = np.median(x)
    if med > 0:
        p_greater_w = p_two_w / 2.0
        p_less_w = 1.0 - p_two_w / 2.0
    else:
        p_greater_w = 1.0 - p_two_w / 2.0
        p_less_w = p_two_w / 2.0
    results["wilcoxon"] = dict(
        stat=w_stat,
        p_two=p_two_w,
        p_greater=p_greater_w,
        p_less=p_less_w,
        n_eff=np.count_nonzero(x != 0),
    )

    # Sign test (exact binomial)
    pos = np.sum(x > 0)
    neg = np.sum(x < 0)
    ties = np.sum(x == 0)
    nz = pos + neg
    if nz == 0:
        p_two_s = p_greater_s = p_less_s = 1.0
        z_stat = 0.0
    else:
        p_two_s = st.binomtest(pos, nz, 0.5, alternative="two-sided").pvalue
        p_greater_s = st.binomtest(pos, nz, 0.5, alternative="greater").pvalue
        p_less_s = st.binomtest(pos, nz, 0.5, alternative="less").pvalue
        phat = pos / nz
        z_stat = (phat - 0.5) / math.sqrt(0.25 / nz)
    results["sign"] = dict(
        stat=z_stat,
        p_two=p_two_s,
        p_greater=p_greater_s,
        p_less=p_less_s,
        n_eff=nz,
        pos=pos,
        neg=neg,
        ties=ties,
    )

    return results


def summary_table(diffs: pd.Series) -> Tuple[pd.DataFrame, Tuple[float, float, float]]:
    """
    Return per-budget ΔPSNR table and overall mean with 95% t-CI.
    """
    x = diffs.values
    n = len(x)
    mean = np.mean(x)
    sd = np.std(x, ddof=1) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else math.nan
    if n > 1:
        tcrit = st.t.ppf(0.975, n - 1)
        ci_low = mean - tcrit * se
        ci_high = mean + tcrit * se
    else:
        ci_low = ci_high = mean
    tbl = pd.DataFrame({"views": diffs.index, "delta_psnr": diffs.values})
    tbl = tbl.reset_index(drop=True)
    overall = pd.DataFrame({"views": ["OVERALL"], "delta_psnr": [mean]})
    tbl = pd.concat([tbl, overall], ignore_index=True)
    return tbl, (mean, ci_low, ci_high)


def main():
    methods = [METHOD_TEST, METHOD_REF]
    df = collect_scene_psnr(BASE_DIR, methods, VIEW_NUMS, METRIC)
    print(
        f"Loaded {len(df)} PSNR rows "
        f"({df['method'].nunique()} methods, {df['views'].nunique()} budgets)."
    )

    diffs = build_diff_series(df, METHOD_TEST, METHOD_REF)
    if diffs.empty:
        raise RuntimeError("No overlapping budgets between the two methods.")

    tbl, (mean_delta, ci_lo, ci_hi) = summary_table(diffs)
    print("\nPer-budget ΔPSNR (TEST − REF):")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(
        f"\nOverall mean Δ = {mean_delta:.3f} dB "
        f"(95% CI: {ci_lo:.3f} … {ci_hi:.3f}, n={len(diffs)})"
    )

    tests = paired_tests(diffs)
    print("\nPaired tests across budgets (Δ = TEST − REF):")
    print("  Two‑sided null: TEST = REF (median Δ = 0)")
    for name, res in tests.items():
        if name == "t":
            print(
                f"  [t-test]    t={res['stat']:.3f}  df={res['df']}  "
                f"p(two)={res['p_two']:.3e}  p(greater)={res['p_greater']:.3e}  p(less)={res['p_less']:.3e}"
            )
        elif name == "wilcoxon":
            print(
                f"  [Wilcoxon]  W={res['stat']:.3f}  n_eff={res['n_eff']}  "
                f"p(two)={res['p_two']:.3e}  p(greater)={res['p_greater']:.3e}  p(less)={res['p_less']:.3e}"
            )
        else:
            print(
                f"  [Sign]      z≈{res['stat']:.3f}  n_eff={res['n_eff']} "
                f"(+{res['pos']}/-{res['neg']}, ties={res['ties']})  "
                f"p(two)={res['p_two']:.3e}  p(greater)={res['p_greater']:.3e}  p(less)={res['p_less']:.3e}"
            )

    p_two_sided = tests["wilcoxon"]["p_two"]
    if p_two_sided < ALPHA:
        direction = "higher" if mean_delta > 0 else "lower"
        verdict = f"YES — significant; TEST is {direction} than REF"
    else:
        verdict = "NO — no significant difference"

    print(
        f"\nDecision @ α={ALPHA:.2f}: {verdict}  "
        f"(Wilcoxon two‑sided p={p_two_sided:.3e})"
    )


if __name__ == "__main__":
    main()
