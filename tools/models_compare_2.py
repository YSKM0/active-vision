import json, os
from statistics import mean
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List, Optional, Tuple, Dict


def generate_json_paths(base_dir: str, model_name: str, view_numbers: List[int]):
    prefix = model_name if model_name in ["fvs", "rs"] else "vlm"
    return [
        os.path.join(base_dir, model_name, f"{prefix}_{v}/{prefix}_{v}.json")
        for v in view_numbers
    ]


def analyze_metrics_by_view(
    base_dir: str,
    model_names: List[str],
    view_numbers: List[int],
    metric: str = "psnr",
    *,
    return_diffs: bool = False,
) -> Tuple[
    Dict[int, Dict[str, Dict[str, float]]],  # stats with model tags
    Optional[List[Tuple[int, float, str, str]]],
]:
    """
    Compute per-view max / min / mean of *metric* across models
    and record which model attains the extremes.

    Returns
    -------
    stats : {view: {'max': {'value': v, 'model': m},
                    'min': {'value': v, 'model': m},
                    'mean': val}}
    diffs : None  OR  [(view, span, max_model, min_model), ...]      # sorted ↓
    """

    # -----------  gather values ------------------------------------------------
    # view → list[(value, model)]
    values_per_view: Dict[int, List[Tuple[float, str]]] = defaultdict(list)

    for model in model_names:
        for fp in generate_json_paths(base_dir, model, view_numbers):
            if not os.path.exists(fp):
                continue
            with open(fp, "r") as f:
                results = json.load(f).get("results", {})
            if metric not in results:
                continue
            view = int(fp.split("_")[-1].split(".")[0])
            values_per_view[view].append((results[metric], model))

    # -----------  compute statistics ------------------------------------------
    stats: Dict[int, Dict[str, Dict[str, float]]] = {}
    for view, vm_list in values_per_view.items():
        if not vm_list:
            continue
        vals = [v for v, _ in vm_list]
        max_val, max_model = max(vm_list, key=lambda x: x[0])
        min_val, min_model = min(vm_list, key=lambda x: x[0])

        stats[view] = {
            "max": {"value": max_val, "model": max_model},
            "min": {"value": min_val, "model": min_model},
            "mean": mean(vals),
        }

    # -----------  optionally return spans -------------------------------------
    if return_diffs:
        diffs = [
            (
                view,
                s["max"]["value"] - s["min"]["value"],
                s["max"]["model"],
                s["min"]["model"],
            )
            for view, s in stats.items()
        ]
        diffs.sort(key=lambda x: x[1], reverse=True)
        return stats, diffs

    return stats, None


def mean_rank_across_views(
    base_dir: str,
    model_names: List[str],
    view_numbers: List[int],
    metric: str = "psnr",
    *,
    higher_is_better: bool = True,
    drop_if_missing: bool = False,
) -> List[Tuple[str, float]]:
    """
    Compute the mean (average) rank of each model over all specified training
    views and return a list sorted from best (lowest mean rank) to worst.

    Parameters
    ----------
    base_dir : str
        Root directory that contains the experiment sub-folders.
    model_names : List[str]
        Sub-folder names (= model identifiers) to include.
    view_numbers : List[int]
        Training-view counts to consider.
    metric : str, default 'psnr'
        Which metric key to read from each JSON’s 'results' section.
    higher_is_better : bool, default True
        True  → higher metric values rank better (e.g. PSNR, SSIM)
        False → lower values rank better (e.g. LPIPS, loss)
    drop_if_missing : bool, default False
        If True, *exclude* views where a model lacks data from that model’s
        mean; otherwise missing views are treated as worst-rank for that view.

    Returns
    -------
    List[(model, mean_rank)]  sorted ascending  (best first, like a leaderboard)
    """

    # view → list[(model, value)]
    values_by_view: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

    for model in model_names:
        for fp in generate_json_paths(base_dir, model, view_numbers):
            if not os.path.exists(fp):
                continue
            with open(fp, "r") as f:
                res = json.load(f).get("results", {})
            if metric not in res:
                continue
            view = int(fp.split("_")[-1].split(".")[0])
            values_by_view[view].append((model, res[metric]))

    # model → cumulative rank + count
    rank_accum: Dict[str, float] = defaultdict(float)
    rank_counts: Dict[str, int] = defaultdict(int)

    for view, mv_list in values_by_view.items():
        if not mv_list:
            continue

        # sort within this view
        mv_list.sort(
            key=lambda x: x[1],
            reverse=higher_is_better,  # descending if higher is better
        )

        # assign ranks starting from 1
        for rank, (model, _) in enumerate(mv_list, start=1):
            rank_accum[model] += rank
            rank_counts[model] += 1

        # handle models missing for this view
        if not drop_if_missing:
            worst_rank = len(mv_list) + 1
            missing = set(model_names) - {m for m, _ in mv_list}
            for model in missing:
                rank_accum[model] += worst_rank
                rank_counts[model] += 1

    # finally compute mean rank
    mean_ranks = [
        (model, rank_accum[model] / rank_counts[model])
        for model in rank_accum
        if rank_counts[model] > 0
    ]

    # NEW: ensure every model in `model_names` is represented
    for m in model_names:
        if m not in {mr[0] for mr in mean_ranks}:
            mean_ranks.append((m, float("inf")))  # or a large number / np.nan

    mean_ranks.sort(key=lambda x: x[1])
    return mean_ranks


def compare_fvs_vs_llm(
    base_dir: str,
    model_names: List[str],
    view_numbers: List[int],
    metric: str = "psnr",
    *,
    higher_is_better: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Return two lists of training-view counts:
      1. views where *any* non-FVS/RS model outperforms FVS
      2. views where FVS beats *all* non-FVS/RS models

    Parameters
    ----------
    base_dir : str
        Root directory of experiments.
    model_names : List[str]
        All model folders to consider (must include 'fvs').
    view_numbers : List[int]
        Training-view counts to examine.
    metric : str, default 'psnr'
        Metric key inside each JSON’s 'results'.
    higher_is_better : bool, default True
        Use False when lower metric values are better (e.g. LPIPS).

    Returns
    -------
    (llm_better, fvs_better)
        llm_better : List[int]  – views where at least one LLM > FVS
        fvs_better : List[int]  – views where FVS > every LLM
    """
    # ---------------- collect metric values -----------------------------
    fvs_values: Dict[int, float] = {}
    llm_values: Dict[int, List[float]] = defaultdict(list)

    for model in model_names:
        for fp in generate_json_paths(base_dir, model, view_numbers):
            if not os.path.exists(fp):
                continue
            with open(fp, "r") as f:
                res = json.load(f).get("results", {})
            if metric not in res:
                continue

            view = int(fp.split("_")[-1].split(".")[0])
            if model == "fvs":
                fvs_values[view] = res[metric]
            elif model not in ("rs",):  # LLM bucket
                llm_values[view].append(res[metric])

    # ---------------- compare per view ----------------------------------
    llm_better: List[int] = []
    fvs_better: List[int] = []

    sign = 1 if higher_is_better else -1

    for view in view_numbers:
        if view not in fvs_values or view not in llm_values:
            # skip if data incomplete for the comparison
            continue

        fvs_val = fvs_values[view]
        best_llm = max(llm_values[view]) if higher_is_better else min(llm_values[view])

        if sign * best_llm > sign * fvs_val:
            llm_better.append(view)
        elif sign * fvs_val > sign * best_llm:
            fvs_better.append(view)
        # ties are ignored

    return llm_better, fvs_better


def compare_baseline_vs_llm(
    base_dir: str,
    model_names: List[str],
    view_numbers: List[int],
    baseline_model: str,  # 'fvs'  OR  'rs'
    metric: str = "psnr",
    *,
    higher_is_better: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Identify training-view counts where:

      • at least one "LLM" model (any model **not** in {'fvs','rs'})
        outperforms the baseline model;
      • the baseline beats **all** LLM models.

    Returns
    -------
    (llm_better, baseline_better)
        llm_better       : views where best-LLM > baseline
        baseline_better  : views where baseline > every LLM
    """

    if baseline_model not in ("fvs", "rs"):
        raise ValueError("baseline_model must be 'fvs' or 'rs'")

    # ------------------- gather values ---------------------------------
    base_vals: Dict[int, float] = {}
    llm_vals: Dict[int, List[float]] = defaultdict(list)

    for model in model_names:
        for fp in generate_json_paths(base_dir, model, view_numbers):
            if not os.path.exists(fp):
                continue
            with open(fp, "r") as f:
                res = json.load(f).get("results", {})
            if metric not in res:
                continue

            view = int(fp.split("_")[-1].split(".")[0])
            val = res[metric]

            if model == baseline_model:
                base_vals[view] = val
            elif model not in ("fvs", "rs"):  # LLM bucket
                llm_vals[view].append(val)

    # ------------------- compare per view ------------------------------
    sign = 1 if higher_is_better else -1

    llm_better: List[int] = []
    baseline_better: List[int] = []

    for view in view_numbers:
        if view not in base_vals or view not in llm_vals or not llm_vals[view]:
            # skip if baseline or LLM data missing for this view
            continue

        base_val = base_vals[view]
        best_llm = max(llm_vals[view]) if higher_is_better else min(llm_vals[view])

        if sign * best_llm > sign * base_val:  # LLM wins
            llm_better.append(view)
        elif sign * base_val > sign * best_llm:  # baseline wins
            baseline_better.append(view)
        # ties are ignored

    return llm_better, baseline_better


stats, diffs = analyze_metrics_by_view(
    base_dir="/local/home/hanwliu/tnt/M60/evaluation_splatfacto",
    model_names=[
        "clip_ViTB32",
        "clip_ViTL14",
        "clip_ViTL14_336px",
        "blip_ViTB16",
        "blip_large",
        "dinov2",
        "dinov2_large_fullres",
        "fvs",
        "rs",
    ],
    view_numbers=[5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    metric="psnr",
    return_diffs=True,
)

stats, diffs = analyze_metrics_by_view(
    base_dir="/local/home/hanwliu/tnt/M60/evaluation_splatfacto",
    model_names=[
        "FVS_clip_ViTB32",
        "FVS_clip_ViTL14",
        "FVS_clip_ViTL14_336px",
        "FVS_blip_ViTB16",
        "FVS_dinov2",
        "FVS_dinov2_large_fullres",
        "fvs",
        "rs",
    ],
    view_numbers=[5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    metric="psnr",
    return_diffs=True,
)

print("Per-view stats:")
for v in sorted(stats):
    s = stats[v]
    print(
        f"{v:4d}: max={s['max']['value']:.4f} ({s['max']['model']}) "
        f"min={s['min']['value']:.4f} ({s['min']['model']}) "
        f"mean={s['mean']:.4f}"
    )

print("\nSpans (max-min) descending:")
for v, span, max_m, min_m in diffs:
    print(f"view {v:4d}: span={span:.4f}   " f"max→{max_m}  min→{min_m}")

leaderboard = mean_rank_across_views(
    base_dir="/local/home/hanwliu/tnt/M60/evaluation_splatfacto",
    model_names=[
        "FVS_clip_ViTB32",
        "FVS_clip_ViTL14",
        "FVS_clip_ViTL14_336px",
        "FVS_blip_ViTB16",
        "FVS_dinov2",
        "FVS_dinov2_large_fullres",
        "fvs",
        "rs",
    ],
    view_numbers=[5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    metric="psnr",  # lower-is-better metric
    higher_is_better=True,
    drop_if_missing=False,  # treat missing as worst rank
)

print("Leaderboard (mean rank across views):")
for model, mr in leaderboard:
    print(f"{model:15s}  mean_rank = {mr:.2f}")

llm_win, fvs_win = compare_fvs_vs_llm(
    base_dir="/local/home/hanwliu/tnt/M60/evaluation_splatfacto",
    model_names=[
        "clip_ViTB32",
        "clip_ViTL14",
        "clip_ViTL14_336px",
        "blip_ViTB16",
        "blip_large",
        "dinov2",
        "dinov2_large_fullres",
        "fvs",
        "rs",
    ],
    view_numbers=[5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    metric="psnr",  # higher-is-better
    higher_is_better=True,
)

print("LLM beats FVS at views : ", llm_win)
print("FVS beats all LLMs at : ", fvs_win)


base_dir = "/local/home/hanwliu/tnt/M60/evaluation_splatfacto"
view_nums = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
all_models = [
    "FVS_clip_ViTB32",
    "FVS_clip_ViTL14",
    "FVS_clip_ViTL14_336px",
    "FVS_blip_ViTB16",
    "FVS_dinov2",
    "FVS_dinov2_large_fullres",
    "fvs",
    "rs",
]

# --- LLM vs FVS --------------------------------------------------------
llm_beats_fvs, fvs_beats_llm = compare_baseline_vs_llm(
    base_dir=base_dir,
    model_names=all_models,
    view_numbers=view_nums,
    baseline_model="fvs",
    metric="psnr",  # higher-is-better
    higher_is_better=True,
)

print("Views where an LLM beats FVS :", llm_beats_fvs)
print("Views where FVS beats all LLMs:", fvs_beats_llm)

# --- LLM vs RS ---------------------------------------------------------
llm_beats_rs, rs_beats_llm = compare_baseline_vs_llm(
    base_dir=base_dir,
    model_names=all_models,
    view_numbers=view_nums,
    baseline_model="rs",
    metric="psnr",
    higher_is_better=True,
)

print("\nViews where an LLM beats RS  :", llm_beats_rs)
print("Views where RS beats all LLMs :", rs_beats_llm)


# plot

import os, json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Assume generate_json_paths(base_dir, model_name, view_numbers) is defined
# --------------------------------------------------------------------------


def compute_llm_vs_baseline_deltas(
    base_dir: str,
    model_names: List[str],
    view_numbers: List[int],
    baseline_model: str,  # 'fvs'  OR  'rs'
    metric: str = "psnr",
    *,
    higher_is_better: bool = True,
) -> Dict[int, Tuple[float, float]]:
    """
    Return {view: (best_llm - baseline, worst_llm - baseline)}.
    Views with incomplete data are omitted.
    """

    if baseline_model not in ("fvs", "rs"):
        raise ValueError("baseline_model must be 'fvs' or 'rs'.")

    baseline_vals: Dict[int, float] = {}
    llm_vals: Dict[int, List[float]] = defaultdict(list)

    # ---------- gather ---------------------------------------------------
    for model in model_names:
        for fp in generate_json_paths(base_dir, model, view_numbers):
            if not os.path.exists(fp):
                continue
            with open(fp, "r") as f:
                res = json.load(f).get("results", {})
            if metric not in res:
                continue
            view = int(fp.split("_")[-1].split(".")[0])
            val = res[metric]

            if model == baseline_model:
                baseline_vals[view] = val
            elif model not in ("fvs", "rs"):
                llm_vals[view].append(val)

    # ---------- compute deltas ------------------------------------------
    sign = 1 if higher_is_better else -1
    deltas: Dict[int, Tuple[float, float]] = {}

    for v in view_numbers:
        if v not in baseline_vals or v not in llm_vals or not llm_vals[v]:
            continue

        base = baseline_vals[v]
        best_llm = max(llm_vals[v]) if higher_is_better else min(llm_vals[v])
        worst_llm = min(llm_vals[v]) if higher_is_better else max(llm_vals[v])

        deltas[v] = (best_llm - base, worst_llm - base)

    return deltas


def plot_deltas(
    deltas: Dict[int, Tuple[float, float]],
    *,
    baseline_label: str = "FVS",
    metric: str = "PSNR",
    colour_best: str = "tab:blue",
    colour_worst: str = "tab:red",
):
    """
    Two curves:
      • best-LLM – baseline
      • worst-LLM – baseline
    with small ‘highest’ / ‘lowest’ annotations.
    """
    views = sorted(deltas)
    best = [deltas[v][0] for v in views]
    worst = [deltas[v][1] for v in views]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(views, best, "-o", label=f"Best LLM − {baseline_label}", color=colour_best)
    ax.plot(
        views, worst, "-o", label=f"Worst LLM − {baseline_label}", color=colour_worst
    )

    # ── add tiny text labels next to extrema ────────────────────────────
    def tag(point_x, point_y, txt, colour, v_shift):
        ax.text(
            point_x,
            point_y + v_shift,
            txt,
            fontsize=7,
            color=colour,
            ha="center",
            va="bottom",
        )

    # best-curve annotations
    idx_max_best = int(np.argmax(best))
    idx_min_best = int(np.argmin(best))
    tag(views[idx_max_best], best[idx_max_best], "highest", colour_best, 0.10)
    tag(views[idx_min_best], best[idx_min_best], "lowest", colour_best, -0.15)

    # worst-curve annotations
    idx_max_worst = int(np.argmax(worst))
    idx_min_worst = int(np.argmin(worst))
    tag(views[idx_max_worst], worst[idx_max_worst], "highest", colour_worst, 0.10)
    tag(views[idx_min_worst], worst[idx_min_worst], "lowest", colour_worst, -0.15)

    # ── cosmetics ───────────────────────────────────────────────────────
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Training views")
    ax.set_ylabel(f"Δ {metric} (dB)" if metric.lower() == "psnr" else f"Δ {metric}")
    ax.set_title(f"LLM vs {baseline_label}: {metric}")
    ax.set_xticks(views)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()


def compute_rs_minus_fvs(
    base_dir: str,
    view_numbers: List[int],
    *,
    metric: str = "psnr",
    higher_is_better: bool = True,
) -> Dict[int, float]:
    """
    Return {view :  RS_metric − FVS_metric}.
    Views missing either metric are skipped.
    """
    vals: Dict[str, Dict[int, float]] = {"fvs": {}, "rs": {}}

    for model in ("fvs", "rs"):
        for fp in generate_json_paths(base_dir, model, view_numbers):
            if not os.path.exists(fp):
                continue
            with open(fp, "r") as f:
                res = json.load(f).get("results", {})
            if metric not in res:
                continue
            view = int(fp.split("_")[-1].split(".")[0])
            vals[model][view] = res[metric]

    diffs = {}
    for v in view_numbers:
        if v in vals["fvs"] and v in vals["rs"]:
            diffs[v] = vals["fvs"][v] - vals["rs"][v]

    # If metric is "lower-is-better" (e.g. LPIPS) but you still want
    # "positive ⇒ RS better", leave the formula as is – sign flips naturally.
    return diffs


def plot_rs_vs_fvs(
    diffs: Dict[int, float],
    *,
    metric: str = "PSNR",
    colour: str = "tab:green",
):
    views = sorted(diffs)
    deltas = [diffs[v] for v in views]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(views, deltas, "-o", color=colour, label="FVS − RS")

    # --- annotate highest & lowest -------------------------------------
    idx_high = int(np.argmax(deltas))
    idx_low = int(np.argmin(deltas))

    def tag(idx, text, v_shift):
        ax.text(
            views[idx],
            deltas[idx] + v_shift,
            text,
            fontsize=7,
            color=colour,
            ha="center",
            va="bottom",
        )

    tag(idx_high, "highest", 0.10)
    tag(idx_low, "lowest", -0.15)

    # --- cosmetics ------------------------------------------------------
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Training views")
    ax.set_ylabel(f"Δ {metric} (RS − FVS)")
    ax.set_title(f"RS vs FVS: {metric} difference")
    ax.set_xticks(views)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
base_dir = "/local/home/hanwliu/tnt/M60/evaluation_splatfacto"
view_nums = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
all_models = [
    "FVS_clip_ViTB32",
    "FVS_clip_ViTL14",
    "FVS_clip_ViTL14_336px",
    "FVS_blip_ViTB16",
    "FVS_dinov2",
    "FVS_dinov2_large_fullres",
    "fvs",
    "rs",
]

# 1)  FVS baseline
deltas_fvs = compute_llm_vs_baseline_deltas(
    base_dir,
    all_models,
    view_nums,
    baseline_model="fvs",
    metric="psnr",
    higher_is_better=True,
)
plot_deltas(deltas_fvs, baseline_label="FVS", metric="PSNR")

# 2)  RS baseline
deltas_rs = compute_llm_vs_baseline_deltas(
    base_dir,
    all_models,
    view_nums,
    baseline_model="rs",
    metric="psnr",
    higher_is_better=True,
)
plot_deltas(deltas_rs, baseline_label="RS", metric="PSNR")


base_dir = "/local/home/hanwliu/tnt/M60/evaluation_splatfacto"
view_nums = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

diffs = compute_rs_minus_fvs(
    base_dir,
    view_nums,
    metric="psnr",
    higher_is_better=True,  # PSNR – higher means better quality
)

plot_rs_vs_fvs(diffs, metric="PSNR")
