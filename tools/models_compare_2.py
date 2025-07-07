import json, os
from statistics import mean
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Dict
import pandas as pd

# ==================== CONFIGURATION ============================
BASE_DIR = "/local/home/hanwliu/table/evaluation_splatfacto"
VIEW_NUMS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
ALL_MODELS = [
    "clip_ViT-B32",
    "clip_ViTL14",
    "clip_ViTL14_336px",
    "blip_ViTB16",
    "dinov2",
    "dinov2_large_fullres",
    "fvs",
    "rs",
]
VANILLA_MODELS = [
    "clip_ViT-B32",
    "clip_ViTL14",
    "clip_ViTL14_336px",
    "blip_ViTB16",
    "dinov2",
    "dinov2_large_fullres",
    "fvs",
    "rs",
]
METRIC = "psnr"
HIGHER_IS_BETTER = True
RETURN_TYPE = "rank"


# ------------------------ Functions -----------------------
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
):
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
):
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

    rank_accum: Dict[str, float] = defaultdict(float)
    rank_counts: Dict[str, int] = defaultdict(int)

    for view, mv_list in values_by_view.items():
        if not mv_list:
            continue
        mv_list.sort(key=lambda x: x[1], reverse=higher_is_better)
        for rank, (model, _) in enumerate(mv_list, start=1):
            rank_accum[model] += rank
            rank_counts[model] += 1

        if not drop_if_missing:
            worst_rank = len(mv_list) + 1
            missing = set(model_names) - {m for m, _ in mv_list}
            for model in missing:
                rank_accum[model] += worst_rank
                rank_counts[model] += 1

    mean_ranks = [
        (model, rank_accum[model] / rank_counts[model])
        for model in rank_accum
        if rank_counts[model] > 0
    ]

    for m in model_names:
        if m not in {mr[0] for mr in mean_ranks}:
            mean_ranks.append((m, float("inf")))

    mean_ranks.sort(key=lambda x: x[1])
    return mean_ranks


def compare_fvs_vs_llm(
    base_dir: str,
    model_names: List[str],
    view_numbers: List[int],
    metric: str = "psnr",
    *,
    higher_is_better: bool = True,
):
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
            elif model not in ("rs",):
                llm_values[view].append(res[metric])

    llm_better, fvs_better = [], []
    sign = 1 if higher_is_better else -1

    for view in view_numbers:
        if view not in fvs_values or view not in llm_values:
            continue
        fvs_val = fvs_values[view]
        best_llm = max(llm_values[view]) if higher_is_better else min(llm_values[view])
        if sign * best_llm > sign * fvs_val:
            llm_better.append(view)
        elif sign * fvs_val > sign * best_llm:
            fvs_better.append(view)

    return llm_better, fvs_better


def compare_baseline_vs_llm(
    base_dir: str,
    model_names: List[str],
    view_numbers: List[int],
    baseline_model: str,
    metric: str = "psnr",
    *,
    higher_is_better: bool = True,
):
    if baseline_model not in ("fvs", "rs"):
        raise ValueError("baseline_model must be 'fvs' or 'rs'")
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
            elif model not in ("fvs", "rs"):
                llm_vals[view].append(val)

    sign = 1 if higher_is_better else -1
    llm_better, baseline_better = [], []

    for view in view_numbers:
        if view not in base_vals or view not in llm_vals or not llm_vals[view]:
            continue
        base_val = base_vals[view]
        best_llm = max(llm_vals[view]) if higher_is_better else min(llm_vals[view])
        if sign * best_llm > sign * base_val:
            llm_better.append(view)
        elif sign * base_val > sign * best_llm:
            baseline_better.append(view)

    return llm_better, baseline_better


def compute_llm_vs_baseline_deltas(
    base_dir: str,
    model_names: List[str],
    view_numbers: List[int],
    baseline_model: str,
    metric: str = "psnr",
    *,
    higher_is_better: bool = True,
):
    if baseline_model not in ("fvs", "rs"):
        raise ValueError("baseline_model must be 'fvs' or 'rs'.")

    baseline_vals: Dict[int, float] = {}
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
                baseline_vals[view] = val
            elif model not in ("fvs", "rs"):
                llm_vals[view].append(val)

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
    views = sorted(deltas)
    best = [deltas[v][0] for v in views]
    worst = [deltas[v][1] for v in views]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(views, best, "-o", label=f"Best LLM − {baseline_label}", color=colour_best)
    ax.plot(
        views, worst, "-o", label=f"Worst LLM − {baseline_label}", color=colour_worst
    )

    def tag(idx, curve, label, color, offset):
        ax.text(
            views[idx],
            curve[idx] + offset,
            label,
            fontsize=7,
            color=color,
            ha="center",
            va="bottom",
        )

    tag(np.argmax(best), best, "highest", colour_best, 0.10)
    tag(np.argmin(best), best, "lowest", colour_best, -0.15)
    tag(np.argmax(worst), worst, "highest", colour_worst, 0.10)
    tag(np.argmin(worst), worst, "lowest", colour_worst, -0.15)

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
):
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

    diffs = {
        v: vals["fvs"][v] - vals["rs"][v]
        for v in view_numbers
        if v in vals["fvs"] and v in vals["rs"]
    }
    return diffs


def plot_rs_vs_fvs(
    diffs: Dict[int, float], *, metric: str = "PSNR", colour: str = "tab:green"
):
    views = sorted(diffs)
    deltas = [diffs[v] for v in views]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(views, deltas, "-o", color=colour, label="FVS − RS")

    def tag(idx, label, offset):
        ax.text(
            views[idx],
            deltas[idx] + offset,
            label,
            fontsize=7,
            color=colour,
            ha="center",
            va="bottom",
        )

    tag(np.argmax(deltas), "highest", 0.10)
    tag(np.argmin(deltas), "lowest", -0.15)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Training views")
    ax.set_ylabel(f"Δ {metric} (FVS − RS)")
    ax.set_title(f"FVS vs RS: {metric} difference")
    ax.set_xticks(views)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()


def compare_fvs_vs_rs(
    base_dir: str,
    view_numbers: List[int],
    metric: str = "psnr",
    *,
    higher_is_better: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Compare FVS and RS models per view.

    Returns:
        - List of views where FVS > RS
        - List of views where RS > FVS
    """
    fvs_vals: Dict[int, float] = {}
    rs_vals: Dict[int, float] = {}

    for model in ("fvs", "rs"):
        for fp in generate_json_paths(base_dir, model, view_numbers):
            if not os.path.exists(fp):
                continue
            with open(fp, "r") as f:
                res = json.load(f).get("results", {})
            if metric not in res:
                continue
            view = int(fp.split("_")[-1].split(".")[0])
            if model == "fvs":
                fvs_vals[view] = res[metric]
            else:
                rs_vals[view] = res[metric]

    fvs_beats, rs_beats = [], []
    sign = 1 if higher_is_better else -1

    for view in view_numbers:
        if view not in fvs_vals or view not in rs_vals:
            continue
        fvs_score, rs_score = fvs_vals[view], rs_vals[view]
        if sign * fvs_score > sign * rs_score:
            fvs_beats.append(view)
        elif sign * rs_score > sign * fvs_score:
            rs_beats.append(view)

    return fvs_beats, rs_beats


def compute_rank_matrix(
    base_dir: str,
    model_names: List[str],
    view_numbers: List[int],
    metric: str = "psnr",
    *,
    higher_is_better: bool = True,
    drop_if_missing: bool = False,
) -> Dict[str, Dict[int, Optional[int]]]:
    """
    Return a nested dict: rank_matrix[model][view] -> rank (1 = best).

    If `drop_if_missing` is False (default), any model that lacks a score for
    a given view is assigned `worst_rank = (#models with scores) + 1`.
    If `drop_if_missing` is True, missing entries are set to None.

    Example return structure (for 3 models and 2 views):
        {
          "dinov2": {5: 1, 10: 2},
          "fvs":    {5: 3, 10: 1},
          "rs":     {5: 2, 10: None},
        }
    """
    # Collect scores per view
    scores_by_view: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    for model in model_names:
        for fp in generate_json_paths(base_dir, model, view_numbers):
            if not os.path.exists(fp):
                continue
            with open(fp, "r") as f:
                res = json.load(f).get("results", {})
            if metric not in res:
                continue
            view = int(fp.split("_")[-1].split(".")[0])
            scores_by_view[view].append((model, res[metric]))

    # Initialise rank matrix with None
    rank_matrix: Dict[str, Dict[int, Optional[int]]] = {
        m: {v: None for v in view_numbers} for m in model_names
    }

    # Compute ranks per view
    for view, mv_list in scores_by_view.items():
        if not mv_list:
            continue
        mv_list.sort(key=lambda x: x[1], reverse=higher_is_better)
        for rank, (model, _) in enumerate(mv_list, start=1):
            rank_matrix[model][view] = rank

        # handle missing models
        if not drop_if_missing:
            worst_rank = len(mv_list) + 1
            for model in set(model_names) - {m for m, _ in mv_list}:
                rank_matrix[model][view] = worst_rank

    return rank_matrix


def rank_matrix_to_latex(
    rank_matrix: Dict[str, Dict[int, Optional[int]]],
    view_numbers: List[int],
    *,
    caption: str = "Per-view rank of each model (1 = best).",
    label: str = "tab:rank_matrix",
) -> str:
    """
    Convert the rank matrix to a LaTeX tabular surrounded by booktabs lines.
    """
    df = pd.DataFrame(rank_matrix).T[view_numbers]
    df.index.name = "Model"
    df = df.fillna("-")

    latex = df.to_latex(
        index=True,
        na_rep="-",
        column_format="l" + "c" * len(view_numbers),  # left, then centered cols
        escape=False,
        caption=caption,
        label=label,
        multicolumn=True,
        multicolumn_format="c",
        bold_rows=False,
    )
    return latex


def compute_metric_matrix(
    base_dir: str,
    model_names: List[str],
    view_numbers: List[int],
    metric: str = "psnr",
    *,
    higher_is_better: bool = True,
    return_type: str = "rank",  # "rank" or "value"
    drop_if_missing: bool = False,
) -> Dict[str, Dict[int, Optional[float]]]:
    """
    Return a matrix of either ranks or raw metric values.

    return_type = "rank" → 1 is best
    return_type = "value" → shows raw PSNR/SSIM/etc.
    """
    assert return_type in ("rank", "value")

    scores_by_view: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    for model in model_names:
        for fp in generate_json_paths(base_dir, model, view_numbers):
            if not os.path.exists(fp):
                continue
            with open(fp, "r") as f:
                res = json.load(f).get("results", {})
            if metric not in res:
                continue
            view = int(fp.split("_")[-1].split(".")[0])
            scores_by_view[view].append((model, res[metric]))

    matrix: Dict[str, Dict[int, Optional[float]]] = {
        m: {v: None for v in view_numbers} for m in model_names
    }

    for view, mv_list in scores_by_view.items():
        if not mv_list:
            continue

        if return_type == "rank":
            mv_list.sort(key=lambda x: x[1], reverse=higher_is_better)
            for rank, (model, _) in enumerate(mv_list, start=1):
                matrix[model][view] = rank
            if not drop_if_missing:
                worst_rank = len(mv_list) + 1
                for model in set(model_names) - {m for m, _ in mv_list}:
                    matrix[model][view] = worst_rank
        else:  # return_type == "value"
            for model, val in mv_list:
                matrix[model][view] = val

    return matrix


# ------------------------ MAIN EXECUTION ------------------------
if __name__ == "__main__":
    stats, diffs = analyze_metrics_by_view(
        BASE_DIR, VANILLA_MODELS, VIEW_NUMS, METRIC, return_diffs=True
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
        print(f"view {v:4d}: span={span:.4f}   max→{max_m}  min→{min_m}")

    leaderboard = mean_rank_across_views(
        BASE_DIR, ALL_MODELS, VIEW_NUMS, METRIC, higher_is_better=HIGHER_IS_BETTER
    )
    print("\nLeaderboard (mean rank across views):")
    for model, mr in leaderboard:
        print(f"{model:20s}  mean_rank = {mr:.2f}")

    llm_win, fvs_win = compare_fvs_vs_llm(
        BASE_DIR, VANILLA_MODELS, VIEW_NUMS, METRIC, higher_is_better=HIGHER_IS_BETTER
    )
    print("\nLLM beats FVS at views :", llm_win)
    print("FVS beats all LLMs at  :", fvs_win)

    llm_beats_fvs, fvs_beats_llm = compare_baseline_vs_llm(
        BASE_DIR,
        ALL_MODELS,
        VIEW_NUMS,
        "fvs",
        METRIC,
        higher_is_better=HIGHER_IS_BETTER,
    )
    llm_beats_rs, rs_beats_llm = compare_baseline_vs_llm(
        BASE_DIR, ALL_MODELS, VIEW_NUMS, "rs", METRIC, higher_is_better=HIGHER_IS_BETTER
    )
    fvs_beats_rs, rs_beats_fvs = compare_fvs_vs_rs(
        BASE_DIR, VIEW_NUMS, METRIC, higher_is_better=HIGHER_IS_BETTER
    )

    print("\nViews where an LLM beats FVS :", llm_beats_fvs)
    print("Views where FVS beats all LLMs:", fvs_beats_llm)
    print("\nViews where an LLM beats RS  :", llm_beats_rs)
    print("Views where RS beats all LLMs :", rs_beats_llm)
    print("\nViews where FVS beats RS:", fvs_beats_rs)
    print("Views where RS beats FVS:", rs_beats_fvs)

    plot_deltas(
        compute_llm_vs_baseline_deltas(
            BASE_DIR,
            ALL_MODELS,
            VIEW_NUMS,
            "fvs",
            METRIC,
            higher_is_better=HIGHER_IS_BETTER,
        ),
        baseline_label="FVS",
        metric=METRIC,
    )

    plot_deltas(
        compute_llm_vs_baseline_deltas(
            BASE_DIR,
            ALL_MODELS,
            VIEW_NUMS,
            "rs",
            METRIC,
            higher_is_better=HIGHER_IS_BETTER,
        ),
        baseline_label="RS",
        metric=METRIC,
    )

    plot_rs_vs_fvs(
        compute_rs_minus_fvs(
            BASE_DIR, VIEW_NUMS, metric=METRIC, higher_is_better=HIGHER_IS_BETTER
        ),
        metric=METRIC,
    )

    rank_matrix = compute_rank_matrix(
        BASE_DIR,
        ALL_MODELS,
        VIEW_NUMS,
        METRIC,
        higher_is_better=HIGHER_IS_BETTER,
        drop_if_missing=False,
    )

    latex_code = rank_matrix_to_latex(rank_matrix, VIEW_NUMS)
    print(latex_code)

    psnr_matrix = compute_metric_matrix(
        BASE_DIR,
        ALL_MODELS,
        VIEW_NUMS,
        metric=METRIC,
        return_type=RETURN_TYPE,
        higher_is_better=True,
    )
    latex_code = rank_matrix_to_latex(psnr_matrix, VIEW_NUMS)
    print("\n", latex_code)
