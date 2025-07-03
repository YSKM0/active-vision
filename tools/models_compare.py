# import matplotlib.cm as cm
# import numpy as np
# import os
# import json
# import matplotlib.pyplot as plt
# from collections import defaultdict

# def generate_json_paths(base_dir, model_name, view_numbers):
#     """
#     Generate file paths based on base directory, model name, and view numbers.
#     Automatically handles the prefix logic:
#     - If model_name is 'fvs' or 'rs', it uses the same as the prefix.
#     - Otherwise, it defaults to 'vlm'.
#     """
#     prefix = model_name if model_name in ['fvs', 'rs'] else 'vlm'
#     file_paths = []

#     for view in view_numbers:
#         file_name = f"{prefix}_{view}/{prefix}_{view}.json"
#         file_path = os.path.join(base_dir, model_name, file_name)
#         file_paths.append(file_path)

#     return file_paths

# def plot_by_training_views(base_dir, model_names, view_numbers, metrics=['psnr'], tight_ylim=False, show_std=True, line_alpha=1.0):
#     """
#     Plots the metrics trends across different training views for multiple model names.

#     Parameters:
#     - base_dir: str, path to the base directory
#     - model_names: list of str, model identifiers
#     - view_numbers: list of int, training view numbers
#     - metrics: list of str, metrics to plot
#     - tight_ylim: bool, whether to set tight y-axis limits
#     - show_std: bool, whether to show standard deviation as error bars
#     - line_alpha: float, transparency of the plot lines (0.0 to 1.0)
#     """
#     data = {metric: defaultdict(dict) for metric in metrics}
#     all_views = set()

#     for model_name in model_names:
#         json_files = generate_json_paths(base_dir, model_name, view_numbers)

#         for file in json_files:
#             if not os.path.exists(file):
#                 print(f"File not found: {file}")
#                 continue

#             with open(file, 'r') as f:
#                 d = json.load(f)
#                 exp_name = d.get('experiment_name', os.path.basename(file))
#                 results = d.get('results', {})

#                 method = model_name
#                 view = int(file.split('_')[-1].split('.')[0])
#                 all_views.add(view)

#                 for metric in metrics:
#                     val = results.get(metric)
#                     std = results.get(f"{metric}_std", 0.0)  # Default std to 0 if missing
#                     if val is None:
#                         print(f"Missing {metric} in {exp_name}")
#                         continue
#                     data[metric][method][view] = (val, std)

#     all_views = sorted(all_views)
#     plt.figure(figsize=(12, 6))

#     colormap = cm.get_cmap('tab20')
#     num_lines = sum(len(data[metric]) for metric in metrics)
#     color_indices = np.linspace(0, 1, num_lines)

#     color_map = {}
#     color_count = 0
#     for metric in metrics:
#         for method in data[metric]:
#             color_map[(metric, method)] = colormap(color_indices[color_count])
#             color_count += 1

#     for metric in metrics:
#         for method, view_map in data[metric].items():
#             y, yerr, x = [], [], []

#             for v in all_views:
#                 if v in view_map:
#                     val, std = view_map[v]
#                     y.append(val)
#                     yerr.append(std)
#                     x.append(v)

#             if x:
#                 plt.errorbar(
#                     x, y,
#                     yerr=yerr if show_std else None,
#                     fmt='-o',
#                     capsize=4,
#                     label=f"{method.upper()} - {metric.upper()}",
#                     color=color_map[(metric, method)],
#                     alpha=line_alpha
#                 )

#         if tight_ylim:
#             all_vals = [val for view_map in data[metric].values() for val, _ in view_map.values()]
#             if all_vals:
#                 min_val = min(all_vals)
#                 max_val = max(all_vals)
#                 margin = (max_val - min_val) * 0.2
#                 plt.ylim(min_val - margin, max_val + margin)

#     plt.xlabel("Training Views")
#     plt.ylabel("Metric Value")
#     plt.title("Metric Trends Across Training Views")
#     plt.xticks(all_views)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(loc='best', fontsize='small')
#     plt.tight_layout()
#     plt.show()


# # Example usage
# base_dir = "/local/home/hanwliu/lab_record/evaluation_splatfacto"
# model_names = ["clip_ViTB32", "clip_ViTL14", "clip_ViTL14_336px", "blip_ViTB16", "blip_large", "dinov2", "fvs", "rs"]
# # view_numbers = list(range(5, 125, 5)) , "fvs", "rs"    "clip_ViTB32", "blip_ViTB16", "dinov2"
# view_numbers = [
#     5,      10,     15,     20,     25,     30,     40,     50,     60,
#     70,     80,     90,     100,    110,    120
# ]

# plot_by_training_views(
#     base_dir=base_dir,
#     model_names=model_names,
#     view_numbers=view_numbers,
#     metrics=['lpips'], # lpips ssim psnr
#     tight_ylim=False,
#     show_std=False,
#     line_alpha = 0.65
# )


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from collections import defaultdict
from typing import List, Optional


def generate_json_paths(base_dir: str, model_name: str, view_numbers: List[int]):
    prefix = model_name if model_name in ["fvs", "rs"] else "vlm"
    return [
        os.path.join(base_dir, model_name, f"{prefix}_{v}/{prefix}_{v}.json")
        for v in view_numbers
    ]


def plot_by_training_views(
    base_dir: str,
    model_names: List[str],
    view_numbers: List[int],
    metrics: List[str] = ["psnr"],
    tight_ylim: bool = False,
    show_std: bool = True,
    line_alpha: float = 1.0,
    show_upper_bound: bool = False,
    upper_bound_path: Optional[str] = None,
):
    # ── read upper-bound json ────────────────────────────────────────────
    if show_upper_bound:
        ub_path = (
            upper_bound_path
            if upper_bound_path is not None
            else os.path.join(base_dir, "full", "full.json")
        )
        if not os.path.exists(ub_path):
            raise FileNotFoundError(
                f"Upper-bound json not found: {ub_path}. "
                "Provide a correct `upper_bound_path` or turn off `show_upper_bound`."
            )
        with open(ub_path, "r") as f:
            ub_results = json.load(f).get("results", {})
        upper_values = {m: ub_results.get(m) for m in metrics}
    # --------------------------------------------------------------------

    data = {m: defaultdict(dict) for m in metrics}
    all_views = set()

    for model in model_names:
        for fp in generate_json_paths(base_dir, model, view_numbers):
            if not os.path.exists(fp):
                print(f"[warn] file not found: {fp}")
                continue
            with open(fp, "r") as f:
                exp = json.load(f)
            view = int(fp.split("_")[-1].split(".")[0])
            all_views.add(view)
            for m in metrics:
                val = exp["results"].get(m)
                std = exp["results"].get(f"{m}_std", 0.0)
                if val is None:
                    print(f"[warn] missing {m} in {exp.get('experiment_name', fp)}")
                    continue
                data[m][model][view] = (val, std)

    all_views = sorted(all_views)

    # ── plotting ────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 6))
    cmap = cm.get_cmap("tab20")
    colour_iter = iter(cmap(np.linspace(0, 1, sum(len(data[m]) for m in metrics))))
    colour_map = {(m, mdl): next(colour_iter) for m in metrics for mdl in data[m]}

    ax = plt.gca()
    ub_ticks = []  # collect UB values we add as yticks

    for m in metrics:
        for mdl, vmap in data[m].items():
            xs, ys, yerr = [], [], []
            for v in all_views:
                if v in vmap:
                    val, std = vmap[v]
                    xs.append(v)
                    ys.append(val)
                    yerr.append(std)
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr if show_std else None,
                    fmt="-o",
                    capsize=4,
                    color=colour_map[(m, mdl)],
                    alpha=line_alpha,
                    label=f"{mdl.upper()} – {m.upper()}",
                )

        if show_upper_bound and upper_values.get(m) is not None:
            ub_val = upper_values[m]
            ax.axhline(
                y=ub_val,
                linestyle="--",
                linewidth=1.5,
                color="k",
                alpha=0.7,
                label=f"Upper Bound – {m.upper()}",
            )
            ub_ticks.append(ub_val)

        if tight_ylim:
            vals = [val for mdl_map in data[m].values() for val, _ in mdl_map.values()]
            if show_upper_bound and upper_values.get(m) is not None:
                vals.append(upper_values[m])
            if vals:
                pad = (max(vals) - min(vals)) * 0.2
                ax.set_ylim(min(vals) - pad, max(vals) + pad)

    # ── add UB value(s) to y-axis ticks ─────────────────────────────────
    if ub_ticks:
        yticks = list(ax.get_yticks())
        for ub in ub_ticks:
            if ub not in yticks:
                yticks.append(ub)
        yticks_sorted = sorted(yticks)
        ax.set_yticks(yticks_sorted)
        # custom labels: mark UB tick(s)
        labels = []
        for t in yticks_sorted:
            if any(abs(t - ub) < 1e-9 for ub in ub_ticks):
                labels.append(f"{t:.2f} (UB)")
            else:
                labels.append(f"{t:.2f}")
        ax.set_yticklabels(labels)

    # ── cosmetics ───────────────────────────────────────────────────────
    ax.set_xlabel("Training Views")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metric Trends Across Training Views")
    ax.set_xticks(all_views)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize="small")
    plt.tight_layout()
    plt.show()


# plot_by_training_views(
#     base_dir="/local/home/hanwliu/lab_record/evaluation_splatfacto",
#     model_names=["FVS_clip_ViTB32", "FVS_clip_ViTL14", "FVS_clip_ViTL14_336px",
#                  "FVS_blip_ViTB16", "FVS_dinov2", "FVS_dinov2_large_fullres", "fvs", "rs"],
#     view_numbers=[5,10,15,20,25,30,40,50,60,70,80,90,100,110,120],
#     metrics=['psnr'],
#     tight_ylim=False,
#     show_std=False,
#     line_alpha=0.65,
#     show_upper_bound=True,
#     upper_bound_path="/local/home/hanwliu/lab_record/evaluation_splatfacto/full/full.json",
# )

# plot_by_training_views(
#     base_dir="/local/home/hanwliu/tnt/M60/evaluation_splatfacto",
#     model_names=["FVS_clip_ViTB32", "FVS_clip_ViTL14", "FVS_clip_ViTL14_336px",
#                  "FVS_blip_ViTB16", "FVS_dinov2", "FVS_dinov2_large_fullres", "fvs", "rs"],  #["fvs", "rs"],
#     view_numbers=[5,10,15,20,25,30,40,50,60,70,80,90,100,110,120],
#     metrics=['ssim'],
#     tight_ylim=False,
#     show_std=False,
#     line_alpha=0.65,
#     show_upper_bound=True,
#     upper_bound_path="/local/home/hanwliu/tnt/M60/evaluation_splatfacto/full/full.json",
# )

plot_by_training_views(
    base_dir="/media/hanwliu/HanwenDisk/ActiveVision/lab_record/evaluation_splatfacto",
    model_names=[
        "FVS_clip_ViTB32",
        "clip_ViTL14",
        "FVS_clip_ViTL14_336px",
        "FVS_blip_ViTB16",
        "FVS_dinov2",
        "FVS_dinov2_large_fullres",
        "FVS_VLM_clip_ViTL14",
        "fvs",
        "rs",
    ],
    view_numbers=[5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    metrics=["psnr"],
    tight_ylim=False,
    show_std=False,
    line_alpha=0.65,
    show_upper_bound=True,
    upper_bound_path="/media/hanwliu/HanwenDisk/ActiveVision/lab_record/evaluation_splatfacto/full/full.json",
)
