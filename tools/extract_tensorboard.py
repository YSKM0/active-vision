import os
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_final_value(events):
    if events:
        return events[-1].value
    return None


def extract_tensorboard_metrics(event_file, tags):
    acc = EventAccumulator(event_file)
    acc.Reload()
    results = {}
    for tag in tags:
        try:
            events = acc.Scalars(tag)
            val = extract_final_value(events)
            if val is not None:
                results[tag] = val
        except KeyError:
            print(f"Tag {tag} not found in {event_file}")
    return results


def parse_method_and_view_from_path(path):
    for part in path.split(os.sep):
        if "_" in part:
            try:
                method, view = part.split("_")
                view = int(view)
                return method.lower(), view
            except ValueError:
                continue
    return None, None


def plot_tensorboard_by_training_views(
    event_files, metrics=["psnr"], tight_ylim=False, mode="eval"  # 'eval' or 'train'
):
    tag_map = {
        "eval": {
            "psnr": "Eval Images Metrics Dict (all images)/psnr",
            "ssim": "Eval Images Metrics Dict (all images)/ssim",
            "lpips": "Eval Images Metrics Dict (all images)/lpips",
        },
        "train": {
            "psnr": "Train Metrics Dict/psnr",  # Corrected
            "ssim": "Train Metrics Dict/ssim",  # If it exists
            "lpips": "Train Metrics Dict/lpips",  # If it exists
        },
    }

    if mode not in tag_map:
        raise ValueError("mode must be 'eval' or 'train'")

    selected_tags = {m: tag_map[mode][m] for m in metrics}

    data = {m: defaultdict(dict) for m in metrics}
    all_views = set()

    for file in event_files:
        method, view = parse_method_and_view_from_path(file)
        if method is None or view is None:
            print(f"Could not parse method/view from {file}")
            continue

        all_views.add(view)
        result = extract_tensorboard_metrics(file, selected_tags.values())

        for metric, tag in selected_tags.items():
            val = result.get(tag)
            if val is not None:
                data[metric][method][view] = val

    all_views = sorted(all_views)
    colors = ["blue", "green", "red", "orange", "purple", "cyan"]
    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        for j, (method, view_map) in enumerate(data[metric].items()):
            x, y = [], []
            for v in all_views:
                if v in view_map:
                    x.append(v)
                    y.append(view_map[v])
            if x and y:
                plt.plot(
                    x,
                    y,
                    "-o",
                    label=f"{method.upper()} - {metric.upper()} ({mode})",
                    color=colors[(i * 3 + j) % len(colors)],
                )

        if tight_ylim:
            all_vals = [
                val for view_map in data[metric].values() for val in view_map.values()
            ]
            if all_vals:
                min_val = min(all_vals)
                max_val = max(all_vals)
                margin = (max_val - min_val) * 0.2
                plt.ylim(min_val - margin, max_val + margin)
            else:
                print(
                    f"[Warning] No data available for metric '{metric}' in mode '{mode}', skipping y-limit adjustment."
                )

    plt.xlabel("Training Views")
    plt.ylabel("Metric Value")
    plt.title(f"{mode.capitalize()} Metric Trends Across Training Views")
    plt.xticks(all_views)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


event_files = [
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_5/splatfacto/2025-04-30_183530/events.out.tfevents.1746030933.cnb-d102-57.inf.ethz.ch.946803.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_10/splatfacto/2025-04-30_185112/events.out.tfevents.1746031875.cnb-d102-57.inf.ethz.ch.947324.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_15/splatfacto/2025-04-30_193629/events.out.tfevents.1746034592.cnb-d102-57.inf.ethz.ch.959432.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_20/splatfacto/2025-04-30_195344/events.out.tfevents.1746035627.cnb-d102-57.inf.ethz.ch.960012.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_25/splatfacto/2025-04-30_203809/events.out.tfevents.1746038293.cnb-d102-57.inf.ethz.ch.972599.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_30/splatfacto/2025-04-30_222226/events.out.tfevents.1746044549.cnb-d102-57.inf.ethz.ch.1009687.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_40/splatfacto/2025-04-30_224249/events.out.tfevents.1746045772.cnb-d102-57.inf.ethz.ch.1010452.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_50/splatfacto/2025-04-30_230312/events.out.tfevents.1746046995.cnb-d102-57.inf.ethz.ch.1011407.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_60/splatfacto/2025-04-30_233600/events.out.tfevents.1746048963.cnb-d102-57.inf.ethz.ch.1023617.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_70/splatfacto/2025-04-30_235509/events.out.tfevents.1746050112.cnb-d102-57.inf.ethz.ch.1024318.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_80/splatfacto/2025-05-01_001328/events.out.tfevents.1746051212.cnb-d102-57.inf.ethz.ch.1035829.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_90/splatfacto/2025-05-01_003922/events.out.tfevents.1746052766.cnb-d102-57.inf.ethz.ch.1036661.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_100/splatfacto/2025-05-01_005921/events.out.tfevents.1746053965.cnb-d102-57.inf.ethz.ch.1037391.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_110/splatfacto/2025-05-01_102351/events.out.tfevents.1746087834.cnb-d102-57.inf.ethz.ch.1161060.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/fvs_120/splatfacto/2025-05-01_104228/events.out.tfevents.1746088951.cnb-d102-57.inf.ethz.ch.1161799.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_5/splatfacto/2025-05-01_110210/events.out.tfevents.1746090133.cnb-d102-57.inf.ethz.ch.1162485.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_10/splatfacto/2025-05-01_111932/events.out.tfevents.1746091175.cnb-d102-57.inf.ethz.ch.1174098.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_15/splatfacto/2025-05-01_121358/events.out.tfevents.1746094442.cnb-d102-57.inf.ethz.ch.1186695.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_20/splatfacto/2025-05-01_123246/events.out.tfevents.1746095569.cnb-d102-57.inf.ethz.ch.1187409.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_25/splatfacto/2025-05-01_124949/events.out.tfevents.1746096593.cnb-d102-57.inf.ethz.ch.1187959.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_30/splatfacto/2025-05-01_131031/events.out.tfevents.1746097834.cnb-d102-57.inf.ethz.ch.1199628.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_40/splatfacto/2025-05-01_133838/events.out.tfevents.1746099521.cnb-d102-57.inf.ethz.ch.1200463.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_50/splatfacto/2025-05-01_135639/events.out.tfevents.1746100602.cnb-d102-57.inf.ethz.ch.1201161.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_60/splatfacto/2025-05-01_141804/events.out.tfevents.1746101887.cnb-d102-57.inf.ethz.ch.1213021.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_70/splatfacto/2025-05-01_143853/events.out.tfevents.1746103136.cnb-d102-57.inf.ethz.ch.1213758.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_80/splatfacto/2025-05-01_150243/events.out.tfevents.1746104567.cnb-d102-57.inf.ethz.ch.1214728.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_90/splatfacto/2025-05-01_153110/events.out.tfevents.1746106273.cnb-d102-57.inf.ethz.ch.1226538.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_100/splatfacto/2025-05-01_155539/events.out.tfevents.1746107742.cnb-d102-57.inf.ethz.ch.1227374.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_110/splatfacto/2025-05-01_163128/events.out.tfevents.1746109891.cnb-d102-57.inf.ethz.ch.1239253.0",
    "/local/home/hanwliu/lab_record/train_splatfacto/rs_120/splatfacto/2025-05-01_165702/events.out.tfevents.1746111425.cnb-d102-57.inf.ethz.ch.1240096.0",
]

plot_tensorboard_by_training_views(
    event_files, metrics=["psnr"], tight_ylim=True, mode="train"  # or 'eval'
)
