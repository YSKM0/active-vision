#!/usr/bin/env python3
"""
Sample Efficiency Score (SES) analyser penalised mean version

* Coverage - % of Q* levels hit (score present, not "—")
* Mean SES - mean of SES with 130 (penality) substituted for misses
* Best-level wins - # of the models being the best
* Penalty-weighted area sum of SES + 150 for every missed level
"""

import argparse, re, math, statistics, os, sys
from typing import Dict, List, Mapping

# Config
PENALTY_MISS_AREA = 150
PENALTY_MISS_MEAN = 150

INPUT_PATHS = ["lab_ses.txt", "table_ses.txt"]


# Utlities
def parse_ses_file(path: str) -> List[Mapping[str, float]]:
    """
    Returns a list (one item per Q* target level).
    Each item is a {model -> score or None} mapping.
    """
    level_hdr = re.compile(r"^SES target")
    score_line = re.compile(r"^\s*(\S.*?)\s+(\d+|—|-)\s*$")

    levels: List[Dict[str, float]] = []
    current: Dict[str, float] = {}

    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if level_hdr.match(ln):
                if current:
                    levels.append(current)
                current = {}
                continue
            m = score_line.match(ln)
            if m:
                model = m.group(1).strip()
                raw = m.group(2)
                current[model] = None if raw in ("—", "-") else float(raw)
        if current:
            levels.append(current)

    return levels


def list_models(levels: List[Mapping[str, float]]) -> List[str]:
    seen = set()
    for lvl in levels:
        seen.update(lvl.keys())
    return sorted(seen)


def compute_metrics(
    levels: List[Mapping[str, float]],
    area_penalty: float = PENALTY_MISS_AREA,
    mean_penalty: float = PENALTY_MISS_MEAN,
) -> Dict[str, dict]:
    """
    Calculates Coverage, penalised Mean SES, Best-level wins, Penalty area.
    """
    n_levels = len(levels)

    # Pre-compute the numeric minimum SES
    minima = []
    for lvl in levels:
        numeric = [v for v in lvl.values() if v is not None]
        minima.append(min(numeric) if numeric else None)

    stats: Dict[str, dict] = {}
    for model in list_models(levels):
        numeric_vals, wins, missing = [], 0, 0
        penalised_vals = []
        for i, lvl in enumerate(levels):
            v = lvl.get(model)
            if v is None:
                missing += 1
                penalised_vals.append(mean_penalty)
            else:
                numeric_vals.append(v)
                penalised_vals.append(v)
                if math.isclose(v, minima[i], rel_tol=1e-9):
                    wins += 1

        coverage = (n_levels - missing) / n_levels
        mean_ses = statistics.mean(penalised_vals)
        penalty_area = sum(numeric_vals) + missing * area_penalty

        stats[model] = dict(
            coverage=coverage,
            mean_ses=mean_ses,
            best_wins=wins,
            penalty_area=penalty_area,
        )
    return stats


def _scene_table(scene: str, stats: Dict[str, dict]) -> str:
    hdr = (
        "Model",
        "Coverage",
        "Mean SES (pen.)",
        "Best-level wins",
        "Penalty-weighted area",
    )
    rows = [
        (
            m,
            f"{d['coverage']*100:>3.0f} %",
            f"{d['mean_ses']:.1f}",
            str(d["best_wins"]),
            f"{d['penalty_area']:.0f}",
        )
        for m, d in sorted(
            stats.items(), key=lambda kv: (-kv[1]["coverage"], kv[1]["penalty_area"])
        )
    ]

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(hdr)]
    sep = " | "
    rule = "-+-".join("-" * w for w in widths)

    out = [
        f"\n### {scene}",
        sep.join(h.ljust(widths[i]) for i, h in enumerate(hdr)),
        rule,
    ]
    out += [sep.join(r[i].ljust(widths[i]) for i in range(len(hdr))) for r in rows]
    return "\n".join(out)


def _combine_penalty(scenes: List[Dict[str, dict]]) -> Dict[str, float]:
    total = {}
    for scene in scenes:
        for m, d in scene.items():
            total[m] = total.get(m, 0) + d["penalty_area"]
    return total


def main() -> None:
    files: List[str] = []

    for p in INPUT_PATHS:
        if os.path.isdir(p):
            for f in os.listdir(p):
                if f.endswith("_ses.txt"):
                    files.append(os.path.join(p, f))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"Warning: '{p}' not found.", file=sys.stderr)

    files = sorted(dict.fromkeys(files))

    if not files:
        print("No input files found from INPUT_PATHS.", file=sys.stderr)
        sys.exit(1)

    scene_metrics = []
    for p in files:
        scene_name = os.path.splitext(os.path.basename(p))[0]
        levels = parse_ses_file(p)
        metrics = compute_metrics(levels)
        scene_metrics.append(metrics)
        print(_scene_table(scene_name, metrics))

    if len(scene_metrics) > 1:
        combined = _combine_penalty(scene_metrics)
        print("\n### Cross-scene ranking (by total penalty-weighted area)")
        for rank, (model, score) in enumerate(
            sorted(combined.items(), key=lambda kv: kv[1]), 1
        ):
            print(f"{rank:>2}. {model:<20} {score:.0f}")


if __name__ == "__main__":
    main()
