"""
Dynamic prefix viewer
─────────────────────
• Load multiple transforms.json files.
• Colour every camera by filename prefix ("train", "val", "eval", …).
• Optional grey “default” for files without a prefix.
• Optional animation: one file visible at a time (INTERVAL_S).

Requires utils.py (load_transforms, get_prefix, add_cameras)
"""
from __future__ import annotations
import collections, pathlib, threading, time, viser
from utils import load_transforms, get_prefix, add_cameras

# ────────── CONFIG ────────── #
TRANSFORMS = [
    "/local/home/hanwliu/table/dataset/train/nerfdirector/FVS/1/5/transforms.json",
    "/local/home/hanwliu/table/dataset/train/nerfdirector/FVS/1/30/transforms.json",
    "/local/home/hanwliu/table/dataset/train/nerfdirector/FVS/1/80/transforms.json",
]
PREFIX_COLOURS = {"train": (255, 0, 0), "val": (0, 0, 255), "eval": (0, 255, 0)}
INCLUDE_DEFAULT = True
SHOW_FRUSTUM = True
ENABLE_ANIMATION = True
INTERVAL_S = 2.0
GREY = (160, 160, 160)
# ──────────────────────────── #


def build_scene(paths: list[str | pathlib.Path]):
    """Return (ViserServer, per-file object lists)."""
    srv = viser.ViserServer()
    srv.scene.world_axes.visible = False
    per_file: list[tuple[list, list]] = []

    for path in paths:
        frames = load_transforms(path).get("frames", [])
        if not frames:
            print(f"⚠ {path} contains 0 frames – skipped")
            per_file.append(([], []))
            continue

        grouped: dict[str, list[dict]] = collections.defaultdict(list)
        for f in frames:
            pre = get_prefix(f.get("file_path", ""))
            if pre in PREFIX_COLOURS or INCLUDE_DEFAULT:
                grouped[pre].append(f)

        file_frames, file_frusta = [], []
        for pre, frs in grouped.items():
            colour = PREFIX_COLOURS.get(pre, GREY)
            tag = f"{pathlib.Path(path).stem}_{pre or 'default'}"
            f_objs, fr_objs = add_cameras(
                srv, frs, colour, tag, show_frustum=SHOW_FRUSTUM
            )
            # initially hidden if we animate
            for o in (*f_objs, *fr_objs):
                o.visible = not ENABLE_ANIMATION
            file_frames.extend(f_objs)
            file_frusta.extend(fr_objs)

        print(
            f"{pathlib.Path(path).name}: kept {sum(len(v) for v in grouped.values())} frames"
        )
        per_file.append((file_frames, file_frusta))

    return srv, per_file


def start_animation(per_file_objects: list[tuple[list, list]], interval: float):
    def cycle():
        idx = 0
        while True:
            for fset, rset in per_file_objects:
                for o in (*fset, *rset):
                    o.visible = False
            for o in (*per_file_objects[idx][0], *per_file_objects[idx][1]):
                o.visible = True
            idx = (idx + 1) % len(per_file_objects)
            time.sleep(interval)

    threading.Thread(target=cycle, daemon=True).start()


if __name__ == "__main__":
    server, objs = build_scene(TRANSFORMS)
    if ENABLE_ANIMATION:
        start_animation(objs, INTERVAL_S)
    else:
        for fset, rset in objs:
            for o in (*fset, *rset):
                o.visible = True
    server.sleep_forever()
