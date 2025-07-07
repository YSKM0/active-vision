from __future__ import annotations
import json, pathlib, colorsys
from typing import List, Tuple
import numpy as np, viser, viser.transforms as tf


# ────────── I/O & helpers ────────── #
def load_transforms(path: str | pathlib.Path) -> dict:
    with open(path, "r") as fp:
        return json.load(fp)


def infer_intrinsics(frames: list[dict]) -> Tuple[float, float, float, float]:
    if frames:
        f0 = frames[0]
        return (
            f0.get("w", 800),
            f0.get("h", 600),
            f0.get("fl_x", 500),
            f0.get("fl_y", 500),
        )
    return 800, 600, 500, 500


def get_prefix(fp: str) -> str:
    return pathlib.Path(fp).name.split("_")[0] if "_" in fp else "default"


# ────────── Colour palettes ───────── #
DEFAULT_PALETTE = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 165, 0),
    (128, 0, 128),
    (0, 255, 255),
    (255, 255, 0),
    (255, 192, 203),
    (165, 42, 42),
    (0, 128, 128),
    (0, 100, 0),
    (75, 0, 130),
]


def hsv_palette(n: int) -> List[Tuple[int, int, int]]:
    palette = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 1, 1)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


# ────────── Scene construction ────── #
def add_cameras(server, frames, colour, prefix, *, show_frustum=True):
    w, h, flx, fly = infer_intrinsics(frames)
    aspect, fov = w / h, 2 * np.arctan(h / (2 * fly))
    frame_objs, frustum_objs = [], []

    for i, f in enumerate(frames):
        T = np.asarray(f["transform_matrix"], dtype=float)
        R, t = T[:3, :3], T[:3, 3]
        wxyz = tf.SO3.from_matrix(R).wxyz

        frame = server.scene.add_frame(
            name=f"{prefix}_frame_{i}",
            wxyz=wxyz,
            position=t,
            axes_length=0.1,
            axes_radius=0.005,
            origin_color=colour,
            visible=False,
        )
        frame_objs.append(frame)

        if show_frustum:
            frustum = server.scene.add_camera_frustum(
                name=f"{prefix}_frustum_{i}",
                fov=fov,
                aspect=aspect,
                scale=0.2,
                wxyz=wxyz,
                position=t,
                color=colour,
                line_width=1,
                visible=False,
            )
            frustum_objs.append(frustum)

    return frame_objs, frustum_objs
