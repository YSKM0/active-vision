"""
Static view of multiple transforms.json files (no animation).
"""
from __future__ import annotations
import viser
from utils import load_transforms, DEFAULT_PALETTE, hsv_palette, add_cameras

TRANSFORMS = [
    "/local/home/hanwliu/table/dataset/transforms_train.json",
    "/local/home/hanwliu/table/dataset/transforms_test.json",
]
SHOW_FRUSTUM = True


def run(paths, *, show_frustum=False):
    palette = DEFAULT_PALETTE + hsv_palette(max(0, len(paths) - len(DEFAULT_PALETTE)))
    srv = viser.ViserServer()
    srv.scene.world_axes.visible = False

    for idx, p in enumerate(paths):
        frames = load_transforms(p).get("frames", [])
        f_objs, fr_objs = add_cameras(
            srv,
            frames,
            palette[idx % len(palette)],
            f"set{idx}",
            show_frustum=show_frustum,
        )
        for o in (*f_objs, *fr_objs):
            o.visible = True

    srv.sleep_forever()


if __name__ == "__main__":
    run(TRANSFORMS, show_frustum=SHOW_FRUSTUM)
