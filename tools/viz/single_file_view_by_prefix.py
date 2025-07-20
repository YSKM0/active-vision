"""
Colour-code a single transforms.json by filename prefix.
"""
from __future__ import annotations
import collections, pathlib, viser
from utils import load_transforms, get_prefix, add_cameras

PATH = "/local/home/hanwliu/table/dataset/train2/nerfdirector/fvs_test/1/150/transforms.json"
PREFIX_COLOURS = {"train": (255, 0, 0), "val": (0, 0, 255), "eval": (0, 255, 0)}
SHOW_FRUSTUM = True
FALLBACK_GREY = (160, 160, 160)
SHOW_UNMATCHED = False


def run(path: str | pathlib.Path):
    frames = load_transforms(path).get("frames", [])
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for f in frames:
        grouped[get_prefix(f.get("file_path", ""))].append(f)

    srv = viser.ViserServer()
    srv.scene.world_axes.visible = False

    for pre, frs in grouped.items():
        if pre not in PREFIX_COLOURS and not SHOW_UNMATCHED:
            continue
        colour = PREFIX_COLOURS.get(pre, FALLBACK_GREY)
        f_objs, fr_objs = add_cameras(srv, frs, colour, pre, show_frustum=SHOW_FRUSTUM)
        for o in (*f_objs, *fr_objs):
            o.visible = True

    srv.sleep_forever()


if __name__ == "__main__":
    run(PATH)
