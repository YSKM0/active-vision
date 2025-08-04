import json
import os
import numpy as np


def get_camera_centers(transform_json_path):
    with open(transform_json_path, "r") as f:
        data = json.load(f)

    camera_centers = []
    for frame in data["frames"]:
        camera_to_world = np.array(frame["transform_matrix"], dtype=np.float32)
        camera_center = camera_to_world[:3, 3]

        camera_centers += [camera_center[None].copy()]
    camera_centers = np.array(camera_centers)

    print(
        "Load {} cameras from {}".format(camera_centers.shape[0], transform_json_path)
    )

    return np.squeeze(camera_centers)


def get_camera_centers2(transform_json_path):
    with open(transform_json_path, "r") as f:
        data = json.load(f)

    camera_centers = []

    for frame in data["frames"]:
        filename = os.path.basename(frame["file_path"])

        # Only select frames starting with "train_"
        if not filename.startswith("train_"):
            continue  # Skip frames without prefix

        camera_to_world = np.array(frame["transform_matrix"], dtype=np.float32)
        camera_center = camera_to_world[:3, 3]

        camera_centers += [camera_center[None].copy()]

    camera_centers = np.array(camera_centers)

    print(
        "Load {} cameras from {}".format(camera_centers.shape[0], transform_json_path)
    )

    return np.squeeze(camera_centers)


def generate_new_transform(
    base_transform, target_transform, all_train_transform, new_frame_idxs=[]
):
    # set the relative path of transforms.json to images
    with open(all_train_transform, "r") as f:
        pools = json.load(f)
        all_frames = pools["frames"]
    src_img = os.path.join(
        os.path.dirname(all_train_transform), all_frames[0]["file_path"]
    )
    rel_path = os.path.relpath(src_img, os.path.dirname(target_transform))
    rel_path = os.path.dirname(rel_path)

    # load base_transform information if base transform exists
    # otherwise, just load the head info of all transform
    if os.path.exists(base_transform):
        with open(base_transform, "r") as f:
            data = json.load(f)
    else:
        data = pools.copy()
        data["frames"] = []

    if len(new_frame_idxs) > 0:
        # correct relative path and add new the newly selected views
        for i in new_frame_idxs:
            f = all_frames[i]
            filename = os.path.basename(f["file_path"])
            f["file_path"] = os.path.join(rel_path, filename)
            data["frames"].append(f)
    else:
        # set correct rel_path to base_transform frames
        for f in data["frames"]:
            filename = os.path.basename(f["file_path"])
            f["file_path"] = os.path.join(rel_path, filename)

    with open(target_transform, "w") as f:
        json.dump(data, f, indent=4)


def generate_new_transform2(
    target_transform,
    all_transform,
    train_frame_idxs=np.array([], dtype=int),
    val_frame_idxs=np.array([], dtype=int),
    test_frame_idxs=np.array([], dtype=int),
    base_image_dir=None,
    use_relative_path=False,
):
    import json, os

    with open(all_transform, "r") as f:
        all_data = json.load(f)

    all_frames = all_data["frames"]

    # make fast membership tests
    train_set = set(int(i) for i in train_frame_idxs)
    val_set = set(int(i) for i in val_frame_idxs)
    test_set = set(int(i) for i in test_frame_idxs)

    # if you're using relative paths, force it here
    if use_relative_path:
        rel_path = "/images"

    out_frames = []
    for i, frame in enumerate(all_frames):
        fname = os.path.basename(frame["file_path"])

        # 1) held-out validation
        if i in val_set:
            prefix = "val"
        # 2) held-out test
        elif i in test_set:
            prefix = "eval"
        # 3) everything else in train
        elif i in train_set:
            prefix = "train"
        # 4) frames you never asked for
        else:
            prefix = None

        if prefix:
            new_name = f"{prefix}_{fname}"
        else:
            new_name = fname

        # rewrite the JSON path
        if base_image_dir and not use_relative_path:
            frame["file_path"] = os.path.join(base_image_dir, new_name)
        else:
            frame["file_path"] = os.path.join(rel_path, new_name)

        out_frames.append(frame)

    all_data["frames"] = out_frames
    with open(target_transform, "w") as f:
        json.dump(all_data, f, indent=4)
