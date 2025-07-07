# This file is for changing the path name of transforms.json file, so that nerfstudio can recongnize the train and test set
# This is applied on the test set's transforms.json

import json
import os
import re
from pathlib import Path
import shutil

# === CONFIG ===
image_folder = Path("/local/home/hanwliu/table/nerfstudio/images")
transforms_path = Path("/local/home/hanwliu/table/nerfstudio/transforms.json")
output_transforms_path = Path("/local/home/hanwliu/table/dataset/transforms.json")

# Split ranges (inclusive)
train_ranges = [(1, 555)]  # [(1, 27), (46, 83)]
eval_ranges = [(556, 740)]


def get_split(frame_id):
    for start, end in train_ranges:
        if start <= frame_id <= end:
            return "train"
    for start, end in eval_ranges:
        if start <= frame_id <= end:
            return "eval"
    return None  # Ignore others


# Load original transforms
with open(transforms_path, "r") as f:
    data = json.load(f)

# Process each frame
for frame in data["frames"]:
    match = re.search(r"frame_(\d+)\.png", frame["file_path"])
    if not match:
        continue
    frame_id = int(match.group(1))
    split = get_split(frame_id)

    if split:
        old_name = f"frame_{frame_id:05d}.png"
        new_name = f"{split}_frame_{frame_id:05d}.png"

        # Update JSON path
        frame["file_path"] = f"images/{new_name}"

        # Rename file on disk
        src = image_folder / old_name
        dst = image_folder / new_name
        if src.exists():
            shutil.move(str(src), str(dst))
        else:
            print(f"?? Warning: {src} not found.")
    else:
        print(f"Skipping frame_{frame_id:05d}.png ? not in split.")

# Save updated transforms.json
with open(output_transforms_path, "w") as f:
    json.dump(data, f, indent=4)

print("? Done! Images renamed and transforms_updated.json written.")
