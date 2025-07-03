# This file is for renaming the images name, for nerfstudio split the whole images into train and test
# So that the new transforms.json can find the images


import os
import json
import re
import shutil
from pathlib import Path

# === CONFIGURATION ===
image_folder = Path("/local/home/hanwliu/lab_record/dataset/test0/images")
transforms_path = Path("/local/home/hanwliu/lab_record/dataset/test0/transforms.json")

# Define frame ranges for train/eval
train_ranges = [(1, 2000)]
eval_ranges = [(2001, 2221)]


# Function to determine split type based on frame ID
def get_split(frame_id):
    for start, end in train_ranges:
        if start <= frame_id <= end:
            return "train"
    for start, end in eval_ranges:
        if start <= frame_id <= end:
            return "eval"
    return None


# Load original transforms.json
with open(transforms_path, "r") as f:
    data = json.load(f)

# Update each frame
for frame in data["frames"]:
    match = re.search(r"frame_(\d+)\.png", frame["file_path"])
    if not match:
        continue

    frame_id = int(match.group(1))
    split = get_split(frame_id)

    if split is None:
        print(f"?? Skipping frame {frame_id}: not in any defined split range")
        continue

    old_filename = f"frame_{frame_id:05d}.png"
    new_filename = f"{split}_frame_{frame_id:05d}.png"

    old_path = image_folder / old_filename
    new_path = image_folder / new_filename

    # Rename the image on disk
    if old_path.exists():
        shutil.move(str(old_path), str(new_path))
        print(f"Renamed: {old_filename} ? {new_filename}")
    else:
        print(f"? File not found: {old_path}")

    # Update the path in JSON
    frame["file_path"] = f"images_4/{new_filename}"

print(f"Done!")
