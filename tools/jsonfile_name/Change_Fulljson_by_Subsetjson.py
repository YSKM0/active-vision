# Partial json only contains the subset of whole frames
# Full json contains whole frames
# We use partial to add prefix for full json and save in output transforms path
# image folder can be omitted, otherwise we change image names
# prefix can be change

import json
import re
from pathlib import Path
import shutil

# === CONFIG ===
partial_transforms_path = Path(
    "/local/home/hanwliu/lab_record/dataset/train0/nerfdirector/FVS/1/20/transforms.json"
)
full_transforms_path = Path(
    "/local/home/hanwliu/lab_record/dataset/train0/nerfdirector/transforms.json"
)
output_transforms_path = Path(
    "/local/home/hanwliu/lab_record/dataset/train0/nerfdirector/debug_fvs20/transforms.json"
)

image_folder = Path(
    "/local/home/hanwliu/lab_record/dataset/train0/nerfdirector/debug_fvs20/images"
)  # Can be None or empty
prefix = "train_"  # Customizable prefix

# Step 1: Load both transforms
with open(partial_transforms_path, "r") as f:
    partial_data = json.load(f)

with open(full_transforms_path, "r") as f:
    full_data = json.load(f)

# Step 2: Extract normalized frame names from partial
selected_filenames = set()
for frame in partial_data["frames"]:
    match = re.search(r"frame_(\d+)\.png", frame["file_path"])
    if match:
        frame_id = int(match.group(1))
        filename = f"frame_{frame_id:05d}.png"
        selected_filenames.add(filename)

print("\n? Selected filenames from partial transforms:")
for name in sorted(selected_filenames):
    print(" -", name)

# Step 3: Modify matching frame paths in the full transforms
modified_count = 0
for frame in full_data["frames"]:
    match = re.search(r"frame_(\d+)\.png", frame["file_path"])
    if match:
        frame_id = int(match.group(1))
        filename = f"frame_{frame_id:05d}.png"
        if filename in selected_filenames:
            new_filename = f"{prefix}{filename}"
            frame["file_path"] = frame["file_path"].replace(filename, new_filename)
            modified_count += 1

print(f"\n? Total frame paths updated in JSON: {modified_count}")

# Step 4: Rename image files on disk (if image_folder is provided and exists)
renamed_count = 0
if image_folder and image_folder.exists():
    for filename in selected_filenames:
        old_path = image_folder / filename
        new_path = image_folder / f"{prefix}{filename}"
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
            print(f"? Renamed image: {old_path.name} ? {new_path.name}")
            renamed_count += 1
        else:
            print(f"??  Image not found: {old_path}")
    print(f"\n? Total image files renamed: {renamed_count}")
else:
    print("\n?? Skipping image renaming (no valid image folder provided).")

# Step 5: Save updated full transforms
output_transforms_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_transforms_path, "w") as f:
    json.dump(full_data, f, indent=4)

print(f"\n? Saved updated transforms to: {output_transforms_path}")
