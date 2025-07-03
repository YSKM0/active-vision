# This file is for given an transforms.json and a image base storage path, we can copy image to our desired path

import json
import os
import shutil

# === CONFIGURATION ===
transforms_path = "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/FVS/1/60/transforms.json"  # JSON file with frames
source_image_dir = "/local/home/hanwliu/lab_record/nerfstudio/images_4"  # Where the original images are
destination_image_dir = "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/FVS/1/60/images_4"  # Where to copy them

# === HELPER FUNCTION ===
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Load the transforms.json
with open(transforms_path, "r") as f:
    data = json.load(f)

# Ensure destination folder exists
ensure_dir(destination_image_dir)

# Copy each image
for frame in data["frames"]:
    file_path = frame["file_path"]
    filename = os.path.basename(file_path)
    
    src = os.path.join(source_image_dir, filename)
    dst = os.path.join(destination_image_dir, filename)
    
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied: {filename}")
    else:
        print(f"? Missing: {filename} (was listed in JSON but not found in source)")

print(f"\n? Image copying complete. Images saved to: {destination_image_dir}")