# This file can seperate one transforms.json into 2 different transforms.json file
# Using this for evaluation it wont be accurate
# But we can use it as the camera position visualization

import json
import os

# === CONFIGURATION ===
input_path = "/local/home/hanwliu/table/nerfstudio/transforms.json"

# Define output files and their corresponding frame ranges (inclusive)
splits = [
    {
        "output_path": "/local/home/hanwliu/table/dataset/transforms_train.json",
        "start_frame": 1,
        "end_frame": 555,
    },
    {
        "output_path": "/local/home/hanwliu/table/dataset/transforms_test.json",
        "start_frame": 556,
        "end_frame": 740,
    },
]


# === HELPER FUNCTION ===
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def get_frame_number(frame):
    file_path = frame["file_path"]
    return int(file_path.split("_")[-1].split(".")[0])


# === MAIN PROCESSING ===

# Load the original transforms.json
with open(input_path, "r") as f:
    data = json.load(f)

all_frames = data["frames"]

shared_metadata = {k: v for k, v in data.items() if k != "frames"}

# For each defined split, extract the frames and save a new transforms.json
for split in splits:
    output_path = split["output_path"]
    start_frame = split["start_frame"]
    end_frame = split["end_frame"]

    # Filter frames in the defined range
    selected_frames = [
        frame
        for frame in all_frames
        if start_frame <= get_frame_number(frame) <= end_frame
    ]

    # Create split JSON structure
    split_data = {
        # "camera_model": data.get("camera_model"),
        **shared_metadata,
        "frames": selected_frames,
    }

    if "orientation_override" in data:
        split_data["orientation_override"] = data["orientation_override"]

    # Ensure output directory exists
    ensure_dir(output_path)

    # Write to file
    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"? Saved {len(selected_frames)} frames to: {output_path}")
