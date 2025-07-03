import json
import os


def convert_to_absolute_paths(json_path, absolute_prefix, output_path=None):
    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Process each frame
    for frame in data.get("frames", []):
        relative_path = frame["file_path"]
        # Get the basename (e.g., '0_00005.png') from the relative path
        file_name = os.path.basename(relative_path)
        # Join it with the absolute prefix
        absolute_path = os.path.join(absolute_prefix, file_name)
        # Normalize and convert to Unix-style path
        frame["file_path"] = os.path.normpath(absolute_path).replace("\\", "/")

    # Decide where to save
    if output_path is None:
        output_path = json_path  # Overwrite original

    # Save the updated JSON
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


# Example usage:
convert_to_absolute_paths(
    json_path="/local/home/hanwliu/table/dataset/transforms_train.json",
    absolute_prefix="/local/home/hanwliu/table/nerfstudio/images",
)
