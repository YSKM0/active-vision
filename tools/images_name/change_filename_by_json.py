import os
import json
import shutil


def rename_images_to_match_transforms(transform_path, image_dir):
    """
    Rename image files in image_dir to match filenames in transforms.json.

    Args:
        transform_path (str): Path to transforms.json file.
        image_dir (str): Path to the directory containing the image files.
    """
    with open(transform_path, "r") as f:
        data = json.load(f)

    renamed_files = []

    for frame in data["frames"]:
        expected_name = os.path.basename(frame["file_path"])

        # Extract base name by stripping known prefixes
        base_name = expected_name
        for prefix in ["train_", "val_", "eval_"]:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix) :]
                break

        # Construct all candidate existing filenames
        possible_names = [
            base_name,
            f"train_{base_name}",
            f"val_{base_name}",
            f"eval_{base_name}",
        ]

        src_path = None
        for name in possible_names:
            candidate = os.path.join(image_dir, name)
            if os.path.exists(candidate):
                src_path = candidate
                break

        dst_path = os.path.join(image_dir, expected_name)

        if src_path is None:
            print(
                f"[WARNING] No matching file for {expected_name} (tried: {possible_names})"
            )
            continue

        if src_path == dst_path:
            continue  # Already correctly named

        if os.path.exists(dst_path):
            print(f"[WARNING] Destination file exists: {dst_path}, skipping")
            continue

        shutil.move(src_path, dst_path)
        renamed_files.append((src_path, dst_path))
        print(f"Renamed: {src_path} → {dst_path}")

    print(f"\n✅ Completed renaming {len(renamed_files)} files.")


transform_path = "/local/home/hanwliu/lab_record/nerfstudio/transforms.json"
image_dir = "/local/home/hanwliu/lab_record/nerfstudio/images_4"

rename_images_to_match_transforms(transform_path, image_dir)
