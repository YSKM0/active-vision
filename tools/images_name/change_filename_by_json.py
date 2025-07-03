# import os
# import json
# import shutil

# def rename_images_to_match_transforms(transform_path, image_dir):
#     """
#     Rename actual image files in image_dir to match filenames in transforms.json.

#     Args:
#         transform_path (str): Path to transforms.json file.
#         image_dir (str): Path to the directory containing the image files.
#     """
#     with open(transform_path, 'r') as f:
#         data = json.load(f)

#     renamed_files = []

#     for frame in data['frames']:
#         expected_name = os.path.basename(frame['file_path'])  # target correct name

#         # Remove ONLY 'train_', 'val_', or 'eval_' prefix, nothing else
#         if expected_name.startswith("train_"):
#             base_name = expected_name[len("train_"):]
#         elif expected_name.startswith("val_"):
#             base_name = expected_name[len("val_"):]
#         elif expected_name.startswith("eval_"):
#             base_name = expected_name[len("eval_"):]
#         else:
#             base_name = expected_name

#         possible_names = [
#             expected_name,                     # already correct
#             "train_" + base_name,
#             "val_" + base_name,                # prefixed train version
#             "eval_" + base_name,                # prefixed eval version
#             base_name                           # pure original version
#         ]

#         src_path = None
#         for name in possible_names:
#             candidate = os.path.join(image_dir, name)
#             if os.path.exists(candidate):
#                 src_path = candidate
#                 break

#         dst_path = os.path.join(image_dir, expected_name)

#         if src_path is None:
#             print(f"[WARNING] No file found for {expected_name} (tried {possible_names})")
#             continue

#         if src_path == dst_path:
#             continue  # Already correct

#         if os.path.exists(dst_path):
#             print(f"[WARNING] Destination already exists: {dst_path}, skipping")
#             continue

#         shutil.move(src_path, dst_path)
#         renamed_files.append((src_path, dst_path))
#         print(f"Renamed: {src_path} → {dst_path}")

#     print(f"\n✅ Completed renaming {len(renamed_files)} files.")



# # "/local/home/hanwliu/lab_record/transforms_debug.json"
# transform_path = "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/RS/1/5/transforms.json"
# image_dir = "/local/home/hanwliu/lab_record/nerfstudio/images"
# rename_images_to_match_transforms(transform_path, image_dir)

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
    with open(transform_path, 'r') as f:
        data = json.load(f)

    renamed_files = []

    for frame in data['frames']:
        expected_name = os.path.basename(frame['file_path'])

        # Extract base name by stripping known prefixes
        base_name = expected_name
        for prefix in ['train_', 'val_', 'eval_']:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
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
            print(f"[WARNING] No matching file for {expected_name} (tried: {possible_names})")
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


# Update images name 

# For lab record
transform_path = "/local/home/hanwliu/table/dataset/train/nerfdirector/dinov2_large_fullres/1/120/transforms.json"
image_dir = "/local/home/hanwliu/table/nerfstudio/images_4" 

# # For tnt
# transform_path = "/local/home/hanwliu/tnt/M60/dataset/nerfdirector/FVS_dinov2_large_fullres/1/10/transforms.json"
# # "/local/home/hanwliu/tnt/M60/dataset/transforms_original.json"
# image_dir = "/local/home/hanwliu/tnt/M60/dataset/images"

rename_images_to_match_transforms(transform_path, image_dir)

# FVS_clip_ViTL14 dinov2_large_fullres

