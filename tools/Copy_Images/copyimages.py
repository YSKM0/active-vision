import os
import json
import numpy as np

# Paths (update if needed)
colmap_txt_path = "/local/home/hanwliu/cafe/colmap/sparse/0/cameras.txt"
images_txt_path = "/local/home/hanwliu/cafe/colmap/sparse/0/images.txt"
output_json = "/local/home/hanwliu/cafe/colmap/transforms.json"

# Read COLMAP images.txt file (extract camera poses)
frames = []
with open(images_txt_path, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, img_name = parts
        # Convert quaternion to rotation matrix
        q = np.array([float(qw), float(qx), float(qy), float(qz)])
        R = np.linalg.inv(
            np.array(
                [
                    [
                        1 - 2 * (q[2] ** 2 + q[3] ** 2),
                        2 * (q[1] * q[2] - q[0] * q[3]),
                        2 * (q[1] * q[3] + q[0] * q[2]),
                    ],
                    [
                        2 * (q[1] * q[2] + q[0] * q[3]),
                        1 - 2 * (q[1] ** 2 + q[3] ** 2),
                        2 * (q[2] * q[3] - q[0] * q[1]),
                    ],
                    [
                        2 * (q[1] * q[3] - q[0] * q[2]),
                        2 * (q[2] * q[3] + q[0] * q[1]),
                        1 - 2 * (q[1] ** 2 + q[2] ** 2),
                    ],
                ]
            )
        )
        t = np.array([float(tx), float(ty), float(tz)])
        transform_matrix = np.column_stack((R, t))
        transform_matrix = np.vstack((transform_matrix, [0, 0, 0, 1]))

        frames.append(
            {
                "file_path": "images/" + img_name,
                "transform_matrix": transform_matrix.tolist(),
            }
        )

# Save to transforms.json
with open(output_json, "w") as f:
    json.dump({"frames": frames}, f, indent=4)

print(f"Saved camera poses to {output_json}")
