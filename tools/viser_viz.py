# This is for visualizing the camera position   (static, the moving one is below)


import json
import numpy as np
import viser
import viser.transforms as tf

transforms_paths = [
    "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/RS/1/15/transforms.json",
    "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/FVS/1/15/transforms.json",
    "/local/home/hanwliu/lab_record/dataset/transforms2.json",
]


# Use RGB tuples for colors (Red for first set, Blue for second set)
colors = [
    (255, 0, 0),       # Red
    (0, 0, 255),       # Blue
    (0, 255, 0),       # Green
    (255, 165, 0),     # Orange
    (128, 0, 128),     # Purple
    (0, 255, 255),     # Cyan
    (255, 255, 0),     # Yellow
    (255, 192, 203),   # Pink
    (165, 42, 42),     # Brown
    (0, 128, 128),     # Teal
    (0, 100, 0),       # Dark Green
    (75, 0, 130),      # Indigo
]


# Toggle to show/hide frustums
show_frustum = True  # Set to True to show frustums, False to hide them

# Initialize viser server
server = viser.ViserServer()
server.scene.world_axes.visible = False  # Make world axes visible

for idx, path in enumerate(transforms_paths):
    with open(path, "r") as f:
        data = json.load(f)

    # Extract focal lengths and image dimensions
    if "frames" in data and len(data["frames"]) > 0:
        first_frame = data["frames"][0]
        w = first_frame.get("w", 800)
        h = first_frame.get("h", 600)
        fl_x = first_frame.get("fl_x", 500)
        fl_y = first_frame.get("fl_y", 500)
    else:
        w, h, fl_x, fl_y = 800, 600, 500, 500  # Safe defaults

    # Compute aspect ratio and field of view
    aspect_ratio = w / h
    fov_y = 2 * np.arctan(h / (2 * fl_y))  # Vertical FOV in radians

    for i, frame in enumerate(data.get("frames", [])):
        transform_matrix = np.array(frame.get("transform_matrix"))
        position = transform_matrix[:3, 3]  # Extract camera position
        rotation = transform_matrix[:3, :3]  # Extract rotation matrix
        wxyz = tf.SO3.from_matrix(rotation).wxyz  # Convert to quaternion

        # Add coordinate frame to represent camera position and orientation
        server.scene.add_frame(
            name=f"camera_frame_{idx+1}_{i+1}",
            wxyz=wxyz,
            position=position,
            axes_length=0.1,  # Length of the axes
            axes_radius=0.005,  # Radius of the axes
            origin_color=colors[idx],  # Color based on dataset
            visible=False
        )

        # Conditionally add frustums based on the toggle
        if show_frustum:
            server.scene.add_camera_frustum(
                name=f"set_{idx+1}_cam_{i+1}",
                fov=fov_y,
                aspect=aspect_ratio,
                scale=0.2,
                wxyz=wxyz,
                position=position,
                color=colors[idx],
                line_width=1
            )

server.sleep_forever()



















# import json
# import numpy as np
# import time
# import viser
# import viser.transforms as tf

# transforms_paths = [
#     "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/RS/1/5/transforms.json",
#     "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/RS/1/10/transforms.json",
#     "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/RS/1/15/transforms.json",
#     "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/RS/1/20/transforms.json",
#     "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/RS/1/25/transforms.json",
#     "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/RS/1/30/transforms.json",
#     "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/RS/1/40/transforms.json",
#     "/local/home/hanwliu/lab_record/dataset/train/nerfdirector/RS/1/50/transforms.json",
# ]

# # transforms_paths = [
# #     "/local/home/hanwliu/lab_record/nerfstudio/transforms.json",

# # ]

# colors = [
#     (255, 0, 0),       # Red
#     (0, 0, 255),       # Blue
#     (0, 255, 0),       # Green
#     (255, 165, 0),     # Orange
#     (128, 0, 128),     # Purple
#     (0, 255, 255),     # Cyan
#     (255, 255, 0),     # Yellow
#     (255, 192, 203),   # Pink
#     (165, 42, 42),     # Brown
#     (0, 128, 128),     # Teal
#     (0, 100, 0),       # Dark Green
#     (75, 0, 130),      # Indigo
# ]

# server = viser.ViserServer()
# server.scene.world_axes.visible = False

# # Store actual object references instead of names
# all_frustum_objects = []
# all_frame_objects = []

# for idx, path in enumerate(transforms_paths):
#     with open(path, "r") as f:
#         data = json.load(f)

#     if "frames" in data and len(data["frames"]) > 0:
#         first_frame = data["frames"][0]
#         w = first_frame.get("w", 800)
#         h = first_frame.get("h", 600)
#         fl_x = first_frame.get("fl_x", 500)
#         fl_y = first_frame.get("fl_y", 500)
#     else:
#         w, h, fl_x, fl_y = 800, 600, 500, 500

#     aspect_ratio = w / h
#     fov_y = 2 * np.arctan(h / (2 * fl_y))

#     frustum_objs = []
#     frame_objs = []

#     for i, frame in enumerate(data.get("frames", [])):
#         transform_matrix = np.array(frame.get("transform_matrix"))
#         position = transform_matrix[:3, 3]
#         rotation = transform_matrix[:3, :3]
#         wxyz = tf.SO3.from_matrix(rotation).wxyz

#         frame_obj = server.scene.add_frame(
#             name=f"camera_frame_{idx+1}_{i+1}",
#             wxyz=wxyz,
#             position=position,
#             axes_length=0.1,
#             axes_radius=0.005,
#             origin_color=colors[idx],
#             visible=False
#         )

#         frustum_obj = server.scene.add_camera_frustum(
#             name=f"set_{idx+1}_cam_{i+1}",
#             fov=fov_y,
#             aspect=aspect_ratio,
#             scale=0.2,
#             wxyz=wxyz,
#             position=position,
#             color=colors[idx],
#             line_width=1,
#             visible=False
#         )

#         frustum_objs.append(frustum_obj)
#         frame_objs.append(frame_obj)

#     all_frustum_objects.append(frustum_objs)
#     all_frame_objects.append(frame_objs)

# # === Animation loop ===
# def cycle_visibility():
#     current_set = 0
#     num_sets = len(all_frustum_objects)

#     while True:
#         # Hide everything
#         for frustums, frames in zip(all_frustum_objects, all_frame_objects):
#             for obj in frustums:
#                 obj.visible = False
#             for obj in frames:
#                 obj.visible = False

#         # Show current set
#         for obj in all_frustum_objects[current_set]:
#             obj.visible = True
#         for obj in all_frame_objects[current_set]:
#             obj.visible = False

#         current_set = (current_set + 1) % num_sets
#         time.sleep(2)

# # Run animation in background
# import threading
# threading.Thread(target=cycle_visibility, daemon=True).start()

# server.sleep_forever()
