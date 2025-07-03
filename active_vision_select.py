import json
import os
import argparse
import shutil
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.sparse import csgraph

from internal.colmap_utils import get_3d_correspondence_matrix
from internal.dataset_utils import *
from internal.fvs import (
    farthest_view_sampling,
    farthest_view_sampling_colmap,
    farthest_embedding_sampling,
    farthest_view_sampling_vlm,
    build_distance_matrix,
)

# ---------------------------------------------------------------------------
# global flags
# ---------------------------------------------------------------------------
np.random.seed(42)
TEST_FRAME_END = 740

VIZ = False
run_laplacian_spectrum = False
plot_laplacian_curve = False


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for NeRF Director")

    # Experiment metadata and setup
    parser.add_argument(
        "--rep",
        default=0,
        type=int,
        help="Repetition ID (used for multiple runs or seeds)",
    )
    parser.add_argument(
        "--sampling",
        default="fvs",
        choices=["rs", "fvs", "vlm", "fvs_vlm"],
        help="Sampling strategy: rs (random), fvs (farthest-view), vlm (VLM-guided), fvs_vlm (hybrid)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="",
        type=str,
        help="Directory to store training checkpoints and test outputs",
    )

    # Dataset and split configuration
    parser.add_argument(
        "--all_train_transform",
        default="",
        type=str,
        help="Path to transforms_train.json for the full training set",
    )
    parser.add_argument(
        "--test_transform",
        default="",
        type=str,
        help="Path to transforms_test.json for evaluation",
    )

    # Farthest-view (FVS) sampling options
    parser.add_argument(
        "--dist_type",
        default="euc",
        choices=["euc", "gcd"],
        help="Distance metric for FVS: euc (Euclidean), gcd (Great Circle Distance)",
    )
    parser.add_argument(
        "--vlm_dist_type",
        default="cos",
        choices=["euc", "cos"],
        help="Distance metric for VLM embeddings: cos (cosine), euc (Euclidean)",
    )
    parser.add_argument(
        "--enable_photo_dist",
        action="store_true",
        help="Enable photometric distance as an additional metric for FVS",
    )
    parser.add_argument(
        "--colmap_log",
        default="",
        type=str,
        help="Path to COLMAP images.txt file for camera pose metadata",
    )
    parser.add_argument(
        "--use_val",
        action="store_true",
        help="Split the test set into validation and evaluation subsets",
    )

    # Embedding-based sampling
    parser.add_argument(
        "--embeddings_path",
        default="",
        type=str,
        help="Path to .pkl file containing CLIP or VLM embeddings for each image",
    )

    # File management and output structure
    parser.add_argument(
        "--use_rel_path",
        action="store_true",
        help="Use relative image paths in the generated JSON output",
    )
    parser.add_argument(
        "--base_image_dir",
        default="",
        help="Base folder where the raw images are stored",
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy the selected images into the checkpoint folder",
    )
    parser.add_argument(
        "--all_transform",
        default="",
        type=str,
        help="Path to a combined transforms JSON file (train + test)",
    )

    # Hyperparameters
    parser.add_argument(
        "--alpha",
        default=0.5,
        type=float,
        help="Weighting parameter to mix distance metrics (e.g., between spatial and semantic)",
    )

    # Visualization output
    parser.add_argument(
        "--viz_dir",
        default="",
        type=str,
        help="Directory to save visualization images (e.g., view selection overlays)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def print_log(msg):
    print("************************************************************************")
    print(msg)
    print("************************************************************************\n")


def print_args(arg):
    s = "EXPERIMENT ARGUMENTS\n"
    for k, v in vars(arg).items():
        s += f"* {k}: {v}\n"
    print_log(s)


def update_available_list(current_centers, all_centers, available_mask):
    new_ids = []
    print(current_centers)
    if current_centers.ndim == 1:
        a = current_centers / np.linalg.norm(current_centers)
    else:
        a = current_centers / np.linalg.norm(current_centers, axis=1, keepdims=True)
    b = all_centers / np.linalg.norm(all_centers, axis=1, keepdims=True)
    for c in a:
        d = np.linalg.norm(b - c, axis=1)
        d[~available_mask] = np.inf
        idx = np.argmin(d)
        new_ids.append(idx)
        available_mask[idx] = False
    return new_ids


def update_available_embeddings(sel, mask):
    new_ids = []
    for i in sel:
        if mask[i]:
            new_ids.append(i)
            mask[i] = False
    return np.array(new_ids)


def split_random_subset(start, end, subset_size, seed=42):
    if seed is not None:
        np.random.seed(seed)
    pool = np.arange(start, end + 1)
    if subset_size > len(pool):
        raise ValueError("subset_size too large")
    chosen = np.random.choice(pool, size=subset_size, replace=False)
    remain = np.setdiff1d(pool, chosen)
    return chosen, remain


def copy_sampled_images(all_train_transform, indices, dst_dir, base_dir):
    os.makedirs(dst_dir, exist_ok=True)
    with open(all_train_transform) as f:
        frames = json.load(f)["frames"]
    for i in indices:
        name = os.path.basename(frames[i]["file_path"])
        src = os.path.join(base_dir, name)
        dst = os.path.join(dst_dir, name)
        if not os.path.exists(src):
            print(f"[copy_sampled_images] WARNING: {src} missing — skipped.")
            continue
        shutil.copy2(src, dst)


def copy_new_images_with_order(
    all_train_transform, new_indices, round_idx, dst_dir, base_dir
):
    """
    Copy just `new_indices`, renaming as '{round_idx}_{idx}.ext'.
    """
    os.makedirs(dst_dir, exist_ok=True)
    with open(all_train_transform) as f:
        frames = json.load(f)["frames"]
    for idx in new_indices:
        name = os.path.basename(frames[idx]["file_path"])
        ext = os.path.splitext(name)[1]
        src = os.path.join(base_dir, name)
        dst = os.path.join(dst_dir, f"{round_idx}_{idx}{ext}")
        if not os.path.exists(src):
            print(f"[copy_new_images_with_order] WARNING: {src} missing — skipped.")
            continue
        shutil.copy2(src, dst)


def distance_to_similarity(D, scheme="exp", k=None):
    """Convert a distance matrix to similarity matrix W for similarity graph"""
    D = D.copy()

    if k is not None and k < D.shape[0]:
        knn_mask = np.argsort(D, axis=1)[:, k:]
        for i, idx in enumerate(knn_mask):
            D[i, idx] = D[idx, i] = np.inf

    finite_vals = D[np.isfinite(D)]
    mu = np.median(finite_vals) if finite_vals.size > 0 else 1.0
    D /= mu

    if scheme == "exp":
        W = np.exp(-D)
    elif scheme == "inv":
        W = np.where(D < np.inf, 1.0 / (D + 1e-8), 0.0)
    else:
        raise ValueError("Unknown similarity scheme")

    np.fill_diagonal(W, 0.0)
    return W


def load_embeddings(embeddings_path: str, train_count: int):
    """Load embeddings and filter to only include training frames.

    Args:
        embeddings_path: Path to the .pkl file with embeddings.
        train_count: Number of training images (used to mask training embeddings only).

    Returns:
        Tuple of:
        - embeddings: np.ndarray of shape [N, D]
        - emb2frame: list of int frame indices
    """
    if not os.path.isfile(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    with open(embeddings_path, "rb") as f:
        raw = pickle.load(f)

    emb, emb2frame = [], []

    if isinstance(raw, dict):
        for path, vec in raw.items():
            idx = int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])
            if torch.is_tensor(vec):
                vec = vec.detach().cpu().numpy()
            emb.append(np.array(vec))
            emb2frame.append(idx)
    else:  # Assume raw is a plain ndarray or list
        arr = np.array(raw)
        for i in range(arr.shape[0]):
            emb.append(arr[i])
            emb2frame.append(i)

    emb = np.stack(emb, axis=0)

    # Filter only those that are part of the training set
    mask = [fid <= train_count for fid in emb2frame]
    emb = emb[mask]
    emb2frame = [emb2frame[i] for i, m in enumerate(mask) if m]

    return emb, emb2frame


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.viz_dir.strip() == "":
        args.viz_dir = os.path.join(args.checkpoint_dir, "viz")
    os.makedirs(args.viz_dir, exist_ok=True)

    # candidate pool (training cameras only)
    all_candidates = get_camera_centers(args.all_train_transform)
    is_available = np.ones(len(all_candidates), dtype=bool)

    # val / test split
    start, end = len(is_available) + 1, TEST_FRAME_END
    if args.use_val:
        val_sz = int((end - start + 1) * 0.5)
        val_set, test_set = split_random_subset(start, end, val_sz, seed=42)
    else:
        val_set = np.array([])
        test_set = np.arange(start, end + 1)

    # single, unified accumulator
    accumulate_indices = []
    order_counter = 1
    last_view_num = 0

    if args.sampling == "fvs" and args.enable_photo_dist:
        with open(args.all_train_transform) as f:
            names = [os.path.basename(x["file_path"]) for x in json.load(f)["frames"]]
        D = get_3d_correspondence_matrix(args.colmap_log, names)

    if args.sampling in ("vlm", "fvs_vlm"):
        train_count = len(is_available)
        embeddings, emb2frame = load_embeddings(args.embeddings_path, train_count)
        avail_emb = np.ones(len(embeddings), dtype=bool)
        is_available_vlm = avail_emb.copy()

    print_args(args)

    view_num_configs = [
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
    ]  # list(range(1, 31)), [5,30]

    for cur_view_num in view_num_configs:
        print_log(f"Running sampling={args.sampling} view_number={cur_view_num}")

        # checkpoint sub-folder
        ckpt_dir = os.path.join(args.checkpoint_dir, str(cur_view_num))
        os.makedirs(ckpt_dir, exist_ok=True)
        imgs_dir = os.path.join(ckpt_dir, "images")
        os.makedirs(imgs_dir, exist_ok=True)
        cur_train_json = os.path.join(ckpt_dir, "transforms.json")

        # holds the *fresh* indices for this round
        new_indices_this_round = np.array([], dtype=int)

        if last_view_num == 0:
            # initial random pick
            np.random.seed(args.rep)
            new_views = np.random.randint(0, len(all_candidates), size=cur_view_num)

            accumulate_indices.extend(new_views.tolist())
            if args.sampling in ("vlm", "fvs_vlm"):
                avail_emb[new_views] = False

            generate_new_transform2(
                target_transform=cur_train_json,
                all_transform=args.all_transform,
                train_frame_idxs=new_views,
                val_frame_idxs=val_set,
                test_frame_idxs=test_set,
                base_image_dir=args.base_image_dir,
                use_relative_path=args.use_rel_path,
            )

            if VIZ:
                print(new_views)
                copy_new_images_with_order(
                    args.all_train_transform,
                    new_views,
                    order_counter,
                    args.viz_dir,
                    args.base_image_dir,
                )
                order_counter += 1

            centers = get_camera_centers2(cur_train_json)
            update_available_list(centers, all_candidates, is_available)

        else:
            k = cur_view_num - last_view_num

            if args.sampling == "vlm":
                new_emb_idx = farthest_embedding_sampling(
                    k,
                    embeddings,
                    seed=args.rep,
                    dist_type=args.vlm_dist_type,
                    selected_status=(~avail_emb).tolist(),
                )
                emb_ids = update_available_embeddings(new_emb_idx, avail_emb)
                is_available[emb_ids] = False
                new_indices_this_round = emb_ids

            elif args.sampling == "fvs_vlm":
                new_emb_idx = farthest_view_sampling_vlm(
                    k,
                    all_candidates,
                    embeddings,
                    seed=args.rep,
                    alpha=args.alpha,
                    spatial_dist_type=args.dist_type,
                    vlm_dist_type=args.vlm_dist_type,
                    selected_status=(~avail_emb).tolist(),
                )
                emb_ids = update_available_embeddings(new_emb_idx, avail_emb)
                is_available[emb_ids] = False
                new_indices_this_round = emb_ids

            elif args.sampling == "rs":
                np.random.seed(args.rep * cur_view_num)
                pool = np.arange(0, len(all_candidates))[is_available]
                np.random.shuffle(pool)
                new_indices_this_round = pool[:k]
                is_available[new_indices_this_round] = False

            elif args.sampling == "fvs":
                if args.enable_photo_dist:
                    new_views = farthest_view_sampling_colmap(
                        k,
                        all_candidates,
                        args.rep,
                        D,
                        dist_type=args.dist_type,
                        selected_status=~is_available,
                    )
                else:
                    new_views = farthest_view_sampling(
                        k,
                        all_candidates,
                        args.rep,
                        dist_type=args.dist_type,
                        selected_status=~is_available,
                    )
                tmp = np.array(new_views[last_view_num:], dtype=int)
                tmp = np.unique(tmp[is_available[tmp]])[:k]
                new_indices_this_round = tmp
                is_available[tmp] = False

            if new_indices_this_round.size > 0:
                accumulate_indices.extend(new_indices_this_round.tolist())

            generate_new_transform2(
                target_transform=cur_train_json,
                all_transform=args.all_transform,
                train_frame_idxs=np.array(accumulate_indices, dtype=int),
                val_frame_idxs=val_set,
                test_frame_idxs=test_set,
                base_image_dir=args.base_image_dir,
                use_relative_path=args.use_rel_path,
            )

        # copy cumulative images
        if args.copy_images:
            copy_sampled_images(
                args.all_train_transform,
                np.array(accumulate_indices, dtype=int),
                imgs_dir,
                args.base_image_dir,
            )

        # copy new images with round-wise names
        if VIZ and new_indices_this_round.size > 0:
            print(new_indices_this_round)
            print(accumulate_indices)
            copy_new_images_with_order(
                args.all_train_transform,
                new_indices_this_round,
                order_counter,
                args.viz_dir,
                args.base_image_dir,
            )
            order_counter += 1
        last_view_num = cur_view_num

    print_log("Sampling complet")

    # Laplacian spectrum pipeline
    if run_laplacian_spectrum:
        # 1.  Select embeddings (only for VLM-based modes)
        use_embedding = args.sampling in ("vlm", "fvs_vlm")
        embs = embeddings if use_embedding else None
        e2f = emb2frame if use_embedding else None

        # 2.  Pair-wise distance matrix
        dist_mat = build_distance_matrix(
            np.array(accumulate_indices, dtype=int),
            args.sampling,
            args.dist_type,
            all_camera_centers=all_candidates,
            embeddings=embs,
            emb2frame=e2f,
            alpha=args.alpha,
            vlm_dist_type=args.vlm_dist_type,
        )
        print_log(f"Distance matrix shape: {dist_mat.shape}")

        # 3.  k-NN exponential similarity  ➜  normalised Laplacian spectrum
        k_nn = min(5, dist_mat.shape[0] - 1)
        W = distance_to_similarity(dist_mat, scheme="exp", k=k_nn)
        L = csgraph.laplacian(W, normed=True)
        eigvals = np.linalg.eigvalsh(L)
        print_log(f"First 20 eigenvalues: {np.round(eigvals[:20], 3)}")

        # Diagnostic plot
        if plot_laplacian_curve:
            reinforce = False  # get the square of eigenvalues
            eigvals_plot = eigvals**2 if reinforce else eigvals
            plot_ylabel = r"$\lambda_i^2$" if reinforce else r"$\lambda_i$"
            title_suffix = "²" if reinforce else ""

            plt.figure()
            plt.plot(eigvals_plot)
            plt.title(f"Laplacian Spectrum{title_suffix} ({args.sampling}, exp-kNN)")
            plt.xlabel("Eigenvalue index")
            plt.ylabel(plot_ylabel)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
