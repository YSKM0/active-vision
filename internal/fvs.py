import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_distance_matrix(
    selected_indices,
    sampling_method,
    dist_type,
    all_camera_centers=None,
    embeddings=None,
    emb2frame=None,
    alpha=0.5,
    vlm_dist_type="cos",
):
    """
    Return an N×N matrix of pair-wise distances for the views listed in
    `selected_indices`, where N == len(selected_indices).

    Supported sampling methods:
    * 'fvs', 'rs'      → Euclidean or great-circle distance on camera centers
    * 'vlm'            → Cosine distance on embeddings
    * 'fvs_vlm'        → Linear combination of spatial + embedding distances

    Parameters
    ----------
    selected_indices : (N,) array_like[int]
        Frame indices in the order they were selected.
    sampling_method  : str
        One of {'fvs', 'rs', 'vlm', 'fvs_vlm'}.
    dist_type        : str
        'gcd' | 'euc' — spatial distance type.
    all_camera_centers : (M,3) ndarray, optional
        Cartesian camera centres for the *entire* training split.
    embeddings : (E,D) ndarray, optional
        Embedding matrix (e.g. CLIP features).
    emb2frame : list[int], optional
        Mapping: emb2frame[i] → frame index of embeddings[i].
    alpha : float, optional
        Mixing ratio for 'fvs_vlm'. 0 = only spatial, 1 = only embeddings.
    vlm_dist_type : str
        'cos' | 'euc' — distance type for VLM embeddings.

    Returns
    -------
    dist_mat : (N,N) ndarray
        Symmetric pairwise distance matrix.
    """
    selected_indices = np.asarray(selected_indices, dtype=int)
    N = selected_indices.size
    dist_mat = np.zeros((N, N), dtype=float)

    # map frame-id → embedding row (only needed if using embeddings)
    if sampling_method in {"vlm", "fvs_vlm"}:
        if embeddings is None or emb2frame is None:
            raise ValueError(
                "embeddings and emb2frame must be provided for vlm/fvs_vlm"
            )
        fid2row = {fid: r for r, fid in enumerate(emb2frame)}
        vecs = np.stack([embeddings[fid2row[idx]] for idx in selected_indices], axis=0)

    if sampling_method in {"fvs", "rs", "fvs_vlm"}:
        if all_camera_centers is None:
            raise ValueError("all_camera_centers must be provided for fvs/rs/fvs_vlm")
        pts = all_camera_centers[selected_indices]

        if dist_type == "gcd":
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            spatial_fn = great_circle_dist
        elif dist_type == "euc":
            spatial_fn = euclidean_dist
        else:
            raise ValueError(f"Unsupported spatial dist_type: {dist_type}")

        spatial_d = np.zeros((N, N))
        for i in range(N):
            spatial_d[i] = spatial_fn(pts[i], pts)

    if sampling_method == "vlm":
        if vlm_dist_type == "cos":
            sim = cosine_similarity(vecs)
            dist_mat = 1.0 - sim
        elif vlm_dist_type == "euc":
            dist_mat = np.linalg.norm(vecs[:, None] - vecs[None, :], axis=-1)
        else:
            raise ValueError(f"Unsupported vlm_dist_type: {vlm_dist_type}")

    elif sampling_method == "fvs_vlm":
        # Compute VLM-based distance
        if vlm_dist_type == "cos":
            sim = cosine_similarity(vecs)
            vlm_d = 1.0 - sim
        elif vlm_dist_type == "euc":
            vlm_d = np.linalg.norm(vecs[:, None] - vecs[None, :], axis=-1)
        else:
            raise ValueError(f"Unsupported vlm_dist_type: {vlm_dist_type}")

        # Combine both distances
        dist_mat = (1 - alpha) * spatial_d + alpha * vlm_d

    elif sampling_method in {"fvs", "rs"}:
        dist_mat = spatial_d

    else:
        raise ValueError(f"Unknown sampling_method '{sampling_method}'")

    # Numerical stability: clean diagonal, enforce symmetry
    np.fill_diagonal(dist_mat, 0.0)
    dist_mat = 0.5 * (dist_mat + dist_mat.T)

    return dist_mat


def cosine_dist(x, y):
    """
    Compute cosine distance (1 - cosine similarity) between vector x and matrix y,
    using sklearn.metrics.pairwise.cosine_similarity.
    """
    # x: array of shape (d,), y: array of shape (n,d)
    cos_sim = cosine_similarity(x.reshape(1, -1), y)[0]
    return 1 - cos_sim


def great_circle_dist(p1, p2):
    """Caluclate the great-circle distance between reference point and others

    Args:
    p1: numpy array with shape [d,], reference point
    p2: numpy array with shape [N, d,], target points
    is_spherical: bool
    """
    chord = np.sqrt(((p1 - p2) ** 2).sum(-1))

    chord = np.clip(chord, -2.0, 2.0)
    return 2 * np.arcsin(chord / 2.0)


def euclidean_dist(p1, p2):
    """Compute the Euclidean distance between reference point and others in cartessian coordinate

    Args:
        p1: numpy array with the shape [3]
        p2: numpy array with the shape [N, 3]
    """

    p1 = np.reshape(p1, (1, 3))
    dist = np.sum((p2 - p1) ** 2, axis=1)

    return dist


def farthest_view_sampling(K, candidates, seed, dist_type="euc", selected_status=[]):
    """Farthest view sampling according to the distance between camera centers

    Args:
        K: int, number of views to be selected
        candidates: list, list of all candidate camera centers
        seed: int, random seed
        dist_type: str, specify the spatial distance metric, including great circle distance and Euclidean distance
        selected_status: list, indicate if one view is selected or not
    Return:
        selected_points: list, all selected camera centers
    """
    dist_dict = {"gcd": "great_circle_dist", "euc": "euclidean_dist"}

    np.random.seed(seed)

    # randomly select N points into the waiting list
    points = np.array(candidates)
    if dist_type == "gcd":
        radius = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / radius

    # initialize dist, point_left_idx and selected_points
    N = points.shape[0]
    dist = np.full(N, np.inf)
    point_left_idx = np.arange(N)
    selected_points = []

    # initialize distance function
    dist_func = (
        dist_dict[dist_type] if dist_type in dist_dict.keys() else "great_circle_dist"
    )
    print("current dist func is {}".format(dist_func))

    # initialize dist list if selected_index provided
    # else random sample the first active point
    if len(selected_status) > 0:
        selected_points = point_left_idx[selected_status].tolist()
        point_left_idx = np.delete(point_left_idx, selected_points)
        for index in selected_points:
            p = points[index, :]
            dist_to_active_point = globals()[dist_func](p, points[point_left_idx])
            dist[point_left_idx] = np.minimum(
                dist_to_active_point, dist[point_left_idx]
            )

        selected_index = selected_points[-1]
        start = 0

    else:
        # sample first active point
        selected_index = np.random.randint(0, N - 1)
        selected_points.append(selected_index)
        point_left_idx = np.delete(point_left_idx, selected_index)
        start = 1

    for i in range(start, K):
        active_point = points[selected_index, :]

        # get the distance from points in waiting list to the active point
        dist_to_active_point = globals()[dist_func](
            active_point, points[point_left_idx]
        )

        # find the nearest neighbor in the selected list for each point in the waiting list
        dist[point_left_idx] = np.minimum(dist_to_active_point, dist[point_left_idx])

        # find the farthest nearest neighbor
        selected_index = point_left_idx[np.argmax(dist[point_left_idx])]

        selected_points.append(selected_index)
        point_left_idx = np.delete(point_left_idx, np.argmax(dist[point_left_idx]))

    return selected_points


def farthest_view_sampling_colmap(
    K, candidates, seed, D, dist_type="euc", selected_status=[]
):
    """Farthest view sampling according to both spatial and photogrammetric distance

    Args:
        K: int, number of views to be selected
        candidates: list, list of all candidate camera centers
        seed: int, random seed
        D: numpy matrix of all_views_num x all_views_num
        dist_type: str, specify the spatial distance metric, including great circle distance and Euclidean distance
        selected_status: list, indicate if one view is selected or not
    Return:
        selected_points: list, all selected camera centers
    """
    dist_dict = {"gcd": "great_circle_dist", "euc": "euclidean_dist"}

    np.random.seed(seed)

    # randomly select N points into the waiting list
    points = np.array(candidates)
    d = D
    if dist_type == "gcd":
        radius = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / radius
        d = D * np.pi  # gcd's range: [0, pi]
        print(d)
    elif dist_type == "euc":
        radius_list = np.linalg.norm(points, axis=1, keepdims=True)
        d = D * np.max(radius_list) * 2
        print(d)

    # initialize dist, point_left_idx and selected_points
    N = points.shape[0]
    dist = np.full(N, np.inf)
    point_left_idx = np.arange(N)
    selected_points = []

    # initialize distance function
    dist_func = (
        dist_dict[dist_type] if dist_type in dist_dict.keys() else "great_circle_dist"
    )
    print("current dist func is {}".format(dist_func))

    # initialize dist list if selected_index provided
    # else random sample the first active point
    if len(selected_status) > 0:
        selected_points = point_left_idx[selected_status].tolist()
        point_left_idx = np.delete(point_left_idx, selected_points)
        for index in selected_points:
            p = points[index, :]
            dist_to_active_point = globals()[dist_func](p, points[point_left_idx])

            # fetch 3d correspondence distance
            print("1.. Fetching [{}, ] from D...".format(index))
            dist_3d_to_active_point = d[index, point_left_idx]
            dist_to_active_point += dist_3d_to_active_point
            dist[point_left_idx] = np.minimum(
                dist_to_active_point, dist[point_left_idx]
            )
        selected_index = selected_points[-1]
        start = 0
    else:
        # sample first active point
        selected_index = np.random.randint(0, N - 1)
        selected_points.append(selected_index)
        point_left_idx = np.delete(point_left_idx, selected_index)
        start = 1

    for i in range(start, K):
        active_point = points[selected_index, :]

        # get the distance from points in waiting list to the active point
        dist_to_active_point = globals()[dist_func](
            active_point, points[point_left_idx]
        )

        # fetch 3d correspondence distance
        print("2.. Fetching [{}, ] from D...".format(selected_index))
        dist_3d_to_active_point = d[selected_index, point_left_idx]
        dist_to_active_point += dist_3d_to_active_point

        dist[point_left_idx] = np.minimum(dist_to_active_point, dist[point_left_idx])

        # find the neighbor satisfying: 1) the farthest nearest, and 2) the most different
        selected_index = point_left_idx[np.argmax(dist[point_left_idx])]

        selected_points.append(selected_index)
        point_left_idx = np.delete(point_left_idx, np.argmax(dist[point_left_idx]))

    return selected_points


def farthest_embedding_sampling(
    K, embeddings, seed=0, dist_type="cos", selected_status=None
):
    """
    Farthest sampling based on embedding distances instead of camera positions.

    Args:
        K: int, number of embeddings to select.
        embeddings: np.ndarray of shape (N, D) or list of embedding vectors.
        seed: int, random seed.
        dist_type: str, 'cos' for cosine distance or 'euc' for Euclidean distance.
        selected_status: list of bool of length N indicating pre-selected indices (optional).

    Returns:
        selected_indices: list of selected embedding indices.
    """
    if selected_status is None:
        selected_status = []

    embeddings = np.array(embeddings)
    N = embeddings.shape[0]
    assert K <= N, "K must be <= number of embeddings"

    dist_funcs = {"euc": euclidean_dist, "cos": cosine_dist}
    assert (
        dist_type in dist_funcs
    ), f"dist_type must be one of {list(dist_funcs.keys())}"

    np.random.seed(seed)

    # initialize distances to infinity
    dist = np.full(N, np.inf)
    all_indices = np.arange(N)

    # initialize selected list
    if selected_status:
        selected_indices = [i for i, flag in enumerate(selected_status) if flag]
    else:
        first = np.random.randint(0, N)
        selected_indices = [first]

    # update distances with initial selection
    for idx in selected_indices:
        dists = dist_funcs[dist_type](embeddings[idx], embeddings)
        dist = np.minimum(dist, dists)

    # candidate pool
    candidates = set(all_indices) - set(selected_indices)

    # iteratively select farthest
    new_emb_idx = []
    while len(new_emb_idx) < K:
        next_idx = max(candidates, key=lambda i: dist[i])
        new_emb_idx.append(next_idx)

        # update distances
        new_dists = dist_funcs[dist_type](embeddings[next_idx], embeddings)
        dist = np.minimum(dist, new_dists)

        candidates.remove(next_idx)

    return new_emb_idx


def farthest_view_sampling_vlm(
    K,
    points,  # (N,3) camera centres
    embeddings,  # (N,D) VLM embeddings
    seed=0,
    alpha=1.0,  # weight for VLM term
    spatial_dist_type="euc",  # 'euc' or 'gcd'
    vlm_dist_type="cos",  # 'cos' or 'euc'
    selected_status=None,
):
    """
    Greedy farthest-point sampling on a *combined* distance:
        d_total = d_spatial_norm + alpha * d_vlm_norm
    Both components are first scaled to [0,1].

    Returns
    -------
    selected_indices : list[int]   (length = K)
    """
    # --------------- sanity & setup -----------------
    if selected_status is None:
        selected_status = []

    points = np.asarray(points)
    embeddings = np.asarray(embeddings)
    N = points.shape[0]
    assert K <= N, "K must be ≤ number of points/embeddings"

    # map distance functions ------------------------------------------
    spatial_funcs = {"euc": euclidean_dist, "gcd": great_circle_dist}
    vlm_funcs = {"euc": euclidean_dist, "cos": cosine_dist}
    assert spatial_dist_type in spatial_funcs
    assert vlm_dist_type in vlm_funcs

    # normalisation constants ----------------------------------------
    if spatial_dist_type == "euc":
        max_spatial = 2 * np.max(np.linalg.norm(points, axis=1))  # scene diameter
    else:  # great-circle
        # points must lie on unit sphere for gcd
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        max_spatial = np.pi  # range of gcd

    if vlm_dist_type == "cos":
        max_vlm = 2.0  # cosine-dist  ∈ [0,2]
    else:  # Euclidean on embeddings – approximate by diameter
        max_vlm = 2 * np.max(np.linalg.norm(embeddings, axis=1))
    # avoid zero division
    max_spatial = max(max_spatial, 1e-9)
    max_vlm = max(max_vlm, 1e-9)

    # ---------------------------------------------------------------
    np.random.seed(seed)
    dist_total = np.full(N, np.inf)
    all_idx = np.arange(N)

    # ---- initial selection ----------------------------------------
    if selected_status:
        sel_idx = [i for i, flag in enumerate(selected_status) if flag]
    else:
        sel_idx = [np.random.randint(0, N)]

    # update distance map w.r.t. current selection ------------------
    for idx in sel_idx:
        ds = spatial_funcs[spatial_dist_type](points[idx], points) / max_spatial
        dv = vlm_funcs[vlm_dist_type](embeddings[idx], embeddings) / max_vlm
        dist_total = np.minimum(dist_total, ds + alpha * dv)

    candidates = set(all_idx) - set(sel_idx)
    new_idx = []
    while len(new_idx) < K:
        # pick the candidate with the largest "closest selected" distance
        nxt = max(candidates, key=lambda i: dist_total[i])
        new_idx.append(nxt)

        ds = spatial_funcs[spatial_dist_type](points[nxt], points) / max_spatial
        dv = vlm_funcs[vlm_dist_type](embeddings[nxt], embeddings) / max_vlm
        dist_total = np.minimum(dist_total, ds + alpha * dv)

        candidates.remove(nxt)

    return new_idx
