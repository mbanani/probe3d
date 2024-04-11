# Copyright (c) Meta Platforms, Inc. and affiliates.
# Original code is licensed under CC BY-NC 4.0.
# Code adapted by Mohamed El Banani
import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
from torch.nn import functional as nn_F
from torch.nn.functional import cosine_similarity

res = faiss.StandardGpuResources()  # use a single GPU


def faiss_knn(query, target, k):
    # make sure query and target are contiguous
    query = query.contiguous()
    target = target.contiguous()

    num_elements, feat_dim = query.shape
    gpu_index = faiss.GpuIndexFlatL2(res, feat_dim)
    gpu_index.add(target)
    dist, index = gpu_index.search(query, k)
    return dist, index


def knn_points(X_f, Y_f, K=1, metric="euclidean"):
    """
    Finds the kNN according to either euclidean distance or cosine distance. This is
    tricky since PyTorch3D's fast kNN kernel does euclidean distance, however, we can
    take advantage of the relation between euclidean distance and cosine distance for
    points sampled on an n-dimension sphere.

    Using the quadratic expansion, we find that finding the kNN between two normalized
    is the same regardless of whether the metric is euclidean distance or cosine
    similiarity.

        -2 * xTy = (x - y)^2 - x^2 - y^2
        -2 * xtY = (x - y)^2 - 1 - 1
        - xTy = 0.5 * (x - y)^2 - 1

    Hence, the metric that would maximize cosine similarity is the same as that which
    would minimize the euclidean distance between the points, with the distances being
    a simple linear transformation.
    """
    assert metric in ["cosine", "euclidean"]
    if metric == "cosine":
        X_f = torch.nn.functional.normalize(X_f, dim=-1)
        Y_f = torch.nn.functional.normalize(Y_f, dim=-1)

    _, X_nn = faiss_knn(X_f, Y_f, K)

    # n_points x k x F
    X_f_nn = Y_f[X_nn]

    if metric == "euclidean":
        dists = (X_f_nn - X_f[:, None, :]).norm(p=2, dim=3)
    elif metric == "cosine":
        dists = 1 - cosine_similarity(X_f_nn, X_f[:, None, :], dim=-1)

    return dists, X_nn


def get_correspondences_ratio_test(
    P1_F, P2_F, num_corres, metric="cosine", bidirectional=False, ratio_test=True
):
    # Calculate kNN for k=2; both outputs are (N, P, K)
    # idx_1 returns the indices of the nearest neighbor in P2
    # output is cosine distance (0, 2)
    K = 2

    dists_1, idx_1 = knn_points(P1_F, P2_F, K, metric)
    idx_1 = idx_1[..., 0]
    if ratio_test:
        weights_1 = calculate_ratio_test(dists_1)
    else:
        weights_1 = dists_1[:, 0]

    # Take the nearest neighbor for the indices for k={1, 2}
    if bidirectional:
        dists_2, idx_2 = knn_points(P2_F, P1_F, K, metric)
        idx_2 = idx_2[..., 0]
        if ratio_test:
            weights_2 = calculate_ratio_test(dists_2)
        else:
            weights_2 = dists_2[:, 0]

        # Get topK matches in both directions
        m12_idx1, m12_idx2, m12_dist = get_topk_matches(
            weights_1, idx_1, num_corres // 2
        )
        m21_idx2, m21_idx1, m21_dist = get_topk_matches(
            weights_2, idx_2, num_corres // 2
        )

        # concatenate into correspondences and weights
        all_idx1 = torch.cat((m12_idx1, m21_idx1), dim=1)
        all_idx2 = torch.cat((m12_idx2, m21_idx2), dim=1)
        all_dist = torch.cat((m12_dist, m21_dist), dim=1)
    else:
        all_idx1, all_idx2, all_dist = get_topk_matches(weights_1, idx_1, num_corres)

    return all_idx1, all_idx2, all_dist


@torch.jit.script
def calculate_ratio_test(dists: torch.Tensor):
    """
    Calculate weights for matches based on the ratio between kNN distances.

    Input:
        (N, P, 2) Cosine Distance between point and nearest 2 neighbors
    Output:
        (N, P, 1) Weight based on ratio; higher is more unique match
    """
    # Ratio -- close to 0 is completely unique; 1 is same feature
    # Weight -- Convert so that higher is more unique
    # clamping because some dists will be 0 (when not in the pointcloud
    dists = dists.clamp(min=1e-9)
    ratio = dists[..., 0] / dists[..., 1].clamp(min=1e-9)
    weight = 1 - ratio
    return weight


# @torch.jit.script
def get_topk_matches(dists, idx, num_corres: int):
    num_corres = min(num_corres, dists.shape[-1])
    dist, idx_source = torch.topk(dists, k=num_corres, dim=-1)
    idx_target = idx[idx_source]
    return idx_source, idx_target, dist


def get_grid(H: int, W: int):
    # Generate a grid that's equally spaced based on image & embed size
    grid_x = torch.linspace(0.5, W - 0.5, W)
    grid_y = torch.linspace(0.5, H - 0.5, H)

    xs = grid_x.view(1, W).repeat(H, 1)
    ys = grid_y.view(H, 1).repeat(1, W)
    zs = torch.ones_like(xs)

    # Camera coordinate frame is +xyz (right, down, into-camera)
    # Dims: 3 x H x W
    grid_xyz = torch.stack((xs, ys, zs), dim=0)
    return grid_xyz


def grid_to_pointcloud(K_inv, depth, grid=None):
    _, H, W = depth.shape

    if grid is None:
        grid = get_grid(H, W)

    # Apply inverse projection
    points = depth * grid

    # Invert intriniscs
    points = points.view(3, H * W)
    points = K_inv @ points
    points = points.permute(1, 0)

    return points


def sample_pointcloud_features(feats, K, pc, image_shape):
    H, W = image_shape
    uvd = pc @ K.transpose(-1, -2)
    uv = uvd[:, :2] / uvd[:, 2:3].clamp(min=1e-9)

    uv[:, 0] = (2 * uv[:, 0] / W) - 1
    uv[:, 1] = (2 * uv[:, 1] / H) - 1

    # sample points
    pc_F = nn_F.grid_sample(feats[None], uv[None, None], align_corners=False)
    pc_F = pc_F[:, :, 0].transpose(1, 2)[0]

    return pc_F


def argmax_2d(x, max_value=True):
    h, w = x.shape[-2:]
    x = torch.flatten(x, start_dim=-2)
    if max_value:
        flat_indices = x.argmax(dim=-1)
    else:
        flat_indices = x.argmin(dim=-1)

    min_row = flat_indices // w
    min_col = flat_indices % w
    xy_indices = torch.stack((min_col, min_row), dim=-1)
    return xy_indices


def project_3dto2d(xyz, K_mat):
    uvd = xyz @ K_mat.transpose(-1, -2)
    uv = uvd[:, :2] / uvd[:, 2:3].clamp(min=1e-9)
    return uv


def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return aucs


def estimate_correspondence_depth(feat_0, feat_1, depth_0, depth_1, K, num_corr=500):
    xyz_0 = grid_to_pointcloud(K.inverse(), depth_0)
    xyz_1 = grid_to_pointcloud(K.inverse(), depth_1)
    xyz_0 = xyz_0[xyz_0[:, 2] > 0]
    xyz_1 = xyz_1[xyz_1[:, 2] > 0]

    feat_0 = sample_pointcloud_features(feat_0, K.clone(), xyz_0, depth_0.shape[-2:])
    feat_1 = sample_pointcloud_features(feat_1, K.clone(), xyz_1, depth_1.shape[-2:])

    idx0, idx1, corr_dist = get_correspondences_ratio_test(feat_0, feat_1, num_corr)

    corr_xyz0 = xyz_0[idx0]
    corr_xyz1 = xyz_1[idx1]

    return corr_xyz0, corr_xyz1, corr_dist


def estimate_correspondence_xyz(
    feat_0, feat_1, xyz_grid_0, xyz_grid_1, num_corr=500, ratio_test=True
):
    # upsample feats
    _, h, w = xyz_grid_0.shape
    feat_0 = nn_F.interpolate(feat_0[None], size=(h, w), mode="bicubic")[0]
    feat_1 = nn_F.interpolate(feat_1[None], size=(h, w), mode="bicubic")[0]

    uvd_0 = get_grid(h, w).to(xyz_grid_0)
    uvd_1 = get_grid(h, w).to(xyz_grid_1)

    # only keep values with real points
    feat_0 = feat_0.permute(1, 2, 0)[xyz_grid_0[2] > 0]
    feat_1 = feat_1.permute(1, 2, 0)[xyz_grid_1[2] > 0]
    xyz_0 = xyz_grid_0.permute(1, 2, 0)[xyz_grid_0[2] > 0]
    xyz_1 = xyz_grid_1.permute(1, 2, 0)[xyz_grid_1[2] > 0]
    uvd_0 = uvd_0.permute(1, 2, 0)[xyz_grid_0[2] > 0]
    uvd_1 = uvd_1.permute(1, 2, 0)[xyz_grid_1[2] > 0]

    idx0, idx1, c_dist = get_correspondences_ratio_test(
        feat_0, feat_1, num_corr, ratio_test=ratio_test
    )

    c_xyz0 = xyz_0[idx0]
    c_xyz1 = xyz_1[idx1]
    c_uv0 = uvd_0[idx0][:, :2]
    c_uv1 = uvd_1[idx1][:, :2]

    return c_xyz0, c_xyz1, c_dist, c_uv0, c_uv1


def compute_binned_performance(y, x, x_bins):
    """
    Given two arrays: (x, y), compute the mean y value for specific x_bins
    """
    y_binned = []
    for i in range(len(x_bins) - 1):
        x_min = x_bins[i]
        x_max = x_bins[i + 1]
        x_mask = (x >= x_min) * (x < x_max)
        y_binned.append(y[x_mask].mean())

    return y_binned
