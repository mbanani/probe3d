"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import scipy
import torch
import torch.nn.functional as nn_F
from tqdm import tqdm

from ..models.dense_encoders import center_padding


def compute_pw_distances(source_feat, target_feat=None):
    target_feat = source_feat if target_feat is None else target_feat
    assert len(source_feat.shape) == 2
    assert len(target_feat.shape) == 2

    # compute pairwise distances
    pw_feat = (source_feat[:, None, :] - target_feat[None, :, :]).norm(p=2, dim=-1)

    return pw_feat


def compute_row_correlation(mat_a, mat_b, method="pearson"):
    assert method in ["pearson", "spearman"]
    assert mat_a.shape == mat_b.shape
    n_rows = mat_a.shape[0]

    # methods return coef and p_value, 0-th element is the correaltion coeof.
    corr_func = getattr(scipy.stats, f"{method}r")
    corr = [corr_func(mat_a[i], mat_b[i])[0] for i in range(n_rows)]
    corr = np.mean(corr)

    return corr


def compute_uppertriangle_correlation(mat_a, mat_b, method="pearson"):
    assert method in ["pearson", "spearman"]
    assert mat_a.shape == mat_b.shape

    # methods return coef and p_value, 0-th element is the correaltion coeof.
    corr_func = getattr(scipy.stats, f"{method}r")
    corr = corr_func(upper(mat_a), upper(mat_b))[0]

    return corr


def upper(matrix):
    """Returns the upper triangle of a correlation matrix.

    Args:
      matrix: numpy correlation matrix

    Returns:
      list of values from upper triangle
    """
    n, m = matrix.shape
    mask = np.triu_indices(n=n, m=m, k=1)
    return matrix[mask]


def matrix_distance(matrix_a, matrix_b, use_upper=False):
    if use_upper:
        spearman = scipy.stats.spearmanr(upper(matrix_a), upper(matrix_b))[0]
        pearson = scipy.stats.pearsonr(upper(matrix_a), upper(matrix_b))[0]
    else:
        spearman = compute_row_correlation(matrix_a, matrix_b, "spearman")
        pearson = compute_row_correlation(matrix_a, matrix_b, "pearson")

    return f"S:{spearman:.3f} P:{pearson:.3f}"


def aggregate_pairwise_matrix(
    pairwise_matrix, points_per_view, col_reduction="min", symmetrical=False
):
    """
    Aggreagtes a pairwise matrix from all-point-pairs to all-view-pairs.
    The aggregation is done for each view pair block with <col_reduction> operation
    for columns and a mean over rows.

    Input:
        pairwise_matrix: FloatTensor(points, points)
        points_per_view: FloatTwnsor(n_views, )
    Output:
        pairwise_matrix: FloatTensor(n_views, n_views)

    """
    assert len(pairwise_matrix.shape) == 2
    assert points_per_view.sum() == pairwise_matrix.shape[0]
    assert col_reduction in ["min", "mean"]

    view_indices = torch.cat((torch.zeros(1).int(), torch.cumsum(points_per_view, 0)))
    num_views = len(points_per_view)
    view_pairwise = torch.zeros(num_views, num_views).to(pairwise_matrix)

    for i in range(num_views):
        for j in range(num_views):
            i_start, i_end = view_indices[i], view_indices[i + 1]
            j_start, j_end = view_indices[j], view_indices[j + 1]

            pairwise_ij = pairwise_matrix[i_start:i_end, j_start:j_end]

            if col_reduction == "min":
                aggregate_ij = pairwise_ij.min(dim=1).values.mean()
            elif col_reduction == "mean":
                aggregate_ij = pairwise_ij.mean()

            view_pairwise[i, j] = aggregate_ij

    if symmetrical:
        view_pairwise = 0.5 * (view_pairwise + view_pairwise.t())

    return view_pairwise


def compute_pairwise_geodesic(poses):
    """
    Computes the pairwise angles between all view pairs

    Input:
        poses: (num_views, 3, 3)
    Output:
        pairwise_poses: (num_views, num_views) pairwise angle in radians
    """
    relposes = poses[:, None] @ poses.transpose(1, 2)[None, :]
    relposes_trace = relposes[..., 0, 0] + relposes[..., 1, 1] + relposes[..., 2, 2]
    cos_angle = 0.5 * relposes_trace - 0.5

    # clamp for safety
    relpose_mag = cos_angle.clamp(min=-1, max=1).acos()

    # enforce diagonal is zero because the above operations have some instability
    relpose_mag.fill_diagonal_(0)
    return relpose_mag


def extract_features(model, test_loader):
    # extract features
    outputs = []
    with torch.inference_mode():
        test_loader_pbar = tqdm(test_loader, desc="extract features")
        for batch in test_loader_pbar:
            image = batch["image"].cuda()
            mask = batch["depth"] > 0.15  # threshold .. should be 0s or 1s
            nocs = batch["nocs_map"]
            c_id = batch["class_id"]
            pose = batch["o2w_pose"]

            feat = model(image)
            patch_size = model.patch_size

            mask = center_padding(mask, patch_size)
            nocs = center_padding(nocs, patch_size)

            # average pool mask and nocs
            mask = nn_F.avg_pool2d(mask.float(), patch_size)

            # average NOCS, not optimal; similar issue to averaging depth
            # divide by mask value to account for averaged zeros
            nocs = nn_F.avg_pool2d(nocs, patch_size)
            nocs = nocs / mask.clamp(min=patch_size ** -2)

            # a bit arbitrary; include pixel in mask if more that 1/4 of it was there
            mask = (mask >= 0.25).float()

            outputs.append((feat.detach().cpu(), mask, nocs, c_id, pose))

    # aggregate ouputs
    feats = torch.cat([out[0] for out in outputs], dim=0)
    masks = torch.cat([out[1] for out in outputs], dim=0)
    smaps = torch.cat([out[2] for out in outputs], dim=0)
    c_ids = torch.cat([out[3] for out in outputs], dim=0)
    poses = torch.cat([out[4] for out in outputs], dim=0)

    return feats, masks, smaps, c_ids, poses


def mask_to_coordinate(masks, normalize=True):
    num_instances, _, h, w = masks.shape
    masks = masks.float()
    mesh_i, mesh_j = torch.meshgrid(
        torch.arange(h) / h, torch.arange(w) / w, indexing="ij"
    )
    mesh_k = torch.zeros_like(mesh_i)
    coord = torch.stack((mesh_i, mesh_j, mesh_k)).to(masks).unsqueeze(0)
    coord = coord * masks

    # normalize coords
    if normalize:
        coord_flat = coord.view(num_instances, -1, h * w)
        coord_min = coord_flat.min(dim=-1).values[:, :, None, None]
        coord_max = coord_flat.max(dim=-1).values[:, :, None, None]
        coord = (coord - coord_min) / (coord_max - coord_min).clamp(min=1e-5)

    return coord
