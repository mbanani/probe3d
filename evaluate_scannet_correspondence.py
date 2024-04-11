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
from datetime import datetime

import hydra
import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from evals.datasets.scannet_pairs import ScanNetPairsDataset
from evals.utils.correspondence import (
    compute_binned_performance,
    estimate_correspondence_depth,
    project_3dto2d,
)
from evals.utils.transformations import so3_rotation_angle, transform_points_Rt


@hydra.main("./configs", "scannet_correspondence", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda")
    dataset = ScanNetPairsDataset()
    loader = DataLoader(
        dataset, 8, num_workers=4, drop_last=False, pin_memory=True, shuffle=False
    )

    # extract features
    err_2d = []
    R_gt = []
    for i in tqdm(range(len(dataset))):
        instance = dataset.__getitem__(i)
        rgbs = torch.stack((instance["rgb_0"], instance["rgb_1"]), dim=0)
        deps = torch.stack((instance["depth_0"], instance["depth_1"]), dim=0)
        K_mat = instance["K"].clone()
        Rt_gt = instance["Rt_1"].float()[:3, :4]
        R_gt.append(Rt_gt[:3, :3])

        feats = model(rgbs.cuda())
        if cfg.multilayer:
            feats = torch.cat(feats, dim=1)

        # scale depth and intrinsics
        feats = feats.detach().cpu()
        deps = nn_F.interpolate(deps, scale_factor=cfg.scale_factor)
        K_mat[:2, :] *= cfg.scale_factor

        # compute corr
        corr_xyz0, corr_xyz1, corr_dist = estimate_correspondence_depth(
            feats[0], feats[1], deps[0], deps[1], K_mat.clone(), cfg.num_corr
        )

        # compute error
        corr_xyz0in1 = transform_points_Rt(corr_xyz0, Rt_gt)
        uv_0in1 = project_3dto2d(corr_xyz0in1, K_mat.clone())
        uv_1in1 = project_3dto2d(corr_xyz1, K_mat.clone())
        corr_err2d = (uv_0in1 - uv_1in1).norm(p=2, dim=1)
        err_2d.append(corr_err2d.detach().cpu())

    err_2d = torch.stack(err_2d, dim=0).float()
    R_gt = torch.stack(R_gt, dim=0).float()

    """
    feats_0 = []
    feats_1 = []
    depth_0 = []
    depth_1 = []
    K_mat = []
    Rt_gt = []

    for batch in tqdm(loader):
        feat_0 = model(batch["rgb_0"].cuda())
        feat_1 = model(batch["rgb_1"].cuda())
        if cfg.multilayer:
            feat_0 = torch.cat(feat_0, dim=1)
            feat_1 = torch.cat(feat_1, dim=1)
        feats_0.append(feat_0.detach().cpu())
        feats_1.append(feat_1.detach().cpu())
        depth_0.append(batch["depth_0"])
        depth_1.append(batch["depth_1"])
        K_mat.append(batch["K"])
        Rt_gt.append(batch["Rt_1"])

    feats_0 = torch.cat(feats_0, dim=0)
    feats_1 = torch.cat(feats_1, dim=0)
    depth_0 = torch.cat(depth_0, dim=0)
    depth_1 = torch.cat(depth_1, dim=0)
    K_mat = torch.cat(K_mat, dim=0)
    Rt_gt = torch.cat(Rt_gt, dim=0).float()[:, :3, :4]

    depth_0 = nn_F.interpolate(depth_0, scale_factor=cfg.scale_factor)
    depth_1 = nn_F.interpolate(depth_1, scale_factor=cfg.scale_factor)
    K_mat[:, :2, :] *= cfg.scale_factor

    err_2d = []
    num_instances = len(loader.dataset)
    for i in tqdm(range(num_instances)):
        corr_xyz0, corr_xyz1, corr_dist = estimate_correspondence_depth(
            feats_0[i],
            feats_1[i],
            depth_0[i],
            depth_1[i],
            K_mat[i].clone(),
            cfg.num_corr,
        )

        corr_xyz0in1 = transform_points_Rt(corr_xyz0, Rt_gt[i].float())
        uv_0in1 = project_3dto2d(corr_xyz0in1, K_mat[i].clone())
        uv_1in1 = project_3dto2d(corr_xyz1, K_mat[i].clone())
        corr_err2d = (uv_0in1 - uv_1in1).norm(p=2, dim=1)

        err_2d.append(corr_err2d.detach().cpu())

    err_2d = torch.stack(err_2d, dim=0).float()
    """

    results = []
    # compute 2D errors
    px_thresh = [5, 10, 20]
    for _th in px_thresh:
        recall_i = 100 * (err_2d < _th).float().mean()
        print(f"Recall at {_th:>2d} pixels:  {recall_i:.2f}")
        results.append(f"{recall_i:5.02f}")

    # compute rel_ang
    rel_ang = so3_rotation_angle(R_gt)
    rel_ang = rel_ang * 180.0 / np.pi

    # compute thresholded recall
    rec_10px = (err_2d < 10).float().mean(dim=1)
    bin_rec = compute_binned_performance(rec_10px, rel_ang, [0, 15, 30, 60, 180])
    for bin_acc in bin_rec:
        results.append(f"{bin_acc * 100:5.02f}")

    # # result summary
    time = datetime.now().strftime("%d%m%Y-%H%M")
    exp_info = ", ".join(
        [
            f"{model.checkpoint_name:30s}",
            f"{model.patch_size:2d}",
            f"{str(model.layer):5s}",
            f"{model.output:10s}",
            loader.dataset.name,
            str(cfg.num_corr),
            str(cfg.scale_factor),
        ]
    )
    results = ", ".join(results)
    log = f"{time}, {exp_info}, {results} \n"
    with open("scannet_correspondence.log", "a") as f:
        f.write(log)


if __name__ == "__main__":
    main()
