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
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.utils.correspondence import (
    compute_binned_performance,
    estimate_correspondence_xyz,
    project_3dto2d,
)
from evals.utils.transformations import so3_rotation_angle, transform_points_Rt


@hydra.main("./configs", "navi_correspondence", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda")
    loader = build_loader(cfg.dataset, "test", 4, 1, pair_dataset=True)
    _ = loader.dataset.__getitem__(0)

    # extract features
    feats_0 = []
    feats_1 = []
    xyz_grid_0 = []
    xyz_grid_1 = []
    Rt_gt = []
    intrinsics = []

    for batch in tqdm(loader):
        feat_0 = model(batch["image_0"].cuda())
        feat_1 = model(batch["image_1"].cuda())
        if cfg.multilayer:
            feat_0 = torch.cat(feat_0, dim=1)
            feat_1 = torch.cat(feat_1, dim=1)
        feats_0.append(feat_0.detach().cpu())
        feats_1.append(feat_1.detach().cpu())
        Rt_gt.append(batch["Rt_01"])
        intrinsics.append(batch["intrinsics_1"])

        # scale down to avoid a huge matching problem
        xyz_grid_0_i = nn_F.interpolate(
            batch["xyz_grid_0"], scale_factor=cfg.scale_factor, mode="nearest"
        )
        xyz_grid_1_i = nn_F.interpolate(
            batch["xyz_grid_1"], scale_factor=cfg.scale_factor, mode="nearest"
        )
        xyz_grid_0.append(xyz_grid_0_i)
        xyz_grid_1.append(xyz_grid_1_i)

    feats_0 = torch.cat(feats_0, dim=0)
    feats_1 = torch.cat(feats_1, dim=0)
    xyz_grid_0 = torch.cat(xyz_grid_0, dim=0)
    xyz_grid_1 = torch.cat(xyz_grid_1, dim=0)
    Rt_gt = torch.cat(Rt_gt, dim=0).float()[:, :3, :4]
    intrinsics = torch.cat(intrinsics, dim=0).float()

    num_instances = len(loader.dataset)
    err_3d = []
    err_2d = []
    for i in tqdm(range(num_instances)):
        c_xyz0, c_xyz1, c_dist, c_uv0, c_uv1 = estimate_correspondence_xyz(
            feats_0[i], feats_1[i], xyz_grid_0[i], xyz_grid_1[i], cfg.num_corr
        )

        c_uv0 = c_uv0 / cfg.scale_factor
        c_uv1 = c_uv1 / cfg.scale_factor

        c_xyz0in1 = transform_points_Rt(c_xyz0, Rt_gt[i].float())
        c_err3d = (c_xyz0in1 - c_xyz1).norm(p=2, dim=1)

        c_xyz1in1_uv = project_3dto2d(c_xyz1, intrinsics[i])
        c_xyz0in1_uv = project_3dto2d(c_xyz0in1, intrinsics[i])
        c_err2d = (c_xyz0in1_uv - c_xyz1in1_uv).norm(p=2, dim=1)

        err_3d.append(c_err3d.detach().cpu())
        err_2d.append(c_err2d.detach().cpu())

    err_3d = torch.stack(err_3d, dim=0).float()
    err_2d = torch.stack(err_2d, dim=0).float()
    results = []

    metric_thresh = [0.01, 0.02, 0.05]
    for _th in metric_thresh:
        recall_i = 100 * (err_3d < _th).float().mean()
        print(f"Recall at {_th:>.2f} m:  {recall_i:.2f}")
        results.append(f"{recall_i:5.02f}")

    px_thresh = [5, 25, 50]
    for _th in px_thresh:
        recall_i = 100 * (err_2d < _th).float().mean()
        print(f"Recall at {_th:>3d}px:  {recall_i:.2f}")
        results.append(f"{recall_i:5.02f}")

    # compute rel_ang
    rel_ang = so3_rotation_angle(Rt_gt[:, :3, :3])
    rel_ang = rel_ang * 180.0 / np.pi

    # compute thresholded recall -- 0.2decimeter = 2cm
    rec_2cm = (err_3d < 0.02).float().mean(dim=1)
    bin_rec = compute_binned_performance(rec_2cm, rel_ang, [0, 30, 60, 90, 120])
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
            str(cfg.num_corr),
            str(cfg.scale_factor),
        ]
    )
    dset = loader.dataset.name
    results = ", ".join(results)
    log = f"{time}, {exp_info}, {dset}, {results} \n"
    with open("navi_correspondence.log", "a") as f:
        f.write(log)


if __name__ == "__main__":
    main()
