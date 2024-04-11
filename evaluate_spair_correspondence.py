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
from einops import einsum
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from evals.datasets.spair import CLASS_IDS, SPairDataset
from evals.utils.correspondence import argmax_2d


def compute_errors(model, instance, mask_feats=False, return_heatmaps=False):
    img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, _ = instance
    mask_i = torch.tensor(np.array(mask_i, dtype=float))
    mask_j = torch.tensor(np.array(mask_j, dtype=float))

    images = torch.stack((img_i, img_j)).cuda()
    masks = torch.stack((mask_i, mask_j)).cuda()
    masks = torch.nn.functional.avg_pool2d(masks.float(), 16)
    masks = masks > 4 / (16**2)

    feats = model(images)
    if isinstance(feats, list):
        feats = torch.cat(feats, dim=1)

    feats = nn_F.normalize(feats, p=2, dim=1)

    if mask_feats:
        feats = feats * masks

    feats_i = feats[0]
    feats_j = feats[1]

    # normalize kps to [0, 1]
    assert images.shape[-1] == images.shape[-2], "assuming square images here"
    kps_i = kps_i.float()
    kps_j = kps_j.float()
    kps_i[:, :2] = kps_i[:, :2] / images.shape[-1]
    kps_j[:, :2] = kps_j[:, :2] / images.shape[-1]

    # get correspondences
    kps_i_ndc = (kps_i[:, :2].float() * 2 - 1)[None, None].cuda()
    kp_i_F = nn_F.grid_sample(
        feats_i[None, :], kps_i_ndc, mode="bilinear", align_corners=True
    )
    kp_i_F = kp_i_F[0, :, 0].t()

    # get max index in [0,1] range
    heatmaps = einsum(kp_i_F, feats_j, "k f, f h w -> k h w")
    pred_kp = argmax_2d(heatmaps, max_value=True).float().cpu() / feats.shape[-1]

    # compute error and scale to threshold (for all pairs)
    errors = (pred_kp[:, None, :] - kps_j[None, :, :2]).norm(p=2, dim=-1)
    errors = errors / thresh_scale

    # only retain keypoints in both (for now)
    valid_kps = (kps_i[:, None, 2] * kps_j[None, :, 2]) == 1
    in_both = valid_kps.diagonal()

    # max error should be 1, so this excludes invalid from NN-search
    errors[valid_kps.logical_not()] = 1e3

    error_same = errors.diagonal()[in_both]
    error_nn, index_nn = errors[in_both].min(dim=1)
    index_same = in_both.nonzero().squeeze(1)

    if return_heatmaps:
        return error_same, error_nn, index_same, index_nn, heatmaps
    else:
        return error_same, error_nn, index_same, index_nn


def evaluate_dataset(model, dataset, thresh, verbose=False):
    pbar = tqdm(range(len(dataset)), ncols=60) if verbose else range(len(dataset))
    error_output = [compute_errors(model, dataset.__getitem__(i)) for i in pbar]

    errors = torch.cat([_err[0] for _err in error_output])
    src_ind = torch.cat([_err[2] for _err in error_output])
    tgt_ind = torch.cat([_err[3] for _err in error_output])

    # compute confusion matrix
    kp_max = max(src_ind.max(), tgt_ind.max()) + 1
    confusion = torch.zeros((kp_max, kp_max))
    for src, tgt in torch.stack((src_ind, tgt_ind), dim=1):
        confusion[src, tgt] += 1

    # compute recall
    recall = (errors < thresh).float().mean().item() * 100.0

    return recall, confusion


@hydra.main("./configs", "spair_correspondence", None)
def main(cfg: DictConfig):
    data_root = "/nfs/turbo/justincj-turbo/mbanani/projects/geoeval/data/SPair-71k"
    thresh = 0.10

    # ===== Get model =====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda")

    # ===== GET DATA LOADERS =====
    if cfg.eval_class == "all":
        classes = list(CLASS_IDS.keys())
    else:
        assert cfg.eval_class in CLASS_IDS
        classes = [cfg.eval_class]

    class_acc = {}
    for class_name in classes:
        recall = []
        confusion = []
        for vp_diff in [0, 1, 2, None]:
            dataset = SPairDataset(
                data_root,
                cfg.split,
                use_bbox=cfg.use_bbox,
                image_size=cfg.image_size,
                image_mean=cfg.image_mean,
                class_name=class_name,
                num_instances=cfg.num_instances,
                vp_diff=vp_diff,
            )
            vp_diff = "all" if vp_diff is None else f"{vp_diff:3d}"
            if len(dataset) > 0:
                rec_i, conf_i = evaluate_dataset(model, dataset, thresh)
                logger.info(
                    f"Recall@{thresh} {class_name:>13s} {vp_diff} |  {rec_i:6.2f}"
                )
            else:
                logger.info(f"Recall@{thresh} {class_name:>13s} {vp_diff} |  N/A")
                rec_i, conf_i = -1, None
            recall.append(rec_i)
            confusion.append(conf_i)

        result_log = [f"{_rec:5.1f}" if _rec >= 0 else " N/A " for _rec in recall]
        result_log = "   ".join(result_log)
        logger.info(f"Recall@{thresh} {class_name:>13s}     |  {result_log}")
        class_acc[class_name] = (recall, confusion)

    all_recall = [torch.tensor(class_acc[cls][0], dtype=float) for cls in class_acc]
    all_recall = torch.stack(all_recall, dim=0)
    valid_rec = (all_recall >= 0).float()  # invalid is set to -1
    avg_recall = (all_recall * valid_rec).sum(dim=0) / valid_rec.sum(dim=0)

    for i, vp_diff in enumerate(["0", "1", "2", "all"]):
        logger.info(f"Recall@{thresh}  view diff={vp_diff:>3s} |  {avg_recall[i]:6.2f}")

    # result summary
    time = datetime.now().strftime("%d%m%Y-%H%M")
    exp_info = ", ".join(
        [
            f"{model.checkpoint_name:30s}",
            f"{model.patch_size:2d}",
            f"{str(model.layer):5s}",
            f"{model.output:10s}",
            "SPair-71k",
            cfg.split,
            f"{cfg.eval_class:>13s}",
            f"{cfg.num_instances:5d}",
        ]
    )
    results = ", ".join([f"{avg_recall[i]:6.2f}" for i in range(4)])
    log = f"{time}, {exp_info}, {results} \n"
    with open("spair_correspondence.log", "a") as f:
        f.write(log)


if __name__ == "__main__":
    main()
