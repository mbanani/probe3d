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
import torch
from loguru import logger


def depth_rmse(depth_pr, depth_gt, image_average=False):
    assert depth_pr.shape == depth_gt.shape, f"{depth_pr.shape} != {depth_gt.shape}"

    if len(depth_pr.shape) == 4:
        depth_pr = depth_pr.squeeze(1)
        depth_gt = depth_gt.squeeze(1)

    # compute RMSE for each image and then average
    valid = (depth_gt > 0).detach().float()

    # clamp to 1 for empty depth images
    num_valid = valid.sum(dim=(1, 2))
    if (num_valid == 0).any():
        num_valid = num_valid.clamp(min=1)
        logger.warning("GT depth is empty. Clamping to avoid error.")

    # compute pixelwise squared error
    sq_error = (depth_gt - depth_pr).pow(2)
    sum_masked_sqe = (sq_error * valid).sum(dim=(1, 2))
    rmse_image = (sum_masked_sqe / num_valid).sqrt()

    return rmse_image.mean() if image_average else rmse_image


def evaluate_depth(
    depth_pr, depth_gt, image_average=False, scale_invariant=False, nyu_crop=False
):
    assert depth_pr.shape == depth_gt.shape, f"{depth_pr.shape} != {depth_gt.shape}"

    if len(depth_pr.shape) == 4:
        depth_pr = depth_pr.squeeze(1)
        depth_gt = depth_gt.squeeze(1)

    if nyu_crop:
        # apply NYU crop --- commonly used in many repos for some reason
        assert depth_pr.shape[-2] == 480
        assert depth_pr.shape[-1] == 640
        depth_pr = depth_pr[..., 45:471, 41:601]
        depth_gt = depth_gt[..., 45:471, 41:601]

    if scale_invariant:
        depth_pr = match_scale_and_shift(depth_pr, depth_gt)

    # zero out invalid pixels
    valid = (depth_gt > 0).detach().float()
    depth_pr = depth_pr * valid

    # get num valid
    num_valid = valid.sum(dim=(1, 2)).clamp(min=1)

    # get recall @ thresholds
    thresh = torch.maximum(
        depth_gt / depth_pr.clamp(min=1e-9), depth_pr / depth_gt.clamp(min=1e-9)
    )
    d1 = ((thresh < 1.25 ** 1).float() * valid).sum(dim=(1, 2)) / num_valid
    d2 = ((thresh < 1.25 ** 2).float() * valid).sum(dim=(1, 2)) / num_valid
    d3 = ((thresh < 1.25 ** 3).float() * valid).sum(dim=(1, 2)) / num_valid

    # compute RMSE
    sse = (depth_gt - depth_pr).pow(2)
    mse = (sse * valid).sum(dim=(1, 2)) / num_valid
    rmse = mse.sqrt()
    metrics = {"d1": d1.cpu(), "d2": d2.cpu(), "d3": d3.cpu(), "rmse": rmse.cpu()}

    if image_average:
        for key in metrics:
            metrics[key] = metrics[key].mean()

    return metrics


def evaluate_surface_norm(snorm_pr, snorm_gt, valid, image_average=False):
    """
    Metrics to evaluate surface norm based on iDISC (and probably Fouhey et al. 2016).
    """
    snorm_pr = snorm_pr[:, :3]
    assert snorm_pr.shape == snorm_gt.shape, f"{snorm_pr.shape} != {snorm_gt.shape}"

    # compute angular error
    cos_sim = torch.cosine_similarity(snorm_pr, snorm_gt, dim=1)
    cos_sim = cos_sim.clamp(min=-1, max=1.0)
    err_deg = torch.acos(cos_sim) * 180.0 / torch.pi

    # zero out invalid errors
    assert len(valid.shape) == 4
    valid = valid.squeeze(1).float()
    err_deg = err_deg * valid
    num_valid = valid.sum(dim=(1, 2)).clamp(min=1)

    # compute rmse
    rmse = (err_deg.pow(2).sum(dim=(1, 2)) / num_valid).sqrt()

    # compute recall at thresholds
    thresh = [11.25, 22.5, 30]
    d1 = ((err_deg < thresh[0]).float() * valid).sum(dim=(1, 2)) / num_valid
    d2 = ((err_deg < thresh[1]).float() * valid).sum(dim=(1, 2)) / num_valid
    d3 = ((err_deg < thresh[2]).float() * valid).sum(dim=(1, 2)) / num_valid

    metrics = {"d1": d1.cpu(), "d2": d2.cpu(), "d3": d3.cpu(), "rmse": rmse.cpu()}

    if image_average:
        for key in metrics:
            metrics[key] = metrics[key].mean()

    return metrics


def match_scale_and_shift(prediction, target):
    # based on implementation from
    # https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0

    assert len(target.shape) == len(prediction.shape)
    if len(target.shape) == 4:
        four_chan = True
        target = target.squeeze(dim=1)
        prediction = prediction.squeeze(dim=1)
    else:
        four_chan = False

    mask = (target > 0).float()

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 *
    # a_10) . b
    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    # compute scale and shift
    scale = torch.ones_like(b_0)
    shift = torch.zeros_like(b_1)
    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    scale = scale.view(-1, 1, 1).detach()
    shift = shift.view(-1, 1, 1).detach()
    prediction = prediction * scale + shift

    return prediction[:, None, :, :] if four_chan else prediction
