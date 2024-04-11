# flake8: noqa
from __future__ import annotations

import math
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.nn.functional import interpolate
from torchvision import transforms

sys.path.insert(0, "/nfs/turbo/justincj-turbo/mbanani/projects/zero123/zero123")

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, model, sampler, n_samples=1, scale=3):

    h, w = input_im.shape[-2:]
    x, y, z = 0, 0, 0

    with model.ema_scope():
        input_im = input_im.unsqueeze(0)
        c = model.get_learned_conditioning(input_im)
        c = c.tile(n_samples, 1, 1)
        T = torch.tensor(
            [math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), z]
        )
        T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
        c = torch.cat([c, T], dim=-1)
        c = model.cc_projection(c)
        cond = {}
        cond["c_crossattn"] = [c]
        cond["c_concat"] = [
            model.encode_first_stage(input_im.to(c.device))
            .mode()
            .detach()
            .repeat(n_samples, 1, 1, 1)
        ]
        if scale != 1.0:
            uc = {}
            uc["c_concat"] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
            uc["c_crossattn"] = [torch.zeros_like(c).to(c.device)]
        else:
            uc = None

        shape = [n_samples, 4, h // 8, w // 8]
        # sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta)
        device = sampler.model.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        step = 1
        ts = torch.full((b,), step, device=device, dtype=torch.long)

        x_in = torch.cat([img] * 2)
        t_in = torch.cat([ts] * 2)

        # taken from ldm/models/diffusion/ddim -- p_sample_ddim
        assert isinstance(uc, dict)
        c_in = dict()
        for k in cond:
            if isinstance(cond[k], list):
                c_in[k] = [
                    torch.cat([uc[k][i], cond[k][i]]) for i in range(len(cond[k]))
                ]
            else:
                c_in[k] = torch.cat([uc[k], cond[k]])

        # multiscale_feat = sampler.model.apply_model(x_in, t_in, c_in)
        c_concat = c_in["c_concat"]
        c_crossattn = c_in["c_crossattn"]
        xc = torch.cat([x_in] + c_concat, dim=1)
        cc = torch.cat(c_crossattn, 1)
        ms_feats = sampler.model.model.diffusion_model(
            xc, t_in, context=cc, return_feats=True
        )

        combined_feats = []
        for i in range(4):
            e_t_uncond, e_t = ms_feats[i]
            e_t = e_t_uncond + scale * (e_t - e_t_uncond)
            combined_feats.append(e_t)

        return combined_feats


def get_zero123():
    ckpt = "/nfs/turbo/justincj-turbo/mbanani/projects/zero123/zero123/105000.ckpt"
    config = "/nfs/turbo/justincj-turbo/mbanani/projects/zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml"
    device = "cuda:0"
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.
    model = load_model_from_config(config, ckpt, device=device)
    return model


class Zero123(torch.nn.Module):
    def __init__(self, time_step=1, output="dense", layer=1, return_multilayer=False):
        super().__init__()
        assert output in ["gap", "dense"], "Only supports gap or dense output"

        self.output = output
        self.time_step = time_step
        self.checkpoint_name = f"zero123_t-{time_step}"
        self.patch_size = 16

        self.model = get_zero123()
        self.sampler = DDIMSampler(self.model)

        self.up_ft_index = [0, 1, 2, 3]  # keep all the upblock feats
        assert layer in [-1, 0, 1, 2, 3]

        feat_dims = [1280, 1280, 640, 320]
        multilayers = [0, 1, 2, 3]

        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            layer = multilayers[-1] if layer == -1 else layer
            self.feat_dim = feat_dims[layer]
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):
        spatial_lists = []
        batch_size = images.shape[0]

        for i in range(batch_size):
            output = sample_model(images[i], self.model, self.sampler)
            spatial_lists.append(output)

        # concat
        spatial = [
            torch.stack([sp[0] for sp in spatial_lists]),
            torch.stack([sp[1] for sp in spatial_lists]),
            torch.stack([sp[2] for sp in spatial_lists]),
            torch.stack([sp[3] for sp in spatial_lists]),
        ]

        h, w = images.shape[2] // self.patch_size, images.shape[3] // self.patch_size
        spatial = [spatial[i] for i in self.multilayers]

        assert self.output in ["gap", "dense"]
        if self.output == "gap":
            spatial = [x.mean(dim=(2, 3)) for x in spatial]
        elif self.output == "dense":
            spatial = [interpolate(x.contiguous(), (h, w)) for x in spatial]

        return spatial[0] if len(spatial) == 1 else spatial
