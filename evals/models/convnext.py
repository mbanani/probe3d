from __future__ import annotations

import open_clip
import timm
import torch
import torch.nn.functional as F
from torch import nn

from .utils import center_padding


class ConvNext(nn.Module):
    def __init__(
        self,
        arch="convnext_base_w",
        checkpoint="laion2b_s13b_b82k",
        output="dense",
        layer=-1,
        return_multilayer=False,
    ):
        super().__init__()
        assert output in ["gap", "dense"]
        self.output = output
        self.patch_size = 16  # is this true for convnext?

        if "laion2b" in checkpoint:
            self.checkpoint_name = f"{arch}_{checkpoint}"

            # Initialize a pre-trained CLIP image encoder and freeze it.
            convnext, _, _ = open_clip.create_model_and_transforms(
                arch, pretrained=checkpoint
            )
            convnext = convnext.eval().to(torch.float32)

            self.visual = convnext.visual.trunk
            del convnext
        elif arch == "convnext_base" and checkpoint == "in22k":
            self.checkpoint_name = "convnext_base_in22k"
            self.visual = timm.create_model("convnext_base_in22k", pretrained=True)
            self.visual = self.visual.eval().to(torch.float32)
        elif arch == "convnext_base" and checkpoint == "fcmae_ft_in22k_in1k_384":
            self.checkpoint_name = "convnext_base_fcmae_ft_in22k_in1k_384"
            self.visual = timm.create_model(
                "convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=True
            )
            self.visual = self.visual.eval().to(torch.float32)

        else:
            raise ValueError()

        self.checkpoint_name = self.checkpoint_name.replace("convnext", "cnxt")
        self.checkpoint_name = self.checkpoint_name.replace("base", "b")

        # Extract some attributes from CLIP module for easy access.
        assert layer in [-1, 0, 1, 2, 3]
        assert len(self.visual.stages) == 4

        # very hacky .. probably there's a config somewhere
        feat_dims = [
            _stg.blocks[-1].norm.normalized_shape[0] for _stg in self.visual.stages
        ]
        multilayers = [0, 1, 2, 3]

        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            layer = multilayers[-1] if layer == -1 else layer
            self.feat_dim = feat_dims[layer]
            self.multilayers = [layer]

        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):
        images = center_padding(images, self.patch_size)
        img_h, img_w = images.shape[-2:]
        out_hw = (img_h // self.patch_size, img_w // self.patch_size)

        # clip stuff
        x = self.visual.stem(images)

        embeds = []
        for i, stage in enumerate(self.visual.stages):
            x = stage(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for i, x_i in enumerate(embeds):
            if self.output == "dense":
                x_i = F.interpolate(x_i, out_hw, mode="bilinear")
            else:
                x_i = x_i.mean(dim=(2, 3))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
