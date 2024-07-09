from __future__ import annotations

import timm
import torch
from timm.layers import resample_abs_pos_embed

from .utils import center_padding, tokens_to_output


class SigLIP(torch.nn.Module):
    def __init__(
        self,
        checkpoint="vit_large_patch16_siglip_384",
        output="dense",
        layer=-1,
        resize_pos_embeds=True,
        pretrained=True,
        return_multilayer=False,
    ):
        super().__init__()

        assert output in ["gap", "dense"], "Options: [cls, gap, dense]"
        self.output = output
        self.checkpoint_name = checkpoint

        self.vit = timm.create_model(checkpoint, pretrained=pretrained)
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.embed_size = self.vit.patch_embed.grid_size
        self.resize_pos_embeds = resize_pos_embeds
        self.vit.patch_embed.strict_img_size = False

        feat_dim = self.vit.num_features
        num_layers = len(self.vit.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]
        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):
        images = center_padding(images, self.patch_size)
        _, _, img_h, img_w = images.shape

        # get embed h, w
        assert img_h % self.patch_size == 0
        assert img_w % self.patch_size == 0
        out_h, out_w = img_h // self.patch_size, img_w // self.patch_size

        if self.resize_pos_embeds and (out_h, out_w) != self.embed_size:
            self.embed_size = (out_h, out_w)
            self.vit.pos_embed.data = resample_abs_pos_embed(
                self.vit.pos_embed, (out_h, out_w), num_prefix_tokens=0
            )

        x = self.vit.patch_embed(images)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for i, x_i in enumerate(embeds):
            x_i = tokens_to_output(self.output, x_i, None, (out_h, out_w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
