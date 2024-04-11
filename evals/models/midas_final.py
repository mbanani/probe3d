import types

import timm
import torch

from .utils import resize_pos_embed, tokens_to_output


def midas_forward(self, x):
    """
    Modification of timm's VisionTransformer forward
    """
    # update shapes
    h, w = x.shape[2:]
    emb_hw = (h // self.patch_size, w // self.patch_size)
    # assert h == w, f"BeIT can only handle square images, not ({h}, {w})."
    if (h, w) != self.image_size:
        self.image_size = (h, w)
        self.patch_embed.img_size = (h, w)
        self.pos_embed.data = resize_pos_embed(self.pos_embed[0], emb_hw, True)[None]

    # actual forward from beit
    x = self.patch_embed(x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.pos_embed

    x = self.norm_pre(x)

    embeds = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if i in self.multilayers:
            embeds.append(x)
            if i == self.layer:
                break

    # map tokens to output
    outputs = []
    for i, x_i in enumerate(embeds):
        x_i = tokens_to_output(self.output, x_i[:, 1:], x_i[:, 0], emb_hw)
        outputs.append(x_i)

    return outputs[0] if len(outputs) == 1 else outputs


def beit_forward_features(self, x):
    h, w = x.shape[2:]
    assert h == w, f"BeIT can only handle square images, not ({h}, {w})."
    if h != self.image_size:
        x = torch.nn.functional.interpolate(
            x, (self.image_size, self.image_size), mode="bicubic"
        )

    # beit forward features
    x = self.patch_embed(x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    x = self.pos_drop(x)

    rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    embeds = []
    for i, blk in enumerate(self.blocks):
        x = blk(x, shared_rel_pos_bias=rel_pos_bias)
        if i in self.multilayers:
            embeds.append(x)
            if len(embeds) == len(self.multilayers):
                break

    # map tokens to output
    hw = self.image_size // self.patch_size
    outputs = []
    for i, x_i in enumerate(embeds):
        x_i = tokens_to_output(self.output, x_i[:, 1:], x_i[:, 0], (hw, hw))
        outputs.append(x_i)

    return outputs[0] if len(outputs) == 1 else outputs


def make_beit_backbone(layer=-1, output="dense", midas=False, return_multilayer=False):

    if midas:
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        model = midas.pretrained.model
        model.forward = types.MethodType(midas_forward, model)
    else:
        model = timm.create_model("beit_large_patch16_384", pretrained=True)
        model.forward = types.MethodType(beit_forward_features, model)

    # add some parameters
    model.checkpoint_name = "midas_vit_l16_384" if midas else "beit_vit_l16_384"

    # set parameters for feature extraction
    model.image_size = (384, 384)
    model.patch_size = 16
    model.output = output

    feat_dim = 1024

    num_layers = len(model.blocks)
    multilayers = [
        num_layers // 4 - 1,
        num_layers // 2 - 1,
        num_layers // 4 * 3 - 1,
        num_layers - 1,
    ]

    if return_multilayer:
        model.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
        model.multilayers = multilayers
    else:
        model.feat_dim = feat_dim
        layer = multilayers[-1] if layer == -1 else layer
        model.multilayers = [layer]

    # define layer name (for logging)
    model.layer = "-".join(str(_x) for _x in model.multilayers)

    return model
