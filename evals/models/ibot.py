from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import torch

from .ibot_transformers import vit_base, vit_large
from .utils import center_padding, tokens_to_output

BASE_URL = "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot"


class iBOT(torch.nn.Module):
    def __init__(
        self, model_type="base", output="dense", layer=-1, return_multilayer=False
    ):
        super().__init__()
        assert output in ["gap", "dense", "cls", "dense-cls"]
        self.output = output
        self.return_multilayer = return_multilayer

        model_dict = {
            "base": ("ibot_vitb16", "vitb_16/checkpoint_teacher.pth"),
            "base_in22k": ("ibot_vitb16_in22k", "vitb_16_22k/checkpoint_student.pth"),
            "large": ("ibot_vitb16", "vitl_16/checkpoint_teacher.pth"),
            "large_22k": ("ibot_vitb16_in22k", "vitl_16_pt22k/checkpoint_student.pth"),
        }

        assert model_type in model_dict

        # Download model checkpoint
        ckpt_name, ckpt_url_path = model_dict[model_type]
        ckpt_path = Path(__file__).parent / f"checkpoint_weights/{ckpt_name}.pth"
        if not ckpt_path.exists():
            download_path = f"{BASE_URL}/{ckpt_url_path}"
            urlretrieve(download_path, ckpt_path)

        # load and cleanup state dict
        state_dict = torch.load(ckpt_path)["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # instantiate model
        model_fn = vit_base if "base" in model_type else vit_large
        feat_dim = 768 if "base" in model_type else 1024
        vit = model_fn(patch_size=16, return_all_tokens=True)
        vit.load_state_dict(state_dict, strict=False)
        vit.eval()

        # set parameters
        self.vit = vit
        self.patch_size = 16
        self.checkpoint_name = ckpt_name
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        print(f"{model_type} has {num_layers} layers")
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
        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            spatial = x_i[:, 1:]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
