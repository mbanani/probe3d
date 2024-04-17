"""
Backbone definition for the RADIO model from:

AM-RADIO: Agglomerative Vision Foundation Model -- Reduce All Domains Into One
https://arxiv.org/abs/2312.06709
"""
import torch

from .utils import center_padding, tokens_to_output


class RADIO(torch.nn.Module):
    """
    Backbone definition for the RADIO model from.

    Args:
        version (str): Version of the model to load.
        output (str): Type of output to return. One of ['dense', 'dense-cls'].
        return_multilayer (bool): Return features from multiple layers.
    """

    def __init__(
        self,
        version="radio_v2",
        output="dense",
        return_multilayer=False,
    ):
        super().__init__()

        # Get model from TorchHub.
        self.version = version
        self.checkpoint_name = f"{version}"
        radio = torch.hub.load('NVlabs/RADIO',
                               'radio_model',
                               version=self.version,
                               progress=True,
                               adaptor_names=None)
        radio.make_preprocessor_external()
        self.radio = radio.eval().to(torch.float32)

        assert output in ["dense", "dense-cls"]
        self.output = output

        patch_gen = radio.model.patch_generator
        # Cropped Positional Embedding (CPE) case.
        patch_size = patch_gen.patch_size
        self.patch_size = patch_size

        feat_dim = radio.model.embed_dim
        # Double the feature dimension if dense-cls output is requested.
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        # Layers to return activations of in case of "return_multilayer=True".
        num_layers = len(radio.model.blocks)
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
            layer = multilayers[-1]
            self.multilayers = [layer]

        # Define layer name (for logging).
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward_features(self, x):
        """Return features from the model."""
        features = []
        x = self.radio.model.patch_generator(x)

        for i, blk in enumerate(self.radio.model.blocks):
            x = blk(x)
            if i in self.multilayers:
                # normalize intermediates with final norm layer if enabled
                features.append(self.radio.model.norm(x))

        return features

    def forward(self, images):
        """Main forward routine."""
        # Pad images (if needed) to ensure it matches patch_size.
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        intermediate_features = self.forward_features(images)

        outputs = []
        for features in intermediate_features:
            # Pick the 1st summary token.
            summary = features[:, 0]
            patches = features[:, self.radio.model.patch_generator.num_skip :]
            output = tokens_to_output(self.output, patches, summary, (h, w))
            outputs.append(output)

        return outputs[0] if len(outputs) == 1 else outputs
