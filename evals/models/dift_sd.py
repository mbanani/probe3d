""" Code taken from the DIFT repo: github:Tsingularity/dift"""

import gc
from typing import Optional, Union

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from loguru import logger


class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
    ):
        r"""
        Args:
            sample (`torch.FloatTensor`):
                (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`):
                (batch, sequence_length, feature_dim) encoder hidden states
        """
        # By default samples have to be AT least a multiple of the overall upsampling
        # factor.The overall upsampling factor is equal to 2 ** (#upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any i
        # upsampling size on the fly if necessary.
        default_overall_up_factor = 2 ** self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of
        # `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        # project
        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb, None)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            _has_attr = hasattr(downsample_block, "has_cross_attention")
            if _has_attr and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=None,
            )

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            if i > np.max(up_ft_indices):
                break

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            _has_attr = hasattr(upsample_block, "has_cross_attention")
            if _has_attr and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=None,
                    upsample_size=upsample_size,
                    attention_mask=None,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

            if i in up_ft_indices:
                up_ft[i] = sample

        output = {}
        output["up_ft"] = up_ft
        return output


class OneStepSDPipeline(StableDiffusionPipeline):
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        device = self._execution_device

        scale_factor = self.vae.config.scaling_factor
        latents = scale_factor * self.vae.encode(img_tensor).latent_dist.mode()

        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(
            latents_noisy, t, up_ft_indices, encoder_hidden_states=prompt_embeds
        )
        return unet_output


class SDFeaturizer(torch.nn.Module):
    def __init__(self, sd_id="stabilityai/stable-diffusion-2-1"):
        super().__init__()

        breakpoint()
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(
            sd_id, unet=unet, safety_checker=None
        )
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(
            sd_id, subfolder="scheduler"
        )
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        # self.pipe = onestep_pipe

        self.tokenizer = onestep_pipe.tokenizer
        self.text_encoder = onestep_pipe.text_encoder
        self.unet = onestep_pipe.unet
        self.vae = onestep_pipe.vae
        self.scheduler = onestep_pipe.scheduler

        for name, param in self.named_parameters():
            if name.split(".")[0] != "unet":
                param.requires_grad = False

    def forward(self, images, prompts, t=1, up_ft_index=[1, 4, 7]):
        """
        Args:
            img_tensor: should be a single tensor of shape [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: upsampling block of the U-Net for feat. extract. [0, 1, 2, 3]
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        """
        device = images.device

        with torch.no_grad():
            prompt_embeds = self.encode_prompt(
                prompt=prompts, device=device
            )  # [1, 77, dim]

            # what was happening in the pipeline
            scale_factor = self.vae.config.scaling_factor
            latents = scale_factor * self.vae.encode(images).latent_dist.mode()

            t = torch.tensor(t, dtype=torch.long, device=device)
            noise = torch.randn_like(latents).to(device)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

        unet_output = self.unet(
            latents_noisy, t, up_ft_index, encoder_hidden_states=prompt_embeds.detach()
        )
        return unet_output["up_ft"]

    def encode_prompt(self, prompt, device):
        """
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
        """
        # function of text encoder can correctly access it
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "Input truncated because CLIP only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        _has_attr_attn = hasattr(self.text_encoder.config, "use_attention_mask")
        if _has_attr_attn and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask
        )
        prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # check if this is needed TODO
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

        return prompt_embeds
