import logging
import pdb
import re
import types
from typing import List, Optional, Tuple, Union

import timm
import timm.data
import torch
import torch.nn.functional as F
from timm.models.eva import Eva
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torchvision import transforms

# https://github.com/Jiawei-Yang/Denoising-ViT/blob/82704df9ba253c9696dcf3a8239434e3cbacf19d/dvt/models/vit_wrapper.py#L59

MODEL_LIST = [
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m",

    'vit_small_patch14_reg4_dinov2'
    'vit_base_patch14_reg4_dinov2'
    'vit_large_patch14_reg4_dinov2'
]

class PretrainedViTWrapper(nn.Module):

    def __init__(
        self,
        name,
        norm: bool = True,
        stride: int | None = None,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        **kwargs,
    ):
        super().__init__()
        # comment out the following line to test the models not in the list
        assert name in MODEL_LIST, f"Model type {name} not tested yet."
        self.name = name
        self.patch_size = int(re.search(r"patch(\d+)", name).group(1))
        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.model, self.config = self.create_model(name, **kwargs)
        self.embed_dim = self.model.embed_dim
        self.norm = norm

        if not stride:
            self.stride = self.model.patch_embed.proj.stride[0]

        # overwrite the stride size
        if stride and stride != self.model.patch_embed.proj.stride[0]:
            self.model.patch_embed.proj.stride = [stride, stride]

            def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
                return (img_size[0] - self.patch_size[0]) // self.proj.stride[0] + 1, (
                    img_size[1] - self.patch_size[1]
                ) // self.proj.stride[1] + 1

            self.model.patch_embed.dynamic_feat_size = types.MethodType(
                dynamic_feat_size, self.model.patch_embed
            )

    @property
    def n_output_dims(self) -> int:
        return self.model.pos_embed.shape[-1]

    @property
    def num_blocks(self) -> int:
        return len(self.model.blocks)

    @property
    def last_layer_index(self) -> int:
        return self.num_blocks - 1

    def create_model(
        self, name: str, **kwargs
    ) -> Tuple[Union[VisionTransformer, Eva], transforms.Compose]:
        model = timm.create_model(
            name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=self.dynamic_img_size,
            dynamic_img_pad=self.dynamic_img_pad,
            **kwargs,
        )
        model = model.eval()
        # Different models have different data configurations
        # e.g., their training resolution, normalization, etc, are different
        data_config = timm.data.resolve_model_data_config(model=model)
        return model, data_config

    def forward(
        self,
        x: torch.Tensor,
        n: Union[int, List[int], Tuple[int]] = 1,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Intermediate layer accessor inspired by DINO / DINOv2 interface.
        Args:
            x: Input tensor.
            n: Take last n blocks if int, all if None, select matching indices if sequence
            reshape: Whether to reshape the output.
        """
        feats = self.model.forward_intermediates(
            x,
            n,
            return_prefix_tokens=False,
            norm=self.norm,
            output_fmt="NCHW",
            intermediates_only=True,
            )[0]
        return feats
