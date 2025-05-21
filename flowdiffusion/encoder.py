import torch
import torch.nn as nn
from einops import rearrange
from r3m import load_r3m
from transformers import (
    AutoImageProcessor,
    AutoModel,
)
from vit_wrapper import PretrainedViTWrapper

# Visual encoder


class DinoV2Encoder(nn.Module):
    def __init__(self, name="facebook/dinov2-base", do_rescale=False):
        super().__init__()
        self.emb_dim = 768
        self.processor = AutoImageProcessor.from_pretrained(name, do_rescale=do_rescale)
        self.model = AutoModel.from_pretrained(name)

    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs[0]

        cls_emb = last_hidden_states[:, 0, :]
        patch_emb = last_hidden_states[:, 1:, :]
        return cls_emb, patch_emb


class ViTEncoder(nn.Module):
    def __init__(self, name="vit_base_patch14_dinov2.lvd142m", device="cuda"):
        super().__init__()
        self.backbone = PretrainedViTWrapper(name=name, norm=True)
        self.backbone = self.backbone.to(device)

    def forward(self, x):
        patch_emb = self.backbone(x).detach()
        patch_emb = rearrange(patch_emb, "b c h w -> b (h w) c")
        return None, patch_emb


class R3MEncoder(nn.Module):
    def __init__(self, name="resnet50", device="cuda"):
        super().__init__()
        r3m = load_r3m(name).to(device)
        r3m.eval()
        resnet_backbone = r3m.module.convnet
        self.patch_encoder = nn.Sequential(*list(resnet_backbone.children())[:-2])

    def forward(self, x):
        x = self.patch_encoder(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return None, x
