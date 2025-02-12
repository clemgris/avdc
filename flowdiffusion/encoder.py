import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


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
