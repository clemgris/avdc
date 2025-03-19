import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50  # noqa: F401


class TaskCompletionClassifier(nn.Module):
    def __init__(
        self,
        text_encoder,
        text_tokenizer,
        device="cuda",
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.device = device

        resnet = resnet18(weights=None)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_projection = nn.Linear(512, 512)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def encode_batch_text(self, batch_text):
        batch_text_ids = self.text_tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state

        return batch_text_embed  # Shape: (batch, seq_len, 512)

    def forward(self, target, text_task):
        # Encode text
        text_task_emb = self.encode_batch_text(text_task)  # (batch, seq_len, 512)

        # Encode image
        target_emb = self.image_encoder(target)  # (batch, 512, 1, 1)
        target_emb = target_emb.view(target_emb.shape[0], -1)  # (batch, 512)
        target_emb = self.image_projection(target_emb).unsqueeze(1)  # (batch, 1, 512)

        attended_features, _ = self.cross_attention(
            text_task_emb, target_emb, target_emb
        )

        prob = self.classifier(attended_features[:, 0, :])

        return prob
