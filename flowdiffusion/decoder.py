import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def create_normal_dist(
    x,
    std=None,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    if std is None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist


class TransposedConvDecoder(nn.Module):
    def __init__(
        self,
        observation_shape=(3, 224, 224),
        emb_dim=768,
        patch_size=16,
        activation=nn.ReLU,
        depth=32,
        kernel_size=5,
        stride=2,
    ):
        super().__init__()

        activation = activation()
        self.observation_shape = observation_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.emb_dim = emb_dim
        self.patch_size = patch_size

        if self.patch_size == 1:
            self.projection = nn.Sequential(
                nn.Linear(self.emb_dim, self.depth * 32),
                nn.Unflatten(1, (self.depth * 32, 1)),
                nn.Unflatten(2, (1, 1)),
            )
        else:
            self.projection = nn.Conv2d(self.emb_dim, self.depth * 32, kernel_size=1)

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                self.depth * 32,
                self.depth * 8,
                self.kernel_size,
                self.stride,
                padding=1,
            ),
            nn.ConvTranspose2d(
                self.depth * 8, self.depth * 4, self.kernel_size, self.stride, padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 4, self.depth * 2, self.kernel_size, self.stride, padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 2, self.depth * 1, self.kernel_size, self.stride, padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 1,
                self.observation_shape[0],
                self.kernel_size,
                self.stride,
                padding=1,
            ),
            nn.Upsample(
                size=(observation_shape[1], observation_shape[2]),
                mode="bilinear",
                align_corners=False,
            ),
        )
        self.network.apply(initialize_weights)

    def forward(self, posterior):
        if self.patch_size > 1:
            posterior = posterior.reshape(
                -1, self.patch_size, self.patch_size, self.emb_dim
            )  # (B, patch_size, patch_size, emb_size)
            posterior = posterior.permute(
                0, 3, 1, 2
            )  # (B, emb_size, patch_size, patch_size)

        x = self.projection(posterior)  # (B, depth * 32, patch, patch_size)
        x = self.network(x)

        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        img = dist.mean
        return img
