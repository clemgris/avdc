import torch


class ScaleImageTensor(object):
    """Scale tensor of shape (batch, C, H, W) containing images to [0, 1] range

    Args:
        tensor (torch.tensor): Tensor to be scaled.
    Returns:
        Tensor: Scaled tensor.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor.float().div(255)


class MinMaxNormalize(object):
    """Normalize a tensor to [-1, 1] range."""

    def __init__(self, min=0.0, max=1.0):
        self.min = min
        self.max = max

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        norm_tensor = (tensor - self.min()) / (self.max() - self.min())
        norm_tensor = norm_tensor * 2 - 1
        return norm_tensor


class NormalizeVector(object):
    """Normalize a tensor vector with mean and standard deviation."""

    def __init__(self, mean=[0.0], std=[1.0]):
        self.std = torch.Tensor(std)
        self.std[self.std == 0.0] = 1.0
        self.mean = torch.Tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return (tensor - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = torch.tensor(std)
        self.mean = torch.tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class AddDepthNoise(object):
    """Add multiplicative gamma noise to depth image.
    This is adapted from the DexNet 2.0 code.
    Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/master/gqcnn/training/tf/trainer_tf.py"""

    def __init__(self, shape=1000.0, rate=1000.0):
        self.shape = torch.tensor(shape)
        self.rate = torch.tensor(rate)
        self.dist = torch.distributions.gamma.Gamma(
            torch.tensor(shape), torch.tensor(rate)
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        multiplicative_noise = self.dist.sample()
        return multiplicative_noise * tensor

    def __repr__(self):
        return self.__class__.__name__ + f"{self.shape=},{self.rate=},{self.dist=}"
