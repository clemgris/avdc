import os
import sys

import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, minmax_scale

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, "flowdiffusion"))

from encoder import ViTEncoder


def pca_project_features(patch_emb):
    """
    Projects patch features to 3D using PCA and returns them as an image-like array.

    Args:
        patch_emb (ndarray): Patch embeddings of shape [B, N, D]

    Returns:
        np.ndarray: PCA-projected feature map of shape [B, H, W, 3], values in [0, 1]
    """
    if isinstance(patch_emb, torch.Tensor):
        patch_emb = patch_emb.cpu()

    img_cnt = patch_emb.shape[0]
    patch_h = patch_w = int(np.sqrt(patch_emb.shape[1]))

    # Flatten and normalize
    flat_features = patch_emb.reshape(-1, patch_emb.shape[-1])
    flat_features = StandardScaler().fit_transform(flat_features)

    # PCA projection
    pca = PCA(n_components=3)
    projected = pca.fit_transform(flat_features)
    projected = minmax_scale(projected)

    pca_image = projected.reshape(img_cnt, patch_h, patch_w, 3)
    pca_image = torch.tensor(pca_image, dtype=torch.float32).permute(0, 3, 1, 2)
    return pca_image


def main(images, patch_emb, output_dir="pca_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Project features to PCA space
    fg_result = pca_project_features(patch_emb)

    # Save raw tensor
    torch.save(
        torch.tensor(fg_result, dtype=torch.float32),
        os.path.join(output_dir, "fg_result.pt"),
    )

    img_cnt = images.shape[0]

    for i in range(img_cnt):
        # Save PCA RGB image
        pca_rgb = (fg_result[i] * 255).astype(np.uint8)
        pca_img = Image.fromarray(pca_rgb)
        pca_img = pca_img.resize(
            (images.shape[3], images.shape[2]), resample=Image.NEAREST
        )
        pca_img.save(os.path.join(output_dir, f"pca_result_{i}.png"))

        # Optional overlay
        orig_img = T.ToPILImage()(images[i].cpu())
        overlay = Image.blend(orig_img, pca_img.convert("RGB"), alpha=0.5)
        overlay.save(os.path.join(output_dir, f"overlay_{i}.png"))

        print(f"Saved PCA image and overlay for image {i}")


if __name__ == "__main__":
    image_path = "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset/training/episode_0358483.npz"
    features_path = "/home/grislain/AVDC/flowdiffusion/features.pt"

    # Load raw image
    image = np.load(image_path)["rgb_static"]
    image = Image.fromarray(image.astype(np.uint8))

    # Load encoder
    encoder_model = ViTEncoder()
    image_size = 224 * 4

    # Image transform
    transforms = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    transformed_image = transforms(image)[None]
    _, features = encoder_model(transformed_image.to("cuda"))
    # features = F.normalize(features, dim=-1)
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    # Run PCA visualization
    main(transformed_image.cpu(), features)
