import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T  # noqa: F401

from flowdiffusion.encoder import DinoV2Encoder  # noqa: E402

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import argparse

from omegaconf import DictConfig
from tqdm import tqdm

from lorel.expert_dataset import ImageDataset  # noqa: E402


def main(args):
    target_size = (64, 64)

    data_folder = "../data_features/lorel"
    data_folder = Path(data_folder)

    cfg = DictConfig(
        {
            "root": "/home/grislain/SkillDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_1k.pkl",  # "/lustre/fsn1/projects/rech/fch/uxv44vw/TrajectoryDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_50k.pkl",
            "num_data": 10,  # 38225,
        },
    )

    if os.path.exists(data_folder):
        if not args.override:
            raise ValueError(
                f"Results folder {data_folder} already exists. Use --override to overwrite."
            )
    data_folder.mkdir(exist_ok=True, parents=True)

    data_folder = Path(data_folder)

    train_set = ImageDataset(cfg.root, num_trajectories=cfg.num_data, use_state=False)

    print("Train data:", len(train_set))

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Frozen encoder model
    if args.features == "dino":
        encoder_model = DinoV2Encoder()
    else:
        raise ValueError(f"Unknown feature type {args.features}")

    all_emb = {"cls_emb": [], "patch_emb": []}
    for image in tqdm(
        train_loader, desc=f"Generate {args.features} features of training data"
    ):
        cls_emb, patch_emb = encoder_model(image)

        all_emb["cls_emb"].append(cls_emb)
        all_emb["patch_emb"].append(patch_emb)

    # Save features
    torch.save(
        all_emb,
        data_folder / f"training/{args.features}_features.pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    parser.add_argument("-f", "--features", type=str, default="dino")
    args = parser.parse_args()
    main(args)
