import os
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

from encoder import DinoV2Encoder  # noqa: E402

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import argparse

from lorel.expert_dataset import ImageDataset  # noqa: E402


def main(args):
    cfg = DictConfig(
        {
            "root": "/home/grislain/SkillDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_1k.pkl",  # "/lustre/fsn1/projects/rech/fch/uxv44vw/TrajectoryDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_50k.pkl",
            "num_data": 3,  # 38225,
        },
    )

    data_filename = Path(cfg.root).stem
    folder_name = Path(cfg.root).parent.name

    print("Data folder:", folder_name / data_filename)

    train_set = ImageDataset(cfg.root, num_trajectories=cfg.num_data, use_state=False)

    print("Train data:", len(train_set))

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Frozen encoder model
    if args.features == "dino":
        encoder_model = DinoV2Encoder(
            name="facebook/dinov2-base",  # "/lustre/fsn1/projects/rech/fch/uxv44vw/facebook/dinov2-base"
        )
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
        folder_name / data_filename / f"{args.features}_features.pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    parser.add_argument("-f", "--features", type=str, default="dino")
    args = parser.parse_args()
    main(args)
