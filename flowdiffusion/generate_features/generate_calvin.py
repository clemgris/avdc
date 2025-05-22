import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T  # noqa: F401
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(root_path)
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)
from encoder import DinoV2Encoder, R3MEncoder, ViTEncoder  # noqa: E402

sys.path.append(
    os.path.join(
        root_path,
        "calvin/calvin_models",
    )
)

from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,  # noqa: E402
)


def main(args):
    cfg = DictConfig(
        {
            "root": args.root,
            "datamodule": {
                "vis_dataset": {
                    "_target_": "calvin_agent.datasets.disk_dataset.DiskImageDataset",
                    "key": "vis",
                    "save_format": "npz",
                    "batch_size": 32,
                    "min_window_size": 16,
                    "max_window_size": 64,
                    "proprio_state": {
                        "n_state_obs": 8,
                        "keep_indices": [[0, 7], [14, 15]],
                        "robot_orientation_idx": [3, 6],
                        "normalize": True,
                        "normalize_robot_orientation": True,
                    },
                    "obs_space": {
                        "rgb_obs": ["rgb_static"],  # ["rgb_gripper"]
                        "depth_obs": [],
                        "state_obs": ["robot_obs"],
                        "actions": ["actions"],
                        "language": ["language"],
                    },
                    "pad": False,
                    "lang_folder": "lang_annotations",
                    "num_workers": 2,
                },
            },
            "patch_size": args.patch_size,
        }
    )

    if args.features == "dino":
        assert args.patch_size % 16 == 0, "Dino patch size must be a multiple of 16"
        print("Using dino features with transform from play_nothing.yaml")
        transforms = OmegaConf.load(
            os.path.join(
                root_path,
                "calvin/calvin_models/conf/datamodule/transforms/play_nothing.yaml",
            )
        )
    elif args.features == "dino_vit":
        assert args.patch_size % 16 == 0, "Dino patch size must be a multiple of 16"
        print("Using dino_vit features with transform from play_features_imagenet.yaml")
        transforms = OmegaConf.load(
            os.path.join(
                root_path,
                "calvin/calvin_models/conf/datamodule/transforms/play_features_imagenet.yaml",
            )
        )
    elif args.features == "r3m":
        assert args.patch_size % 7 == 0, "Dino patch size must be a multiple of 7"
        print("Using r3m features with transform from play_features_imagenet.yaml")
        transforms = OmegaConf.load(
            os.path.join(
                root_path,
                "calvin/calvin_models/conf/datamodule/transforms/play_features_imagenet.yaml",
            )
        )

    if "dino" in args.features:
        for section in ["train", "val"]:
            if hasattr(transforms, section) and hasattr(
                transforms[section], "rgb_static"
            ):
                for transform in transforms[section].rgb_static:
                    if (
                        "_target_" in transform
                        and transform["_target_"] == "torchvision.transforms.Resize"
                    ):
                        print(f"Resize {section} to", args.patch_size * 14)
                        transform["size"] = args.patch_size * 14
                    elif (
                        "_target_" in transform
                        and transform["_target_"] == "torchvision.transforms.CenterCrop"
                    ):
                        print(f"CenterCrop {section} to", args.patch_size * 14)
                        transform["size"] = args.patch_size * 14
    elif args.features == "r3m":
        for section in ["train", "val"]:
            if hasattr(transforms, section) and hasattr(
                transforms[section], "rgb_static"
            ):
                for transform in transforms[section].rgb_static:
                    if (
                        "_target_" in transform
                        and transform["_target_"] == "torchvision.transforms.Resize"
                    ):
                        print(f"Resize {section} to", args.patch_size * 32)
                        transform["size"] = args.patch_size * 32
                    elif (
                        "_target_" in transform
                        and transform["_target_"] == "torchvision.transforms.CenterCrop"
                    ):
                        print(f"CenterCrop {section} to", args.patch_size * 32)
                        transform["size"] = args.patch_size * 32

    data_module = CalvinDataModule(
        cfg.datamodule, transforms=transforms, root_data_dir=cfg.root
    )

    data_module.setup()

    train_set = data_module.train_datasets["vis"]
    valid_set = data_module.val_datasets["vis"]

    print("Train data (img):", len(train_set))
    print("Valid data (img):", len(valid_set))

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Frozen encoder model
    if args.features == "dino":
        num_channels = 768
        if args.server == "hacienda":
            encoder_model = DinoV2Encoder(
                name="facebook/dinov2-base",
            )
        elif args.server == "jz":
            encoder_model = DinoV2Encoder(
                name="/lustre/fsn1/projects/rech/fch/uxv44vw/facebook/dinov2-base",
            )
        else:
            raise ValueError(f"Unknown server {args.server}")
    elif args.features == "dino_vit":
        num_channels = 768
        encoder_model = ViTEncoder()
    elif args.features == "r3m":
        num_channels = 512
        encoder_model = R3MEncoder("resnet18")
    else:
        raise ValueError(f"Unknown feature type {args.features}")

    # Create storing directories
    os.makedirs(
        os.path.join(cfg.root, f"training/features_{args.features}_{args.patch_size}"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(
            cfg.root, f"validation/features_{args.features}_{args.patch_size}"
        ),
        exist_ok=True,
    )
    with open(
        os.path.join(cfg.root, f"{args.features}_{args.patch_size}_config.yaml"), "w"
    ) as file:
        file.write(OmegaConf.to_yaml(cfg))

    # Init stats
    sum = torch.zeros((args.patch_size**2, num_channels))
    sum_squared = torch.zeros((args.patch_size**2, num_channels))
    min = torch.ones((args.patch_size**2, num_channels)) * 1e10
    max = torch.ones((args.patch_size**2, num_channels)) * -1e10

    for data in tqdm(
        train_loader, desc=f"Generate {args.features} features of training data"
    ):
        frame_idx, image, _ = data
        _, patch_emb = encoder_model(image.to("cuda"))
        patch_emb = patch_emb.detach().cpu()
        for i in range(len(frame_idx)):
            all_emb = {}
            all_emb["patch_emb"] = patch_emb[i].cpu().numpy()
            all_emb["frame_idx"] = frame_idx[i].cpu().item()

            # Save as npz
            np.savez(
                Path(cfg.root)
                / f"training/features_{args.features}_{args.patch_size}/features_{all_emb['frame_idx']}.npz",
                **all_emb,
            )

            # Update stats
            min = torch.min(min, patch_emb[i])
            max = torch.max(max, patch_emb[i])
            sum += patch_emb[i]
            sum_squared += patch_emb[i] ** 2

    print(
        "Training features saved in ",
        cfg.root + f"/training/features_{args.features}_{args.patch_size}",
    )

    # Save stats
    num_data = len(train_loader)
    stats = {"dino_features": {}}
    stats["dino_features"]["mean"] = sum / num_data
    stats["dino_features"]["std"] = torch.sqrt(
        sum_squared / num_data - stats["dino_features"]["mean"] ** 2
    )
    stats["dino_features"]["min"] = min
    stats["dino_features"]["max"] = max

    # Save in root directory with pickle
    torch.save(
        stats, os.path.join(cfg.root, f"{args.features}_{args.patch_size}_stats.pt")
    )
    print(
        "Stats saved in",
        os.path.join(cfg.root, f"{args.features}_{args.patch_size}_stats.pt"),
    )

    for data in tqdm(
        valid_loader, desc=f"Generate {args.features} features of validation data"
    ):
        frame_idx, image, _ = data
        _, patch_emb = encoder_model(image.to("cuda"))
        patch_emb = patch_emb.detach().cpu()

        for i in range(len(frame_idx)):
            eval_emb = {}
            eval_emb["patch_emb"] = patch_emb[i].cpu().numpy()
            eval_emb["frame_idx"] = frame_idx[i].cpu().item()

            # save as npz
            np.savez(
                Path(cfg.root)
                / f"validation/features_{args.features}_{args.patch_size}/features_{eval_emb['frame_idx']}.npz",
                **eval_emb,
            )

    print(
        "Validation features saved in ",
        cfg.root + f"/validation/features_{args.features}_{args.patch_size}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        default="/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset",
    )
    parser.add_argument(
        "-s", "--server", type=str, default="hacienda"
    )  # hacienda or jz
    parser.add_argument(
        "-b", "--batch_size", type=int, default=32
    )  # batch size for dataloader
    parser.add_argument("-f", "--features", type=str, default="dino_vit")
    parser.add_argument(
        "-p", "--patch_size", type=int, default=16
    )  # patch size for encoder model
    args = parser.parse_args()
    main(args)
