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

from encoder import DinoV2Encoder  # noqa: E402

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
                "lang_dataset": {
                    "_target_": "calvin_agent.datasets.disk_dataset.DiskImageDataset",
                    "key": "lang",
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
        }
    )

    transforms = OmegaConf.load(
        os.path.join(
            root_path,
            "calvin/calvin_models/conf/datamodule/transforms/play_features_extract.yaml",
        )
    )

    os.makedirs(os.path.join(cfg.root, "training/features"), exist_ok=True)
    os.makedirs(os.path.join(cfg.root, "validation/features"), exist_ok=True)
    with open(os.path.join(cfg.root, "dino_feat_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(cfg))

    data_module = CalvinDataModule(
        cfg.datamodule, transforms=transforms, root_data_dir=cfg.root
    )

    data_module.setup()

    train_set = data_module.train_datasets["lang"]
    valid_set = data_module.val_datasets["lang"]

    print("Train data (img):", len(train_set))
    print("Valid data (img):", len(valid_set))

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Frozen encoder model

    if args.features == "dino":
        if args.server == "hacienda":
            encoder_model = DinoV2Encoder(
                name="facebook/dinov2-base",
            )
        elif args.server == "jz":
            encoder_model = DinoV2Encoder(
                name="/lustre/fsmisc/dataset/HuggingFace_Models/facebook/dinov2-base",
            )
        else:
            raise ValueError(f"Unknown server {args.server}")
    else:
        raise ValueError(f"Unknown feature type {args.features}")

    for data in tqdm(
        train_loader, desc=f"Generate {args.features} features of training data"
    ):
        all_emb = {}
        frame_idx, image, _ = data
        cls_emb, patch_emb = encoder_model(image)

        all_emb["cls_emb"] = np.array(cls_emb.cpu())
        all_emb["patch_emb"] = np.array(patch_emb.cpu())
        all_emb["frame_idx"] = frame_idx.cpu().item()

        # Save as npz
        np.savez(
            Path(cfg.root)
            / f"training/features/{args.features}_features_{all_emb['frame_idx']}.npz",
            **all_emb,
        )

    print("Training features saved in ", cfg.root + "/training/features")

    for data in tqdm(
        valid_loader, desc=f"Generate {args.features} features of validation data"
    ):
        eval_emb = {}
        frame_idx, image, _ = data
        cls_emb, patch_emb = encoder_model(image)
        eval_emb["cls_emb"] = np.array(cls_emb.cpu())
        eval_emb["patch_emb"] = np.array(patch_emb.cpu())
        eval_emb["frame_idx"] = frame_idx.cpu().item()

        # save as npz
        np.savez(
            Path(cfg.root)
            / f"validation/features/{args.features}_features_{eval_emb['frame_idx']}.npz",
            **eval_emb,
        )

    print("Validation features saved in ", cfg.root + "/validation/features")


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
    parser.add_argument("-f", "--features", type=str, default="dino")
    args = parser.parse_args()
    main(args)
