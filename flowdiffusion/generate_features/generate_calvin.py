import argparse
import os
import pickle
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
            "root": "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset",  # "/lustre/fsn1/projects/rech/fch/uxv44vw/CALVIN/task_D_D",
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
                    "skip_frames": 1,
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
            "calvin/calvin_models/conf/datamodule/transforms/play_decoder.yaml",
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

    print("Train data:", len(train_set))
    print("Valid data:", len(valid_set))

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
        encoder_model = DinoV2Encoder(
            name="facebook/dinov2-base",  # "/lustre/fsn1/projects/rech/fch/uxv44vw/facebook/dinov2-base"
        )
    else:
        raise ValueError(f"Unknown feature type {args.features}")

    all_emb = {"cls_emb": [], "patch_emb": [], "frame_idx": []}
    for data in tqdm(
        train_loader, desc=f"Generate {args.features} features of training data"
    ):
        frame_idx, image = data
        cls_emb, patch_emb = encoder_model(image)

        all_emb["cls_emb"].append(np.array(cls_emb.cpu()))
        all_emb["patch_emb"].append(np.array(patch_emb.cpu()))
        all_emb["frame_idx"].append(frame_idx.cpu().item())

    # Save as pickle
    with open(
        Path(cfg.root) / f"training/features/{args.features}_features.pkl", "wb"
    ) as f:
        pickle.dump(all_emb, f)

    all_eval_emb = {"cls_emb": [], "patch_emb": [], "frame_idx": []}
    for data in tqdm(
        valid_loader, desc=f"Generate {args.features} features of validation data"
    ):
        frame_idx, image = data
        cls_emb, patch_emb = encoder_model(image)
        all_eval_emb["cls_emb"].append(np.array(cls_emb.cpu()))
        all_eval_emb["patch_emb"].append(np.array(patch_emb.cpu()))
        all_eval_emb["frame_idx"].append(frame_idx.cpu().item())

    with open(
        Path(cfg.root) / f"validation/features/{args.features}_features.pkl", "wb"
    ) as f:
        pickle.dump(all_eval_emb, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    parser.add_argument("-f", "--features", type=str, default="dino")
    args = parser.parse_args()
    main(args)
