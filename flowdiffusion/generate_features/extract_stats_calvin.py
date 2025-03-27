import argparse
import os
import sys

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


sys.path.append(
    os.path.join(
        root_path,
        "calvin/calvin_models",
    )
)

from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,  # noqa: E402
)

# Print number of GPUs available
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_stats(args):
    if args.server == "jz":
        data_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/CALVIN/task_D_D"
    else:
        data_path = "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset"

    cfg = DictConfig(
        {
            "root": data_path,
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
                    "diffuse_on": "dino_feat",
                    "norm_dino_feat": False,
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

    data_module = CalvinDataModule(
        cfg.datamodule, transforms=transforms, root_data_dir=cfg.root
    )

    data_module.setup()

    train_set = data_module.train_datasets["lang"]
    valid_set = data_module.val_datasets["lang"]

    print("Train data:", len(train_set))
    print("Valid data:", len(valid_set))

    sum = torch.zeros((256, 768))
    sum_squared = torch.zeros((256, 768))
    min = torch.ones((256, 768)) * 1e10
    max = torch.ones((256, 768)) * -1e10

    for data in tqdm(train_set, desc="Computing stats on train set"):
        _, patch_emb = data
        min = torch.min(min, patch_emb)
        max = torch.max(max, patch_emb)
        sum += patch_emb
        sum_squared += patch_emb**2

    for data in tqdm(valid_set, desc="Computing stats on valid set"):
        _, patch_emb = data
        min = torch.min(min, patch_emb)
        max = torch.max(max, patch_emb)
        sum += patch_emb
        sum_squared += patch_emb**2

    num_data = len(train_set) + len(valid_set)

    stats = {"dino_features": {}}
    stats["dino_features"]["mean"] = sum / num_data
    stats["dino_features"]["std"] = torch.sqrt(
        sum_squared / num_data - stats["dino_features"]["mean"] ** 2
    )
    stats["dino_features"]["min"] = min
    stats["dino_features"]["max"] = max

    # Save in root directory with pickle
    torch.save(stats, os.path.join(cfg.root, "dino_stats.pt"))
    print("Stats saved in", os.path.join(cfg.root, "dino_stats.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--server", type=str, default="hacienda"
    )  # set to "jz" to run on jean zay server
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    args = parser.parse_args()
    extract_stats(args)
