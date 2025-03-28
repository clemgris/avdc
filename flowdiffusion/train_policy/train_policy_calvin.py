import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

import pickle

import torch
from omegaconf import DictConfig, OmegaConf

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

sys.path.append(
    os.path.join(
        root_path,
        "calvin/calvin_models",
    )
)

from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,  # noqa: E402
)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Total GPUs available: {torch.cuda.device_count()}")


def main(args):
    results_folder = "../results_policy_single/calvin"

    if args.server == "jz":
        data_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/CALVIN/task_D_D"
    else:
        data_path = "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset"

    cfg = DictConfig(
        {
            "root": data_path,
            "datamodule": {
                "lang_dataset": {
                    "_target_": "calvin_agent.datasets.disk_dataset.DiskActionDataset",
                    "key": "lang",
                    "save_format": "npz",
                    "batch_size": 32,
                    "min_window_size": 16,
                    "max_window_size": 65,
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
                    "num_subgoals": 1,
                    "pad": True,
                    "lang_folder": "lang_annotations",
                    "num_workers": 2,
                    "diffuse_on": "pixel",
                },
            },
            "save_every": 100,  # In gradient steps
        }
    )

    transforms = OmegaConf.load(
        os.path.join(
            root_path,
            "calvin/calvin_models/conf/datamodule/transforms/play_basic.yaml",
        )
    )

    data_module = CalvinDataModule(
        cfg.datamodule, transforms=transforms, root_data_dir=cfg.root
    )

    data_module.setup()
    results_folder = Path(results_folder)

    if os.path.exists(results_folder):
        if not args.override:
            raise ValueError(
                f"Results folder {results_folder} already exists. Use --override to overwrite."
            )
    results_folder.mkdir(exist_ok=True, parents=True)

    train_set = data_module.train_datasets["lang"]
    valid_set = data_module.val_datasets["lang"]

    print("Train data:", len(train_set))
    print("Valid data:", len(valid_set))

    training_steps = 60000  # TO BE MODIFIED
    device = torch.device("cuda")
    log_freq = 10

    diff_cfg = DictConfig(
        {
            "n_obs_steps": 2,
            "horizon": int(
                np.ceil(
                    cfg.datamodule.lang_dataset.max_window_size
                    / cfg.datamodule.lang_dataset.num_subgoals
                )
            )
            - 1,
            "input_shapes": {
                "observation.image": [3, 96, 96],
                "observation.state": [0],
            },
            "output_shapes": {
                "action": [7],
            },
            "n_action_steps": 8,
            "input_normalization_modes": {},
        }
    )
    diff_cfg = DiffusionConfig(**diff_cfg)

    cfg["diff_cfg"] = diff_cfg

    stats_path = os.path.join(cfg.root, "training/dataset_stats.pkl")
    if os.path.exists(stats_path):
        train_stats = pickle.load(open(stats_path, "rb"))
    else:
        # Create stats
        train_stats = {
            "action": {},
        }
        all_actions = []
        for data in tqdm(train_set, desc="Generating stats"):
            action = data["action"][0]
            all_actions.append(action)
        train_stats["action"]["mean"] = torch.mean(torch.stack(all_actions), dim=0)
        train_stats["action"]["std"] = torch.std(torch.stack(all_actions), dim=0)
        train_stats["action"]["min"] = torch.min(torch.stack(all_actions), dim=0).values
        train_stats["action"]["max"] = torch.max(torch.stack(all_actions), dim=0).values

        # Save stats
        pickle.dump(train_stats, open(stats_path, "wb"))

    cfg["stats_path"] = stats_path
    # Save cfg
    with open(os.path.join(results_folder, "data_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(cfg))

    policy = DiffusionPolicy(diff_cfg, dataset_stats=train_stats)
    policy.train()
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Create dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        train_set,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")

            if step % cfg.save_every == 0:
                # Delete previous checkpoints
                past_saving_path = os.path.join(
                    results_folder, f"model-{step // cfg.save_every - 2}.pt"
                )
                if os.path.exists(past_saving_path):
                    os.remove(past_saving_path)
                # Save model
                saving_path = os.path.join(
                    results_folder, f"model-{step // cfg.save_every}.pt"
                )
                torch.save(policy.state_dict(), saving_path)
            step += 1
            if step >= training_steps:
                done = True
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--server", type=str, default="hacienda"
    )  # set to 'jz' to run on jean zay server
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    parser.add_argument(
        "-c", "--checkpoint_num", type=int, default=None
    )  # set to checkpoint number to resume training or generate samples

    args = parser.parse_args()
    main(args)
