import argparse
import os
import sys
from pathlib import Path

import numpy as np

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

import torch
from omegaconf import DictConfig, OmegaConf
from utils import assert_configs_equal

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
    results_folder = args.results_folder
    data_path = args.data_path

    if args.train_on == "lang":
        dataset_name = "lang_dataset"
        dataset_key = "lang"
    elif args.train_on == "vis":
        dataset_name = "vis_dataset"
        dataset_key = "vis"
    else:
        raise ValueError(f"Unknown dataset name {args.train_on}")
    print(f"Training on {dataset_name} dataset")

    if args.diffuse_on == "dino_vit":
        diffuse_on = f"dino_vit_{args.feat_patch_size}"
    else:
        diffuse_on = args.diffuse_on

    cfg = DictConfig(
        {
            "root": data_path,
            "datamodule": {
                dataset_name: {
                    "_target_": "calvin_agent.datasets.disk_dataset.DiskActionDataset",
                    "key": dataset_key,
                    "save_format": "npz",
                    "batch_size": 32,
                    "min_window_size": 32,
                    "max_window_size": 65,
                    "proprio_state": {
                        "n_state_obs": 8,
                        "keep_indices": [[0, 7], [14, 15]],
                        "robot_orientation_idx": [3, 6],
                        "normalize": True,
                        "normalize_robot_orientation": True,
                    },
                    "obs_space": {
                        "rgb_obs": ["rgb_static"]
                        if not args.use_ego_obs
                        else ["rgb_static", "rgb_gripper"],
                        "depth_obs": ["depth_static"] if args.use_depth else [],
                        "state_obs": ["robot_obs"],
                        "actions": ["actions"],
                        "language": ["language"],
                    },
                    "num_subgoals": args.num_subgoals,
                    "pad": True,
                    "lang_folder": "lang_annotations",
                    "num_workers": 2,
                    "diffuse_on": diffuse_on,
                    "norm_dino_feat": args.norm,
                    "prob_aug": args.data_aug_prob,
                    "feat_patch_size": args.feat_patch_size,
                },
            },
            "training_steps": 250000,  # In gradient steps
            "save_every": 100,  # In gradient steps
        }
    )

    print("Config:\n" + OmegaConf.to_yaml(cfg))

    n_channels = 4 if args.use_depth else 3

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

    train_set = data_module.train_datasets[dataset_key]
    valid_set = data_module.val_datasets[dataset_key]

    print("Train data:", len(train_set))
    print("Valid data:", len(valid_set))

    training_steps = cfg.training_steps
    device = torch.device("cuda")
    log_freq = 10

    if args.diffuse_on == "pixel":
        n_obs_steps = len(cfg.datamodule[dataset_name]["obs_space"]["rgb_obs"]) + 1
        image_shape = [n_channels, 96, 96]
    elif "dino" in args.diffuse_on:
        n_obs_steps = 2
        image_shape = [768, args.feat_patch_size, args.feat_patch_size]

    diff_cfg = DictConfig(
        {
            "n_obs_steps": n_obs_steps,
            "horizon": int(
                np.ceil(
                    cfg.datamodule[dataset_name]["max_window_size"]
                    / cfg.datamodule[dataset_name]["num_subgoals"]
                )
            )
            - 1,
            "input_shapes": {
                "observation.image": image_shape,
                "observation.state": [0],
            },
            "output_shapes": {
                "action": [7],
            },
            "n_action_steps": 8,
            "input_normalization_modes": {},
            "output_normalization_modes": {"action": "min_max"},
            "crop_shape": None,
        }
    )
    diff_cfg = DiffusionConfig(**diff_cfg)
    cfg["diff_cfg"] = diff_cfg

    # Load training statistics
    stats_path = os.path.join(data_path, "training/statistics.yaml")
    train_stats = OmegaConf.load(stats_path)

    train_stats_dict = {
        "action": {
            "max": torch.Tensor(train_stats.act_max_bound),
            "min": torch.Tensor(train_stats.act_min_bound),
        }
    }

    cfg["stats_path"] = stats_path
    # Save cfg
    if args.checkpoint_num is not None:
        # Load checkpoint config which is a yaml
        checkpoint_cfg_path = os.path.join(results_folder, "data_config.yaml")
        checkpoint_cfg = OmegaConf.load(checkpoint_cfg_path)

        assert assert_configs_equal(checkpoint_cfg, cfg, ["training_steps"])
    else:
        with open(os.path.join(results_folder, "data_config.yaml"), "w") as file:
            file.write(OmegaConf.to_yaml(cfg))

    policy = DiffusionPolicy(diff_cfg, dataset_stats=train_stats_dict)

    if args.checkpoint_num is not None:
        checkpoint_path = os.path.join(
            results_folder, f"model-{args.checkpoint_num}.pt"
        )
        print(f"Loading checkpoint from {checkpoint_path}")
        policy.load_state_dict(torch.load(checkpoint_path))

    policy.train()
    policy.to(device)

    # Training parameters
    print("Number of training parameters:", sum(p.numel() for p in policy.parameters()))

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Create dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        train_set,
        num_workers=4,
        batch_size=cfg.datamodule[dataset_name]["batch_size"],
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    # Run training loop.
    if args.checkpoint_num is not None:
        step = 0
    else:
        step = args.checkpoint_num
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
    parser.add_argument(
        "--training_steps", type=int, default=150000
    )  # set to number of training steps
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset",
    )  # set to path to dataset
    parser.add_argument(
        "-r", "--results_folder", type=str, default="../results_policy_single/calvin"
    )  # set to path to results folder
    parser.add_argument(
        "--num_subgoals", type=int, default=8
    )  # set to number of subgoals
    parser.add_argument(
        "--train_on", type=str, default="lang"
    )  # set to train on language labelled dataset (38% "lang") or full dataset (100% "vis")
    parser.add_argument(
        "--data_aug_prob", type=float, default=0.0
    )  # set to probability of data augmentation (0.0 for no augmentation)
    parser.add_argument(
        "--use_ego_obs", type=bool, default=False
    )  # set to True to use ego observations
    parser.add_argument(
        "--use_depth", type=bool, default=False
    )  # set to True to use depth observations
    parser.add_argument(
        "--diffuse_on", type=str, default="pixel"
    )  # set to diffuse on pixel or dino features
    parser.add_argument(
        "--feat_patch_size", type=int, default=16
    )  # set to feature patch size for dino features
    parser.add_argument(
        "--norm", type=str, default=None, choices=[None, "l2", "z_score", "min_max"]
    )  # set to normalisation type for features
    args = parser.parse_args()
    main(args)
