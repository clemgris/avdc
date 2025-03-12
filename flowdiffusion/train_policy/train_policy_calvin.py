import argparse
import os
import sys
from pathlib import Path

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
    valid_n = 1
    sample_per_seq = 8

    results_folder = "../results_11_03_dino/calvin"

    cfg = DictConfig(
        {
            "root": "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset",  # "/lustre/fsn1/projects/rech/fch/uxv44vw/CALVIN/task_D_D",
            "datamodule": {
                "lang_dataset": {
                    "_target_": "calvin_agent.datasets.disk_dataset.DiskActionDataset",
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
                    "skip_frames": 8,
                    "pad": True,
                    "lang_folder": "lang_annotations",
                    "num_workers": 2,
                    "diffuse_on": "pixel",
                },
            },
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
    if args.mode == "train":
        data_module.setup()
    results_folder = Path(results_folder)

    if args.mode == "train":
        if os.path.exists(results_folder):
            if not args.override:
                raise ValueError(
                    f"Results folder {results_folder} already exists. Use --override to overwrite."
                )
        results_folder.mkdir(exist_ok=True, parents=True)

    with open(os.path.join(results_folder, "data_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(cfg))

    if args.mode == "inference":
        train_set = valid_set = [None]  # dummy
    else:
        train_set = data_module.train_datasets["lang"]
        valid_set = data_module.val_datasets["lang"]

        print("Train data:", len(train_set))
        print("Valid data:", len(valid_set))

        ## DEBUG
        # breakpoint()
        # for idx in range(len(train_set)):
        #     start, actions, end = train_set[idx]
        #     torchvision.utils.save_image(
        #         start, f"train_start_{idx}.png"
        #     )
        #     torchvision.utils.save_image(
        #         end, f"train_end_{idx}.png"
        #     )
        #     if idx > 10: break

        # for idx in range(len(valid_set)):
        #     x, x_cond, task = valid_set[idx]
        #     torchvision.utils.save_image(
        #         start, f"valid_start_{idx}.png"
        #     )
        #     torchvision.utils.save_image(
        #         end, f"valid_end_{idx}.png"
        #     )
        #     if idx > 10: break
        # breakpoint()

    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/example_pusht_diffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 5000
    device = torch.device("cuda")
    log_freq = 250

    # Set up the the policy.
    # Policies are initialized with a configuration class, in this case `DiffusionConfig`.
    # For this example, no arguments need to be passed because the defaults are set up for PushT.
    # If you're doing something different, you will likely need to change at least some of the defaults.
    diff_cfg = DiffusionConfig(
        n_obs_steps=2,
        horizon=8,
        input_shapes={
            "observation.image": [3, 96, 96],
            "observation.state": [0],
        },
        output_shapes={
            "action": [7],
        },
    )

    stats_path = os.path.join(cfg.root, "training/dataset_stats.pkl")
    if os.path.exists(stats_path):
        train_stats = pickle.load(open(stats_path, "rb"))
    else:
        # Create stats
        train_stats = {
            "observation.image": {
                "min": torch.tensor([-1.0] * 3, dtype=torch.float32)[:, None, None],
                "max": torch.tensor([1.0] * 3)[:, None, None],
                "mean": torch.tensor([0.0] * 3)[:, None, None],  # Do nothing
                "std": torch.tensor([1.0] * 3)[:, None, None],  # Do nothing
            },
            "observation.state": {
                "min": torch.tensor([0.0] * 2),
                "max": torch.tensor([0.0] * 2),
                "mean": torch.tensor([0.0] * 2),
                "std": torch.tensor([1.0] * 2),
            },
            "action": {},
        }
        all_actions = []
        for data in train_set:
            action = data["action"][0]
            all_actions.append(action)
        train_stats["action"]["mean"] = torch.mean(torch.stack(all_actions), dim=0)
        train_stats["action"]["std"] = torch.std(torch.stack(all_actions), dim=0)
        train_stats["action"]["min"] = torch.min(torch.stack(all_actions), dim=0).values
        train_stats["action"]["max"] = torch.max(torch.stack(all_actions), dim=0).values

        # Save stats
        pickle.dump(train_stats, open(stats_path, "wb"))

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
            # breakpoint()
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    parser.add_argument(
        "-num", "--num_samples", type=int, default=1
    )  # set to number of samples to generate
    parser.add_argument(
        "-m", "--mode", type=str, default="train", choices=["train", "inference"]
    )  # set to 'inference' to generate samples
    parser.add_argument(
        "-c", "--checkpoint_num", type=int, default=None
    )  # set to checkpoint number to resume training or generate samples
    parser.add_argument(
        "-p", "--inference_path", type=str, default=None
    )  # set to path to generate samples
    parser.add_argument(
        "-t", "--text", type=str, default=None
    )  # set to text to generate samples
    parser.add_argument(
        "-n", "--sample_steps", type=int, default=100
    )  # set to number of steps to sample
    parser.add_argument(
        "-g", "--guidance_weight", type=int, default=0
    )  # set to positive to use guidance
    args = parser.parse_args()
    if args.mode == "inference":
        assert args.checkpoint_num is not None
        assert args.inference_path is not None
        assert args.text is not None
        assert args.sample_steps <= 100
    main(args)
