import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

from torch.utils.data import Subset

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from lorel.expert_dataset import ExpertActionDataset, ExpertDataset  # noqa: E402, F401

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Total GPUs available: {torch.cuda.device_count()}")


def main(args):
    valid_n = 1

    results_folder = "../results_policy/lorel"

    if args.server == "jz":
        data_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/TrajectoryDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_50k/dec_24_sawyer_50k.pkl"
    else:
        data_path = "/home/grislain/SkillDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_1k/data_with_dino_features"

    cfg = DictConfig(
        {
            "root": data_path,
            "skip_frames": 2,
            "diffuse_on": "pixel",
            "num_data": 100,  # 38225,
        },
    )

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

    train_set = ExpertActionDataset(
        cfg.root, skip_frames=cfg.skip_frames, diffuse_on=cfg.diffuse_on
    )

    # Split train and valid
    valid_inds = [i for i in range(0, len(train_set), len(train_set) // valid_n)][
        :valid_n
    ]
    valid_set = Subset(train_set, valid_inds)

    # Remove valide from train
    all_inds = set(range(len(train_set)))
    train_inds = list(all_inds - set(valid_inds))
    train_set = Subset(train_set, train_inds)

    print("Train data:", len(train_set))
    print("Valid data:", len(valid_set))

    training_steps = 5000  # TO BE MODIFIED
    device = torch.device("cuda")
    log_freq = 10

    diff_cfg = DictConfig(
        {
            "n_obs_steps": 2,
            "horizon": 8,
            "input_shapes": {
                "observation.image": [3, 96, 96],
                "observation.state": [0],
            },
            "output_shapes": {
                "action": [7],
            },
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
