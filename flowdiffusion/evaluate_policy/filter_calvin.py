import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import hydra
import numpy as np
import PIL.Image as Image
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)
from utils import save_images
from vis_features import pca_project_features

sys.path.append(
    os.path.join(
        root_path,
        "calvin/calvin_models",
    )
)
import torchvision

from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,  # noqa: E402
)
from calvin.calvin_models.calvin_agent.models.calvin_base_model import CalvinBaseModel

logger = logging.getLogger(__name__)


class CustomModel(CalvinBaseModel):
    def __init__(self, cfg):
        self.reset()

    def reset(self):
        """
        This is called
        """
        self.steps = 0

    def save_image(self, image, name):
        saving_path = Path(self.debug_path) / name
        if image.shape[1] > 8:
            image = rearrange(image, "f c h w -> f (h w) c")
            image = pca_project_features(image.to(self.device).detach())
        else:
            image = (image + 1) / 2
        save_images(image, saving_path, nrow=image.shape[0])

    def step(self, expert):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        action = expert["actions"][self.steps]
        self.steps += 1
        return action


def evaluate_policy_singlestep(model, env, dataset, args, checkpoint):
    if args.server == "jz":
        conf_dir = Path(
            "/lustre/fswork/projects/rech/fch/uxv44vw/clemgris/avdc/calvin/calvin_models/conf"
        )
    else:
        conf_dir = Path("/home/grislain/AVDC/calvin/calvin_models/conf")

    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)

    results = Counter()
    tot_tasks = Counter()

    auto_lang_ann_filtered = {
        "info": {"episodes": [], "indx": []},
        "language": {"ann": [], "task": []},
    }
    all_sr_lang_ann = {
        "info": {"episodes": [], "indx": [], "sr": []},
        "language": {"ann": [], "task": []},
    }
    num_trial = 5
    prob_sucess = 0.8
    for episode, task, ann, (start_idx, end_idx) in dataset:
        sucesses = 0
        for __ in range(num_trial):
            success, length = rollout(env, model, episode, task_oracle, args, task)
            sucesses += success

        mean_sucess = sucesses / num_trial
        if mean_sucess >= prob_sucess:
            auto_lang_ann_filtered["info"]["indx"].append((start_idx, end_idx))
            auto_lang_ann_filtered["language"]["ann"].append(ann)
            auto_lang_ann_filtered["language"]["task"].append(task)

        all_sr_lang_ann["info"]["indx"].append((start_idx, end_idx))
        all_sr_lang_ann["language"]["ann"].append(ann)
        all_sr_lang_ann["language"]["task"].append(task)
        all_sr_lang_ann["info"]["sr"].append(mean_sucess)

        results[task] += mean_sucess >= prob_sucess
        tot_tasks[task] += 1
        print(f"{task}: {results[task]} / {tot_tasks[task]} ({length})")

    print("\nResults successful expert demonstrations\n" + "-" * 60)
    for task in results:
        print(f"{task}: {results[task]} / {tot_tasks[task]}")

    print(f"SR: {sum(results.values()) / sum(tot_tasks.values()) * 100:.1f}%")

    # Save filtered language annotations
    print(
        f"Saving filtered language annotations : {len(auto_lang_ann_filtered['info']['indx'])}/{len(dataset)}"
    )
    saving_path = (
        dataset.abs_datasets_dir / dataset.lang_folder / "filtered_auto_lang_ann.npy"
    )
    np.save(
        saving_path,
        auto_lang_ann_filtered,
        allow_pickle=True,
    )

    # Save all language annotations with success rate
    print(
        f"Saving all language annotations with success rate : {len(all_sr_lang_ann['info']['indx'])}/{len(dataset)}"
    )
    saving_path = dataset.abs_datasets_dir / dataset.lang_folder / "all_sr_lang_ann.npy"
    np.save(
        saving_path,
        all_sr_lang_ann,
        allow_pickle=True,
    )

    # Save results
    with open(
        os.path.join(
            dataset.abs_datasets_dir / dataset.lang_folder,
            f"results_filtering_{args.test_on}.txt",
        ),
        "w",
    ) as f:
        for task in results:
            f.write(f"{task}: {results[task]} / {tot_tasks[task]}\n")
        f.write(f"SR: {sum(results.values()) / sum(tot_tasks.values()) * 100:.1f}%\n")


def save_gif(obs_list, save_path, duration=0.2):
    frames = []

    for img in obs_list:
        # Convert from CHW torch tensor to HWC numpy array
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()

        # Normalize from [-1, 1] to [0, 255]
        img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        # Convert to PIL Image in RGB mode
        frames.append(Image.fromarray(img_np).convert("RGB"))

    # Save GIF with optimized settings
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration * 100),  # duration in milliseconds
        loop=0,
        optimize=True,
        quality=95,
        disposal=2,  # Replace previous frame
    )


def rollout(env, model, episode, task_oracle, args, task):
    # state_obs, rgb_obs, depth_obs = episode["robot_obs"], episode["rgb_obs"], episode["depth_obs"]
    reset_info = episode["state_info"]
    # idx = episode["idx"]
    obs = env.reset(
        robot_obs=reset_info["robot_obs"][0], scene_obs=reset_info["scene_obs"][0]
    )

    model.reset()
    start_info = env.get_info()
    obs_list = []

    for step in range(args.ep_len):
        if step > episode["actions"].shape[0] - 1:
            break
        # action = episode["actions"][step]
        action = model.step(episode)
        obs, _, _, current_info = env.step(action)
        if args.save_failures:
            obs_list.append(obs["rgb_obs"]["rgb_static"][0, 0])
        # Check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {task.replace(" ", "_")}
        )
        if len(current_task_info) > 0:
            print(colored("S", "green"), end=" ")
            return True, step
    print(colored("F", "red"), end=" ")
    if args.save_failures:
        if args.save_failures:
            # Create folder for this failed episode
            os.makedirs(args.debug_path, exist_ok=True)
            failed_episode_path = os.path.join(
                args.debug_path, f"failed_{task.replace(' ', '_')}_{episode['idx']}"
            )
            os.makedirs(
                failed_episode_path,
                exist_ok=True,
            )

        # Save episode (as png)
        torchvision.utils.save_image(
            (torch.stack(obs_list) + 1) / 2,
            os.path.join(
                failed_episode_path,
                "trajectory.png",
            ),
        )
        # Save episode (as gif)
        save_gif(
            obs_list, os.path.join(failed_episode_path, "trajectory.gif"), duration=1.0
        )

    return False, step


if __name__ == "__main__":
    seed_everything(0, workers=True)
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )

    parser.add_argument(
        "--test_on",
        type=str,
        help="Train on train or val",
        default="train",
    )

    parser.add_argument(
        "--server",
        "-s",
        type=str,
        help="Server",
        default="hacienda",
    )

    parser.add_argument(
        "--debug_path",
        type=str,
        default=None,
        help="Path to save debug images.",
    )

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()
    args.save_failures = args.debug_path is not None

    # Do not change
    args.ep_len = 240

    if args.server == "jz":
        data_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/CALVIN/task_D_D_jz"
        rollout_cfg_path = "/lustre/fswork/projects/rech/fch/uxv44vw/clemgris/avdc/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
    elif args.server == "hacienda":
        data_path = "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset"
        rollout_cfg_path = "/home/grislain/AVDC/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
    else:
        raise ValueError("Invalid server argument")

    config = DictConfig({"debug_path": args.debug_path, "device": args.device})

    transforms_dict = OmegaConf.load(
        os.path.join(
            root_path,
            "calvin/calvin_models/conf/datamodule/transforms/play_basic.yaml",
        )
    )

    cfg = DictConfig(
        {
            "root": data_path,
            "datamodule": {
                "lang_dataset": {
                    "_target_": "calvin_agent.datasets.disk_dataset.DiskFilterDataset",
                    "key": "lang",
                    "save_format": "npz",
                    "batch_size": 1,
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
                        "rgb_obs": ["rgb_static"],
                        "depth_obs": (["depth_static"]),
                        "state_obs": ["robot_obs"],
                        "actions": ["actions"],
                        "language": ["language"],
                    },
                    "num_subgoals": 8,
                    "pad": True,
                    "lang_folder": "lang_annotations",
                    "num_workers": 2,
                    "diffuse_on": "pixel",
                    "norm_feat": None,
                    "feat_patch_size": 16,
                },
            },
            "training_steps": 1,  # In gradient steps
            "save_every": 100,  # In gradient steps
        }
    )

    data_module = CalvinDataModule(
        cfg.datamodule,
        transforms=transforms_dict,
        root_data_dir=data_path,
    )
    data_module.setup()

    if args.test_on == "train":
        dataloader = data_module.train_dataloader()
        policy_dataset = dataloader["lang"].dataset
    elif args.test_on == "val":
        dataloader = data_module.val_dataloader()
        policy_dataset = dataloader.dataset.datasets["lang"]
    else:
        raise ValueError("Invalid test_on argument")

    device = torch.device("cuda:0")
    config.device = "cuda"
    checkpoint = "None"

    if config.debug_path is not None:
        # Create debug folder
        debug_path = Path(config.debug_path)
        os.makedirs(debug_path, exist_ok=True)

    rollout_cfg = OmegaConf.load(rollout_cfg_path)
    env = hydra.utils.instantiate(
        rollout_cfg.env_cfg, policy_dataset, device, show_gui=False
    )
    model = CustomModel(config)

    evaluate_policy_singlestep(model, env, policy_dataset, args, checkpoint)
