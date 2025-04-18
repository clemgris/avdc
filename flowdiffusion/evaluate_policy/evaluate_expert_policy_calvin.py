import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import hydra
import torch
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
        self.device = torch.device(cfg.device)
        self.cfg = cfg

        # Debug
        self.debug = cfg.debug
        self.debug_path = cfg.debug_path

    def reset(self):
        """
        This is called
        """
        self.steps = 0

    def save_image(self, image, name):
        saving_path = Path(self.debug_path) / name
        torchvision.utils.save_image((image + 1) / 2, saving_path)

    def step(self, obs, text_goal, oracle_subgoals=None):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        # Normalise obs
        obs_image = obs["rgb_obs"]["rgb_static"]
        # Save image
        if self.debug:
            self.save_image(
                obs_image[0],
                f"obs_{self.steps}.png",
            )

        self.actions = oracle_subgoals["actions"]
        action_idx = min(self.steps, len(self.actions) - 1)

        selected_action = self.actions[action_idx]
        self.steps += 1
        return selected_action


def evaluate_policy_singlestep(model, env, high_level_dataset, args, checkpoint):
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
    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml"
    )

    high_level_dataset = high_level_dataset

    results = Counter()
    tot_tasks = Counter()

    for episode in high_level_dataset:
        task = episode["task"]
        results[task] += rollout(
            env, model, episode, task_oracle, args, task, val_annotations
        )
        tot_tasks[task] += 1
        print(f"{task}: {results[task]} / {tot_tasks[task]}")
    print("\nResults\n" + "-" * 60)
    for task in results:
        print(f"{task}: {results[task]} / {tot_tasks[task]}")

    print(f"SR: {sum(results.values()) / sum(tot_tasks.values()) * 100:.1f}%")


def rollout(
    env,
    model,
    episode,
    task_oracle,
    args,
    task,
    val_annotations,
):
    # state_obs, rgb_obs, depth_obs = episode["robot_obs"], episode["rgb_obs"], episode["depth_obs"]
    reset_info = episode["state_info"]
    # idx = episode["idx"]
    obs = env.reset(
        robot_obs=reset_info["robot_obs"][0], scene_obs=reset_info["scene_obs"][0]
    )
    # get lang annotation for subtask
    # lang_annotation = val_annotations[task][0]
    lang_annotation = episode["lang"]

    model.reset()
    start_info = env.get_info()

    for step in range(args.ep_len):
        # action = episode["actions"][step]
        action = model.step(obs, lang_annotation, episode)
        obs, _, _, current_info = env.step(action)
        # Check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {task.replace(" ", "_")}
        )
        if len(current_task_info) > 0:
            print(colored("S", "green"), end=" ")
            return True
    print(colored("F", "red"), end=" ")
    return False


if __name__ == "__main__":
    seed_everything(0, workers=True)
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )

    parser.add_argument(
        "--eval_log_dir",
        type=str,
        help="Where to log the evaluation results.",
        default=None,
    )

    parser.add_argument(
        "--policy_checkpoint_num",
        type=int,
        help="Policy checkpoint num",
        default=1033,
    )

    parser.add_argument(
        "--policy_results_folder",
        type=str,
        help="Results folder",
        default="/home/grislain/AVDC/calvin/models/policy_huit",
    )

    parser.add_argument(
        "--high_level_checkpoint_num",
        type=int,
        help="High level checkpoint number",
        default=19,
    )

    parser.add_argument(
        "--high_level_results_folder",
        type=str,
        help="Results folder",
        default="/home/grislain/AVDC/results_huit/calvin",
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
        "--use_oracle_subgoals",
        type=bool,
        default="True",
    )

    parser.add_argument(
        "-db",
        "--debug",
        action="store_true",
        help="Print debug info and visualize environment.",
    )

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # Load data config
    policy_data_config = OmegaConf.load(
        os.path.join(args.policy_results_folder, "data_config.yaml")
    )
    high_level_data_config = OmegaConf.load(
        os.path.join(args.high_level_results_folder, "data_config.yaml")
    )
    config = DictConfig(
        {
            "policy": {
                "checkpoint_num": args.policy_checkpoint_num,
                "results_folder": args.policy_results_folder,
                **policy_data_config,
            },
            "high_level": {
                "checkpoint_num": args.high_level_checkpoint_num,
                "results_folder": args.high_level_results_folder,
                "use_oracle_subgoals": args.use_oracle_subgoals,
                **high_level_data_config,
            },
            "debug": args.debug,
            "debug_path": "/home/grislain/AVDC/debug",
        }
    )

    args = parser.parse_args()

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

    # High level config
    high_level_cfg = DictConfig(
        {
            "root": data_path,
            "datamodule": {
                "lang_dataset": {
                    "_target_": "calvin_agent.datasets.disk_dataset.DiskDiffusionOracleDataset",
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
                    "num_subgoals": 64,
                    "pad": True,
                    "lang_folder": "lang_annotations",
                    "num_workers": 2,
                    "diffuse_on": "pixel",
                },
            },
        }
    )

    transforms_dict = OmegaConf.load(
        os.path.join(
            root_path,
            "calvin/calvin_models/conf/datamodule/transforms/play_basic.yaml",
        )
    )

    data_module = CalvinDataModule(
        high_level_cfg.datamodule,
        transforms=transforms_dict,
        root_data_dir=high_level_cfg.root,
    )
    data_module.setup()

    if args.test_on == "train":
        dataloader = data_module.train_dataloader()
        high_level_dataset = dataloader["lang"].dataset
    elif args.test_on == "val":
        dataloader = data_module.val_dataloader()
        high_level_dataset = dataloader.dataset.datasets["lang"]
    else:
        raise ValueError("Invalid test_on argument")

    device = torch.device("cuda:0")
    config.device = "cuda"
    checkpoint = "None"

    if args.debug:
        # Create debug folder
        debug_path = Path(config.debug_path)
        os.makedirs(debug_path, exist_ok=True)

    rollout_cfg = OmegaConf.load(rollout_cfg_path)
    env = hydra.utils.instantiate(
        rollout_cfg.env_cfg, high_level_dataset, device, show_gui=False
    )

    model = CustomModel(config)

    evaluate_policy_singlestep(model, env, high_level_dataset, args, checkpoint)
