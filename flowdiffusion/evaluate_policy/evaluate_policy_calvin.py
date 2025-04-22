import argparse
import logging
import os
import sys
from collections import Counter
from distutils.util import strtobool
from pathlib import Path

import hydra
import numpy as np
import PIL.Image as Image
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
from transformers import CLIPTextModel, CLIPTokenizer

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

from goal_diffusion import GoalGaussianDiffusion, Trainer
from unet import UnetMW as Unet

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
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

logger = logging.getLogger(__name__)


class CustomModel(CalvinBaseModel):
    def __init__(self, cfg):
        self.device = torch.device(cfg.device)
        self.cfg = cfg

        # Debug
        self.debug = cfg.debug
        self.debug_path = cfg.debug_path

        # Low level
        stats_path = os.path.join(data_path, "training/statistics.yaml")
        train_stats = OmegaConf.load(stats_path)

        self.stats = {
            "action": {
                "max": torch.Tensor(train_stats.act_max_bound),
                "min": torch.Tensor(train_stats.act_min_bound),
            }
        }

        self.steps = 0
        self.guidance_weight = 3
        self.sample_action_every = int(np.ceil(65 / cfg.num_subgoals)) - 1

        self.policy = DiffusionPolicy(
            self.cfg.policy.diff_cfg, dataset_stats=self.stats
        )
        policy_checkpoint_path = (
            self.cfg.policy.results_folder
            + f"/model-{self.cfg.policy.checkpoint_num}.pt"
        )
        if os.path.exists(policy_checkpoint_path):
            self.policy.load_state_dict(
                torch.load(
                    policy_checkpoint_path,
                )
            )
        else:
            print(f"Policy checkpoint not found at {policy_checkpoint_path}")
        self.policy.eval()
        self.policy.to(self.device)

        # High level

        self.use_oracle_subgoals = cfg.high_level.use_oracle_subgoals
        self.sample_subgoals_every = 8
        if not self.use_oracle_subgoals:
            if cfg.high_level.datamodule.lang_dataset.diffuse_on == "pixel":
                self.high_level_channels = 3
                feature_decoder = None
            elif cfg.high_level.datamodule.lang_dataset.diffuse_on == "dino_feat":
                self.high_level_channels = 768
                raise NotImplementedError

            unet = Unet(in_channels=self.high_level_channels)
            unet = unet.to(self.device)

            if cfg.server == "jz":
                pretrained_model = "/lustre/fsmisc/dataset/HuggingFace_Models/openai/clip-vit-base-patch32"
            else:
                pretrained_model = "openai/clip-vit-base-patch32"

            tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
            text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
            text_encoder.requires_grad_(False)
            text_encoder.eval()

            sample_per_seq = 8
            self.sample_steps = 100

            diffusion = GoalGaussianDiffusion(
                channels=3 * (sample_per_seq - 1),
                model=unet,
                image_size=(96, 96),
                timesteps=100,
                sampling_timesteps=self.sample_steps,
                loss_type="l2",
                objective=cfg.high_level.diffusion_objective,
                beta_schedule="cosine",
                min_snr_loss_weight=True,
                auto_normalize=False,
            )

            trainer = Trainer(
                diffusion_model=diffusion,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                train_set=[None],
                valid_set=[None],
                train_lr=1e-4,
                train_num_steps=60000,
                save_and_sample_every=2500,
                ema_update_every=10,
                ema_decay=0.999,
                train_batch_size=16,
                valid_batch_size=32,
                gradient_accumulate_every=1,
                num_samples=1,
                results_folder=self.cfg.high_level.results_folder,
                fp16=True,
                amp=True,
                calculate_fid=False,
                feature_decoder=feature_decoder,
            )

            if self.cfg.high_level.checkpoint_num is not None:
                trainer.load(self.cfg.high_level.checkpoint_num)

            self.high_level = trainer

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

        # Orcale subgoals
        if self.use_oracle_subgoals:
            self.sub_goals = oracle_subgoals["rgb_obs"]["rgb_static"][1:][None]

            if self.debug:
                # Save subgoals
                self.save_image(
                    self.sub_goals[0],
                    f"oracle_subgoals_{text_goal}.png",
                )
                # Save initial frame
                self.save_image(
                    oracle_subgoals["rgb_obs"]["rgb_static"][0],
                    f"oracle_init_{text_goal}.png",
                )
        else:
            # Generate sequence of subgoals
            if self.steps == 0:
                self.sub_goals = (
                    self.high_level.sample(
                        obs_image[0], [text_goal], 1, self.guidance_weight
                    )
                    .cpu()
                    .detach()
                )
                self.sub_goals = rearrange(
                    self.sub_goals, "b (f c) w h -> b f c w h", c=3
                )
                if self.debug:
                    # Save subgoals
                    self.save_image(
                        self.sub_goals[0],
                        f"generated_subgoals_{text_goal}.png",
                    )

        if self.steps % self.sample_action_every == 0:
            # Select subgoal
            sub_gaol_idx = min(
                self.steps // self.sample_action_every, self.sub_goals.shape[1] - 1
            )

            target = self.sub_goals[0, sub_gaol_idx].to(self.device)
            init = obs["rgb_obs"]["rgb_static"][0, 0]

            obs_goal_images = torch.cat([init[None], target[None]], dim=0)

            # Save initial and target frames
            if self.debug:
                self.save_image(
                    obs_goal_images,
                    f"init_target_{self.steps}.png",
                )

            state = torch.zeros((2, 0)).to(self.device)
            obs_goal = {
                "observation.state": state[None],
                "observation.images": obs_goal_images[None, :, None, ...],
            }
            # Predict action
            with torch.inference_mode():
                self.actions = (
                    self.policy.diffusion.generate_actions(obs_goal).cpu().detach()[0]
                )

            # Unormalise actions
            self.actions = (self.actions + 1) / 2
            self.actions = (
                self.actions
                * (self.stats["action"]["max"] - self.stats["action"]["min"])
                + self.stats["action"]["min"]
            )

            # Project gripper closure to {-1, 1}
            self.actions[:, -1] = torch.sign(self.actions[:, -1])
        selected_action = self.actions[self.steps % self.sample_action_every]
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

    # Save results
    with open(os.path.join(args.eval_folder, f"results_{args.test_on}.txt"), "w") as f:
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
    obs_list = []
    for step in range(args.ep_len):
        # action = episode["actions"][step]
        action = model.step(obs, lang_annotation, episode)
        obs, _, _, current_info = env.step(action)
        if args.save_failures:
            obs_list.append(obs["rgb_obs"]["rgb_static"][0, 0])
        # Check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {task.replace(" ", "_")}
        )
        if len(current_task_info) > 0:
            print(colored("S", "green"), end=" ")
            return True
    print(colored("F", "red"), end=" ")
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
        # Save subgoals
        torchvision.utils.save_image(
            (model.sub_goals[0] + 1) / 2,
            os.path.join(
                failed_episode_path,
                "subgoals.png",
            ),
        )
    return False


if __name__ == "__main__":
    seed_everything(0, workers=True)
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
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
        default=100,
    )

    parser.add_argument(
        "--high_level_results_folder",
        type=str,
        help="Results folder",
        default="/home/grislain/AVDC/calvin/models/results_huit_ann/calvin",
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
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Use oracle subgoals",
    )

    parser.add_argument(
        "-db",
        "--debug",
        action="store_true",
        help="Print debug info and visualize environment.",
    )

    parser.add_argument(
        "--num_subgoals",
        type=int,
        default=8,
        help="Number of subgoals to generate.",
    )

    parser.add_argument(
        "--save_failures",
        action="store_true",
        help="Save failed episodes.",
    )

    parser.add_argument(
        "--debug_path",
        type=str,
        default="/home/grislain/AVDC/debug",
        help="Path to save debug images.",
    )

    parser.add_argument(
        "--eval_folder",
        type=str,
        default="eval",
        help="Folder to save evaluation results.",
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
            "debug_path": args.debug_path,
            "server": args.server,
            "num_subgoals": args.num_subgoals,
        }
    )

    # Save config
    os.makedirs(args.eval_folder, exist_ok=True)
    with open(
        os.path.join(args.eval_folder, "config.yaml"),
        "w",
    ) as f:
        OmegaConf.save(config, f)

    if args.use_oracle_subgoals:
        print("Using oracle subgoals")
    else:
        print("Using generated subgoals")

    # Do not change
    args.ep_len = 240

    if args.server == "jz":
        data_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/CALVIN/task_D_D"
        rollout_cfg_path = "/lustre/fswork/projects/rech/fch/uxv44vw/clemgris/avdc/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
    elif args.server == "hacienda":
        data_path = "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset"
        rollout_cfg_path = "/home/grislain/AVDC/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
    else:
        raise ValueError("Invalid server argument")

    # load high level config
    high_level_cfg = OmegaConf.load(
        os.path.join(
            args.high_level_results_folder,
            "data_config.yaml",
        )
    )
    high_level_cfg.datamodule.lang_dataset._target_ = (
        "calvin_agent.datasets.disk_dataset.DiskDiffusionOracleDataset"
    )
    high_level_cfg.root = data_path

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
