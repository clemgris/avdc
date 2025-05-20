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
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    SiglipTextModel,
    SiglipTokenizer,
    T5EncoderModel,
)

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
from utils import save_images

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
            raise ValueError(f"Policy checkpoint not found at {policy_checkpoint_path}")
        self.policy.eval()
        self.policy.to(self.device)

        # High level

        self.replan = cfg.replan
        self.ref_traj_length = 64

        self.use_oracle_subgoals = cfg.high_level.use_oracle_subgoals
        self.sample_subgoals_every = 8
        if not self.use_oracle_subgoals:
            if cfg.high_level.datamodule.lang_dataset.diffuse_on == "pixel":
                if (
                    "depth_static"
                    in cfg.high_level.datamodule.lang_dataset.obs_space.depth_obs
                ):
                    self.high_level_channels = 4
                else:
                    self.high_level_channels = 3
                feature_decoder = None
                target_size = (96, 96)
                channel_mult = (1, 2, 3, 4, 5)
            elif "dino" in cfg.high_level.datamodule.lang_dataset.diffuse_on:
                self.high_level_channels = 768
                target_size = (
                    cfg.high_level.datamodule.lang_dataset.feat_patch_size,
                    cfg.high_level.datamodule.lang_dataset.feat_patch_size,
                )
                if cfg.high_level.datamodule.lang_dataset.feat_patch_size == 16:
                    channel_mult = (1, 2, 3)
                elif cfg.high_level.datamodule.lang_dataset.feat_patch_size == 32:
                    channel_mult = (1, 2, 3, 4)
                elif cfg.high_level.datamodule.lang_dataset.feat_patch_size == 64:
                    channel_mult = (1, 2, 3, 4, 5)

            if cfg.high_level.text_encoder == "CLIP":
                if cfg.server == "jz":
                    text_pretrained_model = "/lustre/fsmisc/dataset/HuggingFace_Models/openai/clip-vit-base-patch32"
                else:
                    text_pretrained_model = "openai/clip-vit-base-patch32"

                tokenizer = CLIPTokenizer.from_pretrained(text_pretrained_model)
                text_encoder = CLIPTextModel.from_pretrained(text_pretrained_model)
                text_embed_dim = 512
                amp = True
                precision = "fp16"

            elif cfg.high_level.text_encoder == "Flan-t5":
                if cfg.server == "jz":
                    text_pretrained_model = (
                        "/lustre/fsmisc/dataset/HuggingFace_Models/google/flan-t5-base"
                    )
                else:
                    text_pretrained_model = "google/flan-t5-base"
                text_encoder = T5EncoderModel.from_pretrained(text_pretrained_model)
                tokenizer = AutoTokenizer.from_pretrained(text_pretrained_model)
                text_embed_dim = 768
                amp = False
                precision = "no"

            elif cfg.high_level.text_encoder == "Siglip":
                if cfg.server == "jz":
                    text_pretrained_model = "/lustre/fsn1/projects/rech/fch/uxv44vw/models/google/siglip-base-patch16-224"
                else:
                    text_pretrained_model = "google/siglip-base-patch16-224"
                tokenizer = SiglipTokenizer.from_pretrained(text_pretrained_model)
                text_encoder = SiglipTextModel.from_pretrained(text_pretrained_model)
                text_embed_dim = 768
                amp = True
                precision = "fp16"

            text_encoder = text_encoder.to(device)
            text_encoder.requires_grad_(False)
            text_encoder.eval()

            text_encoder_num_params = sum(p.numel() for p in text_encoder.parameters())
            print(
                f"Number of parameters in text encoder {text_pretrained_model}: {text_encoder_num_params / 1e6:.2f}M"
            )

            unet = Unet(
                in_channels=self.high_level_channels,
                channel_mult=channel_mult,
                text_embed_dim=text_embed_dim,
            )
            unet = unet.to(self.device)

            sample_per_seq = 8
            self.sample_steps = 100

            diffusion = GoalGaussianDiffusion(
                channels=self.high_level_channels * (sample_per_seq - 1),
                model=unet,
                image_size=target_size,
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
                precision=precision,
                amp=amp,
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
        save_images((image + 1) / 2, saving_path, nrow=image.shape[0])

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
        if "depth_static" in obs["depth_obs"].keys():
            obs_image = torch.cat(
                [
                    obs_image,
                    obs["depth_obs"]["depth_static"],
                ],
                dim=-3,
            )
        # Save image
        if self.debug:
            self.save_image(
                obs_image[0],
                f"obs_{self.steps}.png",
            )

        # Orcale subgoals
        if self.use_oracle_subgoals:
            subgoals_views_static = []
            subgoals_views_gripper = []
            for key in oracle_subgoals["rgb_obs"].keys():
                if key == "rgb_static":
                    subgoals_views_static.append(oracle_subgoals["rgb_obs"][key][1:])
                elif key == "rgb_gripper":
                    subgoals_views_gripper.append(oracle_subgoals["rgb_obs"][key][1:])
            for key in oracle_subgoals["depth_obs"].keys():
                if key == "depth_static":
                    subgoals_views_static.append(oracle_subgoals["depth_obs"][key][1:])
                elif key == "depth_gripper":
                    subgoals_views_gripper.append(oracle_subgoals["depth_obs"][key][1:])

            assert len(subgoals_views_static) > 0 or len(subgoals_views_gripper) > 0

            subgoals_image_static = (
                torch.cat(subgoals_views_static, dim=1)
                if len(subgoals_views_static) > 0
                else None
            )
            subgoals_image_gripper = (
                torch.cat(subgoals_views_gripper, dim=1)
                if len(subgoals_views_gripper) > 0
                else None
            )

            self.sub_goals = (
                torch.cat(
                    [
                        subgoals_image_static[:, None],
                        subgoals_image_gripper[:, None],
                    ],
                    dim=-4,
                )
                if len(subgoals_views_gripper) > 0
                else subgoals_image_static[:, None]
            )[None].to(self.device)

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
            sample_subgoals = (
                self.steps % model.ref_traj_length == 0
                if self.replan
                else self.steps == 0
            )

            if sample_subgoals:
                print(
                    f"Trial {self.steps // 64}: generating subgoals for '{text_goal}'"
                )
                self.sub_goals = (
                    self.high_level.sample(
                        obs_image[0], [text_goal], 1, self.guidance_weight
                    )
                    .cpu()
                    .detach()
                )
                self.sub_goals = rearrange(
                    self.sub_goals,
                    "b (f c) w h -> b f c w h",
                    c=self.high_level_channels,
                )
                if self.debug:
                    # Save subgoals
                    self.save_image(
                        self.sub_goals[0],
                        f"generated_subgoals_{text_goal}_{self.steps // self.ref_traj_length}.png",
                    )

        if self.steps % self.sample_action_every == 0:
            # Select subgoal
            if self.replan:
                sub_goal_idx = (self.steps // self.sample_action_every) % (
                    self.sub_goals.shape[1]
                )
            else:
                sub_goal_idx = min(
                    self.steps // self.sample_action_every, self.sub_goals.shape[1] - 1
                )

            target = self.sub_goals[:, sub_goal_idx].to(self.device)
            views_static = []
            views_gripper = []
            for key in obs["rgb_obs"].keys():
                if key == "rgb_static":
                    views_static.append(obs["rgb_obs"][key][0])
                elif key == "rgb_gripper":
                    views_gripper.append(obs["rgb_obs"][key][0])
            for key in obs["depth_obs"].keys():
                if key == "depth_static":
                    views_static.append(obs["depth_obs"][key][0])
                elif key == "depth_gripper":
                    views_gripper.append(obs["depth_obs"][key][0])

            assert len(views_static) > 0 or len(views_gripper) > 0

            start_image_static = (
                torch.cat(views_static, dim=1) if len(views_static) > 0 else None
            )
            start_image_gripper = (
                torch.cat(views_gripper, dim=1) if len(views_gripper) > 0 else None
            )

            init = (
                torch.cat(
                    [start_image_static[:, None], start_image_gripper[:, None]],
                    dim=-4,
                )
                if len(views_gripper) > 0
                else start_image_static[:, None]
            ).to(self.device)
            obs_goal_images = torch.cat([init, target], dim=0)

            # Save initial and target frames
            if self.debug:
                self.save_image(
                    obs_goal_images,
                    f"init_target_{self.steps}.png",
                )

            state = torch.zeros((obs_goal_images.shape[0], 0)).to(self.device)
            obs_goal = {
                "observation.state": state[None],
                "observation.images": obs_goal_images[None],
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
        success, length = rollout(
            env, model, episode, task_oracle, args, task, val_annotations
        )
        results[task] += success
        tot_tasks[task] += 1
        print(f"{task}: {results[task]} / {tot_tasks[task]} ({length})")

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
    subgoals = []
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

        if args.save_failures:
            sample_subgoals = (
                step % model.ref_traj_length == 0 if model.replan else step == 0
            )
            if sample_subgoals:
                subgoals.append(model.sub_goals[0])

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
        # Save subgoals
        for kk, subgoal in enumerate(subgoals):
            model.save_image(
                subgoal[:, 0, ...],
                f"failed_{task.replace(' ', '_')}_{episode['idx']}/subgoals_{kk}.png",
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
        help="Save generated subgoals and observation.",
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

    parser.add_argument(
        "--replan",
        action="store_true",
        help="Replan subgoals every 64 steps.",
    )

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # Load data config
    policy_data_config = OmegaConf.load(
        os.path.join(args.policy_results_folder, "data_config.yaml")
    )
    if not args.use_oracle_subgoals:
        high_level_data_config = OmegaConf.load(
            os.path.join(args.high_level_results_folder, "data_config.yaml")
        )
    else:
        high_level_data_config = {}
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
            "replan": args.replan,
        }
    )

    print("Config:\n" + OmegaConf.to_yaml(config))

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
        data_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/CALVIN/task_D_D_jz"
        rollout_cfg_path = "/lustre/fswork/projects/rech/fch/uxv44vw/clemgris/avdc/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
    elif args.server == "hacienda":
        data_path = "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset"
        rollout_cfg_path = "/home/grislain/AVDC/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
    else:
        raise ValueError("Invalid server argument")

    # load low level config
    policy_data_config.datamodule.lang_dataset._target_ = (
        "calvin_agent.datasets.disk_dataset.DiskDiffusionOracleDataset"
    )
    del policy_data_config.datamodule.lang_dataset.prob_aug
    policy_data_config.root = data_path

    transforms_dict = OmegaConf.load(
        os.path.join(
            root_path,
            "calvin/calvin_models/conf/datamodule/transforms/play_basic.yaml",
        )
    )

    data_module = CalvinDataModule(
        policy_data_config.datamodule,
        transforms=transforms_dict,
        root_data_dir=policy_data_config.root,
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
