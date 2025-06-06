import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from distutils.util import strtobool
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from termcolor import colored
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    SiglipTextModel,
    SiglipTokenizer,
    T5EncoderModel,
)

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

from encoder import R3MEncoder, ViTEncoder
from goal_diffusion import GoalGaussianDiffusion, Trainer
from unet import UnetMW as Unet
from utils import save_images
from vis_features import pca_project_features

sys.path.append(
    os.path.join(
        root_path,
        "calvin/calvin_models",
    )
)


from calvin.calvin_env.calvin_env.envs.play_table_env import get_env
from calvin.calvin_models.calvin_agent.evaluation.multistep_sequences import (
    get_sequences,
)
from calvin.calvin_models.calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin.calvin_models.calvin_agent.models.calvin_base_model import CalvinBaseModel
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,
)

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000


def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class CustomModel(CalvinBaseModel):
    def __init__(self, cfg):
        self.device = torch.device(cfg.device)
        self.cfg = cfg

        # Debug
        self.debug = cfg.debug
        self.debug_path = cfg.debug_path

        # Low level
        stats_path = os.path.join(cfg.policy.root, "training/statistics.yaml")
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

            sample_per_seq = cfg.num_subgoals + 1
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

        # Encoder
        self.use_feat = self.cfg.policy.datamodule.lang_dataset.diffuse_on != "pixel"
        if self.use_feat:
            if "dino_vit" in self.cfg.policy.datamodule.lang_dataset.diffuse_on:
                self.vision_encoder = ViTEncoder()
            elif "r3m" in self.cfg.policy.datamodule.lang_dataset.diffuse_on:
                self.vision_encoder = R3MEncoder("resnet18")

        # Text encoder
        self.use_text = self.cfg.policy.diff_cfg.get("use_text", False)
        if cfg.policy.text_encoder == "CLIP":
            if cfg.server == "jz":
                text_pretrained_model = "/lustre/fsmisc/dataset/HuggingFace_Models/openai/clip-vit-base-patch32"
            else:
                text_pretrained_model = "openai/clip-vit-base-patch32"

            self.tokenizer = CLIPTokenizer.from_pretrained(text_pretrained_model)
            self.text_encoder = CLIPTextModel.from_pretrained(text_pretrained_model)
            text_embed_dim = 512

        elif cfg.policy.text_encoder == "Flan-t5":
            if cfg.server == "jz":
                text_pretrained_model = (
                    "/lustre/fsmisc/dataset/HuggingFace_Models/google/flan-t5-base"
                )
            else:
                text_pretrained_model = "google/flan-t5-base"
            self.text_encoder = T5EncoderModel.from_pretrained(text_pretrained_model)
            self.tokenizer = AutoTokenizer.from_pretrained(text_pretrained_model)
            text_embed_dim = 768

        elif cfg.policy.text_encoder == "Siglip":
            if cfg.server == "jz":
                text_pretrained_model = "/lustre/fsn1/projects/rech/fch/uxv44vw/models/google/siglip-base-patch16-224"
            else:
                text_pretrained_model = "google/siglip-base-patch16-224"
            self.tokenizer = SiglipTokenizer.from_pretrained(text_pretrained_model)
            self.text_encoder = SiglipTextModel.from_pretrained(text_pretrained_model)
            text_embed_dim = 768

        self.text_encoder = self.text_encoder.to(device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # Normlisation
        self.norm_feat = self.cfg.policy.datamodule.lang_dataset.norm_feat
        self.feat_stats_path = (
            Path(self.cfg.policy.root)
            / f"{self.cfg.policy.datamodule.lang_dataset.diffuse_on}_stats.pt"
        )
        if os.path.exists(self.feat_stats_path):
            self.feat_stats = torch.load(self.feat_stats_path)["dino_features"]
        else:
            self.feat_stats = None

    def norm_fonct(self, x):
        if self.use_feat:
            if self.norm_feat is not None:
                if self.feat_stats is not None:
                    if self.norm_feat == "l2":
                        return F.normalize(x, p=2, dim=-1)
                    elif self.norm_feat == "z_score":
                        # Z-score normalization
                        return (x - self.feat_stats["mean"]) / (
                            self.feat_stats["std"] + 1e-6
                        )
                    elif self.norm_feat == "min_max":
                        # MinMax normalization
                        return (x - self.feat_stats["min"]) / (
                            self.feat_stats["max"] - self.feat_stats["min"]
                        ) * 2 - 1
                    else:
                        raise ValueError(
                            f"Normalization method {self.norm_feat} not supported"
                        )
                else:
                    raise FileNotFoundError("Features statistics is None")
            else:
                return x
        else:
            raise ValueError(
                "Normalization is only supported for features, not for pixel data"
            )

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

    def step(self, obs, text_goal, oracle_subgoals=None):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        if self.use_text:
            text_ids = self.tokenizer(
                text_goal,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            encoded_text = self.text_encoder(**text_ids).last_hidden_state

        if not self.use_feat:
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
        else:
            _, init = self.vision_encoder(obs["rgb_obs"]["rgb_static"][0])
            init = self.norm_fonct(init)
            init = rearrange(
                init,
                "f (w h) c -> f c w h",
                w=self.cfg.policy.datamodule.lang_dataset.feat_patch_size,
                h=self.cfg.policy.datamodule.lang_dataset.feat_patch_size,
            )

            # Normalise the init image
            views_static = [init]
            views_gripper = []

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

        # Orcale subgoals
        if not self.cfg.policy.datamodule.lang_dataset.without_guidance:
            if self.use_oracle_subgoals:
                if not self.use_feat:
                    subgoals_views_static = []
                    subgoals_views_gripper = []
                    for key in oracle_subgoals["rgb_obs"].keys():
                        if key == "rgb_static":
                            subgoals_views_static.append(
                                oracle_subgoals["rgb_obs"][key][1:]
                            )
                        elif key == "rgb_gripper":
                            subgoals_views_gripper.append(
                                oracle_subgoals["rgb_obs"][key][1:]
                            )
                    for key in oracle_subgoals["depth_obs"].keys():
                        if key == "depth_static":
                            subgoals_views_static.append(
                                oracle_subgoals["depth_obs"][key][1:]
                            )
                        elif key == "depth_gripper":
                            subgoals_views_gripper.append(
                                oracle_subgoals["depth_obs"][key][1:]
                            )
                else:
                    _, subgoals = self.vision_encoder(
                        oracle_subgoals["rgb_obs"]["rgb_static"][1:].to(self.device)
                    )

                    # Normalise subgoals
                    subgoals = self.norm_fonct(subgoals)

                    subgoals = rearrange(
                        subgoals,
                        "f (w h) c -> f c w h",
                        w=self.cfg.policy.datamodule.lang_dataset.feat_patch_size,
                        h=self.cfg.policy.datamodule.lang_dataset.feat_patch_size,
                    )

                    subgoals_views_static = [subgoals]
                    subgoals_views_gripper = []

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

            else:
                # Generate sequence of subgoals
                sample_subgoals = (
                    self.steps % self.ref_traj_length == 0
                    if self.replan
                    else self.steps == 0
                )

                if sample_subgoals:
                    print(
                        f"Trial {self.steps // 64}: generating subgoals for '{text_goal}'"
                    )
                    self.sub_goals = (
                        self.high_level.sample(
                            init[0], [text_goal], 1, self.guidance_weight
                        )
                        .cpu()
                        .detach()
                    )
                    self.sub_goals = rearrange(
                        self.sub_goals,
                        "b (f c) w h -> b f c w h",
                        c=self.high_level_channels,
                    )[:, :, None, ...]
                    if (
                        self.debug
                        and not self.cfg.policy.datamodule.lang_dataset.without_guidance
                    ):
                        # Save subgoals
                        self.save_image(
                            self.sub_goals[0],
                            f"generated_subgoals_{text_goal}_{self.steps // self.ref_traj_length}.png",
                        )
        else:
            self.sub_goals = torch.zeros_like(init)

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
            if self.cfg.policy.datamodule.lang_dataset.without_guidance:
                obs_goal_images = init
            else:
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
            # Using text goal
            if self.use_text:
                obs_goal["text"] = encoded_text
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


def evaluate_policy(
    model,
    env,
    epoch=0,
    eval_log_dir=None,
    debug=False,
    debug_path=None,
    create_plan_tsne=False,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(__file__).absolute().parents[2] / "calvin/calvin_models/conf"
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml"
    )

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            plans,
            debug,
            debug_path,
        )
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join(
                    [
                        f"{i + 1}/5 : {v * 100:.1f}% |"
                        for i, v in enumerate(count_success(results))
                    ]
                )
                + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    plans,
    debug,
    debug_path,
):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success, length = rollout(
            env, model, task_checker, subtask, val_annotations, plans, debug, debug_path
        )
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


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
    env, model, task_oracle, subtask, val_annotations, plans, debug, debug_path=None
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    obs_list = []
    subgoals = []
    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        if debug_path:
            obs_list.append(obs["rgb_obs"]["rgb_static"][0, 0])

        if debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if debug_path:
            sample_subgoals = (
                step % model.ref_traj_length == 0 if model.replan else step == 0
            )
            if sample_subgoals:
                subgoals.append(model.sub_goals[0])
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            return True, step
    if debug:
        print(colored("fail", "red"), end=" ")

    if debug_path:
        # Create folder for this failed episode
        os.makedirs(os.getcwd() + "/" + debug_path + "/failures/", exist_ok=True)
        failure_idx = len(os.listdir(os.getcwd() + "/" + debug_path + "/failures/"))
        failed_episode_path = os.path.join(
            debug_path, f"failures/failed_{subtask.replace(' ', '_')}_{failure_idx}"
        )
        os.makedirs(
            failed_episode_path,
            exist_ok=True,
        )
        # Save episode (as png)
        model.save_image(
            torch.stack(obs_list),
            os.path.join(
                f"failures/failed_{subtask.replace(' ', '_')}_{failure_idx}",
                "trajectory.png",
            ),
        )
        # Save episode (as gif)
        save_gif(
            obs_list,
            os.path.join(failed_episode_path, "trajectory.gif"),
            duration=1.0,
        )
        # Save subgoals
        if not model.cfg.policy.datamodule.lang_dataset.without_guidance:
            for kk, subgoal in enumerate(subgoals):
                model.save_image(
                    subgoal[:, 0, ...],
                    os.path.join(
                        f"failures/failed_{subtask.replace(' ', '_')}_{failure_idx}",
                        f"subgoals_{kk}.png",
                    ),
                )
    return False, step


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset root directory.",
        default="/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset",
    )

    parser.add_argument(
        "--eval_log_dir",
        type=str,
        help="Where to log the evaluation results.",
        default="eval",
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
        default="/home/grislain/AVDC/calvin/models/results_huit_CLIP",
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
        "--debug_model",
        action="store_true",
        help="Print debug info and visualize environment.",
    )

    parser.add_argument(
        "--debug_path",
        type=str,
        help="Path to save debug images.",
        default="/home/grislain/AVDC/debug_sequential",
    )

    parser.add_argument(
        "--num_subgoals",
        type=int,
        help="Number of subgoals to generate.",
        default=8,
    )

    parser.add_argument(
        "--replan",
        action="store_true",
        help="Replan subgoals every 64 steps.",
    )

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    assert not args.use_oracle_subgoals

    if args.debug_path:
        # Create debug folder
        debug_path = Path(args.debug_path)
        os.makedirs(debug_path, exist_ok=True)

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

    # Load data configs
    policy_data_config = OmegaConf.load(
        os.path.join(args.policy_results_folder, "data_config.yaml")
    )
    policy_data_config.root = data_path

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
                "use_oracle_subgoals": args.use_oracle_subgoals
                if not policy_data_config.datamodule.lang_dataset.without_guidance
                else True,
                **high_level_data_config,
            },
            "debug": args.debug,
            "debug_path": args.debug_path,
            "server": args.server,
            "num_subgoals": args.num_subgoals,
            "replan": args.replan,
        }
    )

    policy_data_config.datamodule.lang_dataset._target_ = (
        "calvin_agent.datasets.disk_dataset.DiskDiffusionOracleDataset"
    )
    del policy_data_config.datamodule.lang_dataset.prob_aug

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

    dataloader = data_module.val_dataloader()
    policy_dataset = dataloader.dataset.datasets["lang"]

    device = torch.device("cuda:0")
    config.device = "cuda"

    print("Config:\n" + OmegaConf.to_yaml(config))

    # Save config
    os.makedirs(args.eval_log_dir, exist_ok=True)
    with open(
        os.path.join(args.eval_log_dir, "config.yaml"),
        "w",
    ) as f:
        OmegaConf.save(config, f)

    if args.debug:
        # Create debug folder
        debug_path = Path(config.debug_path)
        os.makedirs(debug_path, exist_ok=True)

    rollout_cfg = OmegaConf.load(rollout_cfg_path)
    env = hydra.utils.instantiate(
        rollout_cfg.env_cfg, policy_dataset, device, show_gui=False
    )
    model = CustomModel(config)
    evaluate_policy(
        model,
        env,
        debug=args.debug,
        eval_log_dir=debug_path,
        debug_path=args.debug_path,
    )


if __name__ == "__main__":
    main()
