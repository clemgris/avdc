import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    SiglipTextModel,
    SiglipTokenizer,
    T5EncoderModel,
)

# === Project Path Setup ===
ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "flowdiffusion"),
        str(ROOT_PATH / "calvin/calvin_models"),
    ]
)

# === Local Imports ===
from encoder import R3MEncoder, ViTEncoder
from goal_diffusion import GoalGaussianDiffusion, Trainer
from tools import save_images
from unet import UnetMW as Unet
from vis_features import pca_project_features

# === CALVIN and LEROBOT Imports ===
from calvin.calvin_models.calvin_agent.models.calvin_base_model import CalvinBaseModel
from lerobot.common.policies.diffusion.modeling_diffusion import (
    DiffusionConfig,
    DiffusionPolicy,
)

logger = logging.getLogger(__name__)


class HierarchicalModel(CalvinBaseModel):
    def __init__(self, cfg):
        self.device = torch.device(cfg.device)
        self.cfg = cfg
        self.reset()

        # Debug
        self.debug_path = cfg.debug_path

        # Load statistics
        self.stats = self._load_stats(cfg)

        # Policy
        self.sample_action_every = int(np.ceil(65 / cfg.num_subgoals)) - 1
        self._init_policy(cfg)
        self.without_guidance = self.cfg.policy.datamodule.lang_dataset.get(
            "without_guidance", False
        )

        # High level
        self.replan = cfg.replan
        self.ref_traj_length = 64
        self.guidance_weight = 3
        self.sample_subgoals_every = 8
        self.use_oracle_subgoals = cfg.high_level.use_oracle_subgoals
        self.use_text = self.cfg.policy.diff_cfg.get("use_text", False)

        if not self.use_oracle_subgoals:
            # Text encoder
            self._init_text_encoder(cfg)
            # High level diffusion model
            self._init_high_level(cfg)

        # Vision encoder
        self.use_feat = self.cfg.policy.datamodule.lang_dataset.diffuse_on != "pixel"
        if self.use_feat:
            self._init_vision_encoder(cfg)

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

    def _load_stats(self, cfg):
        # Low level
        stats_path = os.path.join(cfg.policy.root, "training/statistics.yaml")
        train_stats = OmegaConf.load(stats_path)

        return {
            "action": {
                "max": torch.Tensor(train_stats.act_max_bound),
                "min": torch.Tensor(train_stats.act_min_bound),
            }
        }

    def _init_policy(self, cfg):
        self.policy = DiffusionPolicy(
            DiffusionConfig(**cfg.policy.diff_cfg), dataset_stats=self.stats
        )
        policy_checkpoint_path = (
            cfg.policy.results_folder + f"/model-{cfg.policy.checkpoint_num}.pt"
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

    def _init_vision_encoder(self, cfg):
        if "dino_vit" in cfg.policy.datamodule.lang_dataset.diffuse_on:
            self.vision_encoder = ViTEncoder()
        elif "r3m" in cfg.policy.datamodule.lang_dataset.diffuse_on:
            self.vision_encoder = R3MEncoder("resnet18")

    def _init_text_encoder(self, cfg):
        text_encoder_name = cfg.policy.get("text_encoder", "CLIP")
        if text_encoder_name == "CLIP":
            if cfg.server == "jz":
                text_pretrained_model = "/lustre/fsmisc/dataset/HuggingFace_Models/openai/clip-vit-base-patch32"
            else:
                text_pretrained_model = "openai/clip-vit-base-patch32"

            self.tokenizer = CLIPTokenizer.from_pretrained(text_pretrained_model)
            self.text_encoder = CLIPTextModel.from_pretrained(text_pretrained_model)
            self.text_embed_dim = 512
            self.precision = "fp16"
            self.amp = True

        elif text_encoder_name == "Flan-t5":
            if cfg.server == "jz":
                text_pretrained_model = (
                    "/lustre/fsmisc/dataset/HuggingFace_Models/google/flan-t5-base"
                )
            else:
                text_pretrained_model = "google/flan-t5-base"
            self.text_encoder = T5EncoderModel.from_pretrained(text_pretrained_model)
            self.tokenizer = AutoTokenizer.from_pretrained(text_pretrained_model)
            self.text_embed_dim = 768
            self.precision = "no"
            self.amp = False

        elif text_encoder_name == "Siglip":
            if cfg.server == "jz":
                text_pretrained_model = "/lustre/fsn1/projects/rech/fch/uxv44vw/models/google/siglip-base-patch16-224"
            else:
                text_pretrained_model = "google/siglip-base-patch16-224"
            self.tokenizer = SiglipTokenizer.from_pretrained(text_pretrained_model)
            self.text_encoder = SiglipTextModel.from_pretrained(text_pretrained_model)
            self.text_embed_dim = 768
            self.precision = "fp16"
            self.amp = True

        self.text_encoder = self.text_encoder.to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

    def _init_high_level(self, cfg):
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
            feature_decoder = None
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

        unet = Unet(
            in_channels=self.high_level_channels,
            channel_mult=channel_mult,
            text_embed_dim=self.text_embed_dim,
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
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
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
            results_folder=cfg.high_level.results_folder,
            precision=self.precision,
            amp=self.amp,
            calculate_fid=False,
            feature_decoder=feature_decoder,
        )

        if cfg.high_level.checkpoint_num is not None:
            trainer.load(cfg.high_level.checkpoint_num)

        self.high_level = trainer

    def _norm_feat_fonct(self, x):
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

    def _unorm_action_fonct(self, actions):
        actions = (actions + 1) / 2
        actions = (
            actions * (self.stats["action"]["max"] - self.stats["action"]["min"])
            + self.stats["action"]["min"]
        )
        return actions

    def reset(self):
        self.steps = 0

    def save_image(self, image, name):
        saving_path = Path(self.debug_path) / name
        if image.shape[1] > 8:
            image = rearrange(image, "f c h w -> f (h w) c")
            image = pca_project_features(image.to(self.device).detach())
        else:
            image = (image + 1) / 2
        save_images(image, saving_path, nrow=image.shape[0])

    def _extract_text_embedding(self, text_goal):
        text_ids = self.tokenizer(
            text_goal,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        encoded_text = self.text_encoder(**text_ids).last_hidden_state
        return encoded_text

    def _extract_obs(self, obs: dict, idx: int | np.ndarray):
        if not self.use_feat:
            views_static = []
            views_gripper = []
            for key in obs["rgb_obs"].keys():
                if key == "rgb_static":
                    views_static.append(obs["rgb_obs"][key][idx])
                elif key == "rgb_gripper":
                    views_gripper.append(obs["rgb_obs"][key][idx])
            for key in obs["depth_obs"].keys():
                if key == "depth_static":
                    views_static.append(obs["depth_obs"][key][idx])
                elif key == "depth_gripper":
                    views_gripper.append(obs["depth_obs"][key][idx])
        else:
            _, image = self.vision_encoder(
                obs["rgb_obs"]["rgb_static"][idx].to(self.device)
            )
            image = self._norm_feat_fonct(image)
            image = rearrange(
                image,
                "f (w h) c -> f c w h",
                w=self.cfg.policy.datamodule.lang_dataset.feat_patch_size,
                h=self.cfg.policy.datamodule.lang_dataset.feat_patch_size,
            )

            # Normalise the init image
            views_static = [image]
            views_gripper = []

        assert len(views_static) > 0 or len(views_gripper) > 0

        image_static = torch.cat(views_static, dim=1) if len(views_static) > 0 else None
        image_gripper = (
            torch.cat(views_gripper, dim=1) if len(views_gripper) > 0 else None
        )

        image = (
            torch.cat(
                [image_static[:, None], image_gripper[:, None]],
                dim=-4,
            )
            if len(views_gripper) > 0
            else image_static[:, None]
        ).to(self.device)

        return image

    def step(self, obs, text_goal, oracle_subgoals=None):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        # Extract text goal embedding
        if self.use_text:
            encoded_text = self._extract_text_embedding(text_goal)
        init = self._extract_obs(obs, 0)

        # Orcale subgoals
        if self.use_oracle_subgoals:
            self.sub_goals = self._extract_obs(oracle_subgoals, slice(1, None))[None]
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
            if self.without_guidance:
                obs_goal_images = init
            else:
                obs_goal_images = torch.cat([init, target], dim=0)

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
            self.actions = self._unorm_action_fonct(self.actions)

            # Project gripper closure to {-1, 1}
            self.actions[:, -1] = torch.sign(self.actions[:, -1])
        selected_action = self.actions[self.steps % self.sample_action_every]
        self.steps += 1
        return selected_action
