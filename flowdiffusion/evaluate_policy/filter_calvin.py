# === Standard Library ===
import argparse
import logging
import os
import sys
from pathlib import Path

# === Third-party Libraries ===
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

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
from methods.evaluate_policy import evaluate_policy_singlestep_and_save
from model.expert_model import ExpertModel

from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,
)

logger = logging.getLogger(__name__)


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
        "--data_path",
        type=str,
        default=None,
        help="Path to the dataset.",
    )

    parser.add_argument(
        "--debug_path",
        type=str,
        default=None,
        help="Path to save debug images.",
    )

    parser.add_argument(
        "--num_trial",
        type=int,
        default=5,
        help="Trial for testing expert demo.",
    )

    parser.add_argument(
        "--prob_sucess",
        type=float,
        default=0.8,
        help="Probability of success for expert demo.",
    )

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()
    args.save_failures = args.debug_path is not None

    # Do not change
    args.ep_len = 100

    if args.server == "jz":
        data_path = (
            args.data_path
            or "/lustre/fsn1/projects/rech/fch/uxv44vw/CALVIN/task_D_D_jz"
        )
        rollout_cfg_path = "/lustre/fswork/projects/rech/fch/uxv44vw/clemgris/avdc/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
        conf_dir = Path(
            "/lustre/fswork/projects/rech/fch/uxv44vw/clemgris/avdc/calvin/calvin_models/conf"
        )

    elif args.server == "hacienda":
        data_path = (
            args.data_path or "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset"
        )
        rollout_cfg_path = "/home/grislain/AVDC/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
        conf_dir = Path("/home/grislain/AVDC/calvin/calvin_models/conf")
    else:
        raise ValueError("Invalid server argument")

    transforms_dict = OmegaConf.load(
        os.path.join(
            ROOT_PATH,
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
    checkpoint = "None"

    if args.debug_path is not None:
        # Create debug folder
        debug_path = Path(args.debug_path)
        os.makedirs(debug_path, exist_ok=True)

    rollout_cfg = OmegaConf.load(rollout_cfg_path)
    env = hydra.utils.instantiate(
        rollout_cfg.env_cfg, policy_dataset, device, show_gui=False
    )
    model = ExpertModel()

    evaluate_policy_singlestep_and_save(
        model,
        env,
        policy_dataset,
        args,
        conf_dir,
        prob_sucess=args.prob_sucess,
        num_trial=args.num_trial,
    )
