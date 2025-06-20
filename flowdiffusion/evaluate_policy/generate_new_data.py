# === Standard Library ===
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

# === Third-party Libraries ===
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

# === Set Up Paths ===
ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))  # Top-level project root
sys.path.extend(
    [
        str(ROOT_PATH / "flowdiffusion"),
        str(ROOT_PATH / "calvin/calvin_models"),
    ]
)

# === Local Project Imports ===
from methods.generate_data import generate_new_data
from model.hierarchical_model import HierarchicalModel

# === CALVIN Imports ===
from calvin.calvin_env.calvin_env.envs.play_table_env import get_env
from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,
)

# === Device Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


if __name__ == "__main__":
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )

    parser.add_argument(
        "--eval_folder",
        type=str,
        help="Where to log the evaluation results.",
        default="eval_long_horizon",
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

    parser.add_argument(
        "--saving_path",
        type=str,
        help="Path to save the generated data.",
        default=None,
    )

    parser.add_argument(
        "--num_data",
        type=int,
        help="Number of data points to generate.",
        default=100,
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Task to generate data for.",
        default=None,
    )

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    if args.debug_path:
        # Create debug folder
        debug_path = Path(args.debug_path)
        os.makedirs(debug_path, exist_ok=True)

    # Do not change
    args.ep_len = 240

    if args.server == "jz":
        data_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/CALVIN/task_D_D_jz"
        rollout_cfg_path = "/lustre/fswork/projects/rech/fch/uxv44vw/clemgris/avdc/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
        conf_dir = Path(
            "/lustre/fswork/projects/rech/fch/uxv44vw/clemgris/avdc/calvin/calvin_models/conf"
        )

    elif args.server == "hacienda":
        data_path = "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset"
        rollout_cfg_path = "/home/grislain/AVDC/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
        conf_dir = Path("/home/grislain/AVDC/calvin/calvin_models/conf")
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
                "use_oracle_subgoals": policy_data_config.datamodule.lang_dataset.get(
                    "without_guidance", False
                ),
                **high_level_data_config,
            },
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
            ROOT_PATH,
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

    saving_path = args.saving_path or os.path.join(args.debug_path, "generated_data")

    # Save config
    os.makedirs(saving_path, exist_ok=True)
    with open(
        os.path.join(saving_path, "config.yaml"),
        "w",
    ) as f:
        OmegaConf.save(config, f)

    for subdir in ["training", "validation"]:
        os.makedirs(os.path.join(saving_path, subdir), exist_ok=True)
        os.makedirs(
            os.path.join(saving_path, subdir, "lang_annotations"), exist_ok=True
        )

    if args.debug_path:
        # Create debug folder
        debug_path = Path(config.debug_path)
        os.makedirs(debug_path, exist_ok=True)

    rollout_cfg = OmegaConf.load(rollout_cfg_path)
    env = hydra.utils.instantiate(
        rollout_cfg.env_cfg, policy_dataset, device, show_gui=False
    )
    model = HierarchicalModel(config)
    generate_new_data(
        model,
        env,
        eval_folder=args.eval_folder,
        debug_path=args.debug_path,
        conf_dir=conf_dir,
        num_data=args.num_data,
        task=args.task,
        saving_path=os.path.join(saving_path, "training"),
    )

    shutil.copytree(
        os.path.join(policy_data_config.root, "training", ".hydra"),
        os.path.join(saving_path, "training", ".hydra"),
        dirs_exist_ok=True,
    )

    generate_new_data(
        model,
        env,
        eval_folder=args.eval_folder,
        debug_path=args.debug_path,
        conf_dir=conf_dir,
        num_data=1,
        task=args.task,
        saving_path=os.path.join(saving_path, "validation"),
    )

    shutil.copytree(
        os.path.join(policy_data_config.root, "validation", ".hydra"),
        os.path.join(saving_path, "validation", ".hydra"),
        dirs_exist_ok=True,
    )
