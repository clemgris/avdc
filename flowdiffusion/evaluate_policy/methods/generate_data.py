# === Standard Library ===
import logging
import os
import sys
from pathlib import Path

# === Third-party Libraries ===
import hydra
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored

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
# === CALVIN Imports ===
from calvin.calvin_models.calvin_agent.evaluation.multistep_sequences import (
    get_initial_states,
)
from calvin.calvin_models.calvin_agent.evaluation.utils import (
    get_env_state_for_initial_condition,
    get_log_dir,
)
from methods.rollout import rollout_data_collection

# === Logger ===
logger = logging.getLogger(__name__)


NUM_SEQUENCES = 1000


def generate_new_data(
    model,
    env,
    epoch=0,
    eval_folder=None,
    debug_path=None,
    conf_dir=None,
    num_data=1000,
    task: str = None,
    saving_path: str = None,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_folder: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.

    Returns:
        Dictionary with results
    """
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable.yaml")

    eval_folder = get_log_dir(eval_folder)

    eval_sequences = get_initial_states(num_data=num_data, task=task)
    results = []
    auto_lang_ann = {
        "info": {"episodes": [], "indx": [], "length": []},
        "language": {"ann": [], "task": []},
    }
    success_counter = 0
    for initial_state, eval_sequence in eval_sequences:
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        subtask = eval_sequence[0]
        success, length, (start, end), ann = rollout_data_collection(
            env, model, task_oracle, subtask, val_annotations, debug_path, saving_path
        )
        if success:
            success_counter += 1
            print(colored("S", "green"), subtask, end=" ")

            auto_lang_ann["info"]["indx"].append((start, end))
            auto_lang_ann["language"]["ann"].append(ann)
            auto_lang_ann["language"]["task"].append(task)
            auto_lang_ann["info"]["length"].append(length)

    # Save language annotations
    ann_saving_path = os.path.join(saving_path, "lang_annotations/auto_lang_ann.npy")
    np.save(
        ann_saving_path,
        auto_lang_ann,
        allow_pickle=True,
    )

    print(
        "Created",
        success_counter,
        "successful episodes out of",
        num_data,
        "for the task",
        task,
        "at",
        saving_path,
    )

    return results
