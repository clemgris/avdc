import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from termcolor import colored

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "flowdiffusion"),
        str(ROOT_PATH / "calvin/calvin_models"),
    ]
)

from utils.vis import save_gif

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000


def rollout_with_oracle(
    env,
    model,
    episode,
    task_oracle,
    args,
    task,
):
    reset_info = episode["state_info"]
    obs = env.reset(
        robot_obs=reset_info["robot_obs"][0], scene_obs=reset_info["scene_obs"][0]
    )
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


def rollout(env, model, task_oracle, subtask, val_annotations, debug_path=None):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
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
            print(colored("S", "green"), end=" ")
            return True, step
    print(colored("F", "red"), end=" ")

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
        if not model.cfg.policy.datamodule.lang_dataset.get("without_guidance", False):
            for kk, subgoal in enumerate(subgoals):
                model.save_image(
                    subgoal[:, 0, ...],
                    os.path.join(
                        f"failures/failed_{subtask.replace(' ', '_')}_{failure_idx}",
                        f"subgoals_{kk}.png",
                    ),
                )
    return False, step


def rollout_data_collection(
    env, model, task_oracle, subtask, annotations, debug_path=None, saving_path=None
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = random.choice(annotations[subtask])
    model.reset()
    start_info = env.get_info()

    frames = []
    for step in range(65):
        # Count the number of frames in the saving folder
        frame_idx = sum(
            1
            for f in os.listdir(saving_path)
            if f.startswith("episode_") and os.path.isfile(f"{saving_path}/{f}")
        )

        action = model.step(obs, lang_annotation)
        frame = {
            "actions": action.detach().cpu().numpy(),
            "rel_actions": None,
            "robot_obs": obs["raw_obs"]["robot_obs"],
            "scene_obs": obs["raw_obs"]["scene_obs"],
            "rgb_static": obs["raw_obs"]["rgb_obs"]["rgb_static"],
            "rgb_gripper": obs["raw_obs"]["rgb_obs"]["rgb_gripper"],
            "rgb_tactile": obs["raw_obs"]["rgb_obs"]["rgb_tactile"],
            "depth_static": obs["raw_obs"]["depth_obs"]["depth_static"],
            "depth_gripper": obs["raw_obs"]["depth_obs"]["depth_gripper"],
            "depth_tactile": obs["raw_obs"]["depth_obs"]["depth_tactile"],
        }
        frames.append(frame)
        obs, _, _, current_info = env.step(action)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if len(current_task_info) > 0:
            idx = frame_idx
            for frame in frames:
                idx += 1
                # Save the frame
                frame_path = os.path.join(
                    saving_path,
                    f"episode_{idx:07d}.npz",
                )
                np.savez(frame_path, **frame)
            return True, step, (frame_idx + 1, idx), lang_annotation

    return False, step, None
