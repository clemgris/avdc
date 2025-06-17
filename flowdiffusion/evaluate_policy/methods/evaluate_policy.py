# === Standard Library ===
import logging
import os
import sys
from collections import Counter
from pathlib import Path

# === Third-party Libraries ===
import hydra
import numpy as np
from omegaconf import OmegaConf

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
    get_sequences,
)
from calvin.calvin_models.calvin_agent.evaluation.utils import (
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
from methods.rollout import rollout, rollout_with_oracle

# === Logger ===
logger = logging.getLogger(__name__)


NUM_SEQUENCES = 1000


def evaluate_policy_singlestep(model, env, dataset, args, conf_dir):
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)

    dataset = dataset

    results = Counter()
    tot_tasks = Counter()

    for episode in dataset:
        task = episode["task"]
        success, length = rollout_with_oracle(
            env, model, episode, task_oracle, args, task
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


def evaluate_policy(
    model,
    env,
    epoch=0,
    eval_folder=None,
    debug_path=None,
    conf_dir=None,
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
    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml"
    )

    eval_folder = get_log_dir(eval_folder)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            debug_path,
        )
        results.append(result)

    print_and_save(results, eval_sequences, eval_folder, epoch)

    return results


def evaluate_sequence(
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    debug_path,
):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    for subtask in eval_sequence:
        success, length = rollout(
            env, model, task_checker, subtask, val_annotations, debug_path
        )
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def evaluate_policy_singlestep_and_save(
    model, env, dataset, args, conf_dir, prob_sucess=0.8, num_trial=5
):
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
    for episode, task, ann, (start_idx, end_idx) in dataset:
        sucesses = 0
        for __ in range(num_trial):
            success, length = rollout_with_oracle(
                env, model, episode, task_oracle, args, task
            )
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
