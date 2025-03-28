import argparse
import os
import sys
from collections import Counter

import PIL.Image as Image
import torch
from transformers import (
    pipeline,
)

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)
from omegaconf import DictConfig, OmegaConf

sys.path.append(
    os.path.join(
        root_path,
        "calvin/calvin_models",
    )
)

from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,  # noqa: E402
)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Total GPUs available: {torch.cuda.device_count()}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def main(args):
    if args.server == "jz":
        data_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/CALVIN/task_D_D"
        vlm_path = "/lustre/fsmisc/dataset/HuggingFace_Models/llava-hf/llava-v1.6-mistral-7b-hf"
        llm_path = (
            "/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.2-3B-Instruct"
        )
    else:
        data_path = "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset"
        vlm_path = "llava-hf/llava-v1.6-mistral-7b-hf"
        llm_path = "meta-llama/Llama-3.2-3B-Instruct"

    cfg = DictConfig(
        {
            "root": data_path,
            "datamodule": {
                "lang_dataset": {
                    "_target_": "calvin_agent.datasets.disk_dataset.DiskEvaluatorDataset",
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
                    "num_subgoals": 8,
                    "pad": True,
                    "lang_folder": "lang_annotations",
                    "num_workers": 2,
                    "diffuse_on": "pixel",
                },
            },
            "save_every": 10,  # In epochs
            "eval_every": 5,  # In epochs
        }
    )

    transforms = OmegaConf.load(
        os.path.join(
            root_path,
            "calvin/calvin_models/conf/datamodule/transforms/play_basic.yaml",
        )
    )

    data_module = CalvinDataModule(
        cfg.datamodule, transforms=transforms, root_data_dir=cfg.root
    )

    data_module.setup()

    train_set = data_module.train_datasets["lang"]
    valid_set = data_module.val_datasets["lang"]

    print("Train data:", len(train_set))
    print("Valid data:", len(valid_set))

    results = Counter()
    num_unanswered = Counter()
    num_tasks = Counter()

    vlm_pipe = pipeline(
        "image-text-to-text",
        model=vlm_path,
    )

    llm_pipe = pipeline(
        "text-generation",
        model=llm_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    for init, target, task, label in train_set:
        print("Task", task)

        # Preprocess image
        init = (init + 1) / 2
        init = init.clip(0, 1).numpy()
        init = Image.fromarray((init * 255).astype("uint8").transpose(1, 2, 0))

        target = (target + 1) / 2
        target = target.clip(0, 1).numpy()
        target = Image.fromarray((target * 255).astype("uint8").transpose(1, 2, 0))

        # VLM - Describe the difference between the first and last images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": init},
                    {"type": "image", "image": target},
                    {
                        "type": "text",
                        "text": "Describe in detail how the scene has changed between the two images?",
                    },
                ],
            },
        ]
        outputs = vlm_pipe(text=messages, generate_kwargs={"max_new_tokens": 200})

        print(outputs[0]["generated_text"])
        parsed_answer = outputs[0]["generated_text"]

        # LLM - Determine if the task is accomplished
        messages = [
            {
                "role": "system",
                "content": "You are a task accomplishement assessor. You are asked to determine if a task (like 'open the drawer' or 'light the red block') is accomplished during a trajectory based on the description of the differences bteween the first and the last images.",
            },
            {
                "role": "user",
                "content": f"Here is the description of the difference between the first and last frame: {parsed_answer}. \n Is the task {task} accomplished? (answer by yes or no)",
            },
        ]

        outputs = llm_pipe(
            messages,
            max_new_tokens=256,
        )
        print(outputs[0]["generated_text"][-1], label)

        if ("no" in outputs[0]["generated_text"][-1]) or (
            "No" in outputs[0]["generated_text"][-1]
        ):
            pred = 0
        elif ("yes" in outputs[0]["generated_text"][-1]) or (
            "Yes" in outputs[0]["generated_text"][-1]
        ):
            pred = 1
        else:
            pred = -1
            print("No answer found")

        if pred == -1:
            num_unanswered[task] += 1
            continue
        results[task] += pred == label
        num_tasks[task] += 1

    # Save results
    print("-" * 50)

    print("SR")
    for task in results:
        print(task, results[task] / num_tasks[task])

    print("Unanswered")
    for task in num_unanswered:
        print(task, num_unanswered[task] / num_tasks[task])

    # with open("results.json", "w") as f:
    #     json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--server", type=str, default="hacienda"
    )  # set to 'jz' to run on jean zay server
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    parser.add_argument(
        "-c", "--checkpoint_num", type=int, default=None
    )  # set to checkpoint number to resume training or generate samples

    args = parser.parse_args()
    main(args)
