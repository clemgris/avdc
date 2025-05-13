import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)
import torch
from evaluator import TaskCompletionClassifier  # noqa: E402, F401
from omegaconf import DictConfig, OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer

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


def main(args):
    results_folder = Path(args.results_folder)

    data_path = args.data_path

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
                        "rgb_obs": ["rgb_static", "rgb_gripper"]
                        if args.use_gripper
                        else ["rgb_static"],
                        "depth_obs": (
                            ["depth_static"]
                            if (args.use_depth and not args.use_gripper)
                            else ["depth_static", "depth_gripper"]
                            if (args.use_depth and args.use_gripper)
                            else []
                        ),
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
    results_folder = Path(results_folder)

    if os.path.exists(results_folder):
        if not args.override:
            raise ValueError(
                f"Results folder {results_folder} already exists. Use --override to overwrite."
            )
    results_folder.mkdir(exist_ok=True, parents=True)

    train_set = data_module.train_datasets["lang"]
    valid_set = data_module.val_datasets["lang"]

    print("Train data:", len(train_set))
    print("Valid data:", len(valid_set))

    num_epochs = 500
    device = torch.device("cuda")
    log_freq = 1

    if args.server == "jz":
        pretrained_model = (
            "/lustre/fsmisc/dataset/HuggingFace_Models/openai/clip-vit-base-patch32"
        )
    else:
        pretrained_model = "openai/clip-vit-base-patch32"

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)

    # Freeze text encoder
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    evaluator = TaskCompletionClassifier(
        text_encoder=text_encoder, text_tokenizer=tokenizer
    )
    evaluator = evaluator.to(device)

    # Number of parameters
    print(
        "Number of parameters:",
        sum(p.numel() for p in evaluator.parameters()),
    )
    print(
        "Number of trainable parameters:",
        sum(p.numel() for p in evaluator.parameters() if p.requires_grad),
    )

    optimizer = torch.optim.Adam(evaluator.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        valid_set,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    # Training loop.
    for epoch in range(num_epochs):
        all_losses = []
        all_metric = []
        for batch in tqdm(train_dataloader, desc="Training", leave=False):
            episode, text_task, sucess = batch
            breakpoint()
            pred_loggits = evaluator(episode.to(device), text_task)

            # Loss
            loss = criterion(pred_loggits, sucess.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            all_losses.append(loss.item())

            # Metric
            pred_sucess = torch.argmax(pred_loggits, dim=1)
            metric = torch.mean((pred_sucess == sucess.to(device)).float())
            all_metric.append(metric.item())

        if epoch % log_freq == 0:
            print(
                f"Train | epoch: {epoch} loss: {np.mean(all_losses):.3f} metric: {np.mean(all_metric):.3f}"
            )

        if epoch % cfg.save_every == 0:
            # Save model
            saving_path = os.path.join(
                results_folder, f"model-{epoch // cfg.save_every}.pt"
            )
            torch.save(evaluator.state_dict(), saving_path)

        if epoch % cfg.eval_every == 0:
            evaluator.eval()
            all_eval_losses = []
            all_eval_metric = []
            for batch in val_dataloader:
                episode, text_task, sucess = batch
                pred_loggits = evaluator(episode.to(device), text_task)

                # Loss
                loss = criterion(pred_loggits, sucess.to(device))
                all_eval_losses.append(loss.item())

                # Metric
                pred_sucess = torch.argmax(pred_loggits, dim=1)
                metric = torch.mean((pred_sucess == sucess.to(device)).float())
                all_eval_metric.append(metric.item())

            print(
                f"Eval | epoch: {epoch} eval loss: {np.mean(all_eval_losses):.3f} eval metric: {np.mean(all_eval_metric):.3f}"
            )


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
    parser.add_argument(
        "-r", "--results_folder", type=str, default="../results_evaluator_debug/calvin"
    )  # set to results folder to resume training or generate samples
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset",
    )  # set to data path to resume training or generate samples
    parser.add_argument(
        "--batch_size", type=int, default=64
    )  # set to batch size to resume training or generate samples
    parser.add_argument("--use_depth", action="store_true")  # use depth images
    parser.add_argument("--use_gripper", action="store_true")  # use gripper images
    args = parser.parse_args()
    main(args)
