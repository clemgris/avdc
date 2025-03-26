import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T  # noqa: F401
from omegaconf import DictConfig, OmegaConf
from torch.nn import DataParallel
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(root_path)
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

from decoder import TransposedConvDecoder  # noqa: E402

sys.path.append(
    os.path.join(
        root_path,
        "calvin/calvin_models",
    )
)

from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,  # noqa: E402
)

# Print number of GPUs available
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    target_size = (96, 96)

    results_folder = "../results_decoder_debug/calvin"
    results_folder = Path(results_folder)

    if args.server == "jz":
        data_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/CALVIN/task_D_D"
    else:
        data_path = "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset"

    cfg = DictConfig(
        {
            "root": data_path,
            "datamodule": {
                "lang_dataset": {
                    "_target_": "calvin_agent.datasets.disk_dataset.DiskImageDataset",
                    "key": "lang",
                    "save_format": "npz",
                    "batch_size": 32,
                    "min_window_size": 16,
                    "max_window_size": 64,
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
                    "pad": False,
                    "lang_folder": "lang_annotations",
                    "num_workers": 2,
                    "diffuse_on": "dino_feat",
                },
            },
        }
    )

    transforms = OmegaConf.load(
        os.path.join(
            root_path,
            "calvin/calvin_models/conf/datamodule/transforms/play_decoder.yaml",
        )
    )

    training_cfg = DictConfig(
        {
            "eval_every": 1,
            "num_epochs": 20,
            "batch_size": 256,
            "lr": 1e-3,
            "save_every": 1,
        },
    )

    if os.path.exists(results_folder):
        if not args.override:
            raise ValueError(
                f"Results folder {results_folder} already exists. Use --override to overwrite."
            )
    results_folder.mkdir(exist_ok=True, parents=True)

    with open(os.path.join(results_folder, "training_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(cfg))

    results_folder = Path(results_folder)

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

    with open(os.path.join(results_folder, "data_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(cfg))

    train_set = data_module.train_datasets["lang"]
    valid_set = data_module.val_datasets["lang"]

    print("Train data:", len(train_set))
    print("Valid data:", len(valid_set))

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Decoder model
    decoder_model = TransposedConvDecoder(
        emb_dim=768,
        observation_shape=(3, target_size[0], target_size[0]),
        patch_size=16,
    )

    # Print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in decoder_model.parameters())}")

    decoder_model = DataParallel(decoder_model)
    decoder_model = decoder_model.to(device)
    decoder_model.train()

    optimizer = torch.optim.Adam(
        decoder_model.parameters(),
        lr=training_cfg.lr,
    )
    loss = torch.nn.MSELoss()

    for epoch in range(training_cfg.num_epochs):
        # Training loop
        decoder_model.train()
        all_losses = []
        for data in tqdm(
            train_loader, desc=f"Epoch {epoch} / {training_cfg.num_epochs}"
        ):
            image, patch_emb = data
            patch_emb = patch_emb.to(device)
            image = image.to(device)
            rec_image = decoder_model(patch_emb)

            optimizer.zero_grad()
            loss_value = loss(rec_image, image)

            loss_value.backward()
            optimizer.step()

            all_losses.append(loss_value.item())

        print(f"Epoch : {epoch} | loss | {np.mean(all_losses)}")

        # Evaluation loop
        if epoch % training_cfg.eval_every == 0:
            decoder_model.eval()
            all_eval_losses = []
            for data in tqdm(valid_loader, desc=f"Eval Epoch {epoch}"):
                image, patch_emb = data
                patch_emb = patch_emb.to(device)
                image = image.to(device)

                with torch.no_grad():
                    rec_image = decoder_model(patch_emb)

                loss_value = loss(rec_image, image)
                all_eval_losses.append(loss_value.item())

            print(f"Epoch : {epoch} | eval_loss | {np.mean(all_eval_losses)}")

            # Save results
            torchvision.utils.save_image(
                rec_image,
                results_folder / f"rec_image_{epoch}.png",
            )
            torchvision.utils.save_image(
                image,
                results_folder / f"image_{epoch}.png",
            )

        # Save model
        if epoch % training_cfg.save_every == 0:
            torch.save(
                decoder_model.state_dict(),
                results_folder / f"decoder_model_{epoch}.pth",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server", type=str, default="hacienda"
    )  # set to "jz" to run on jean zay server
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    args = parser.parse_args()
    main(args)
