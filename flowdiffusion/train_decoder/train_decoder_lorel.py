import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T  # noqa: F401
from torch.utils.data import Subset

from flowdiffusion.decoder import TransposedConvDecoder  # noqa: E402
from flowdiffusion.encoder import DinoV2Encoder  # noqa: E402

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import argparse

import numpy as np
import torchvision
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from lorel.expert_dataset import ImageDataset  # noqa: E402


def main(args):
    target_size = (64, 64)

    results_folder = "../results_decoder/lorel"
    results_folder = Path(results_folder)

    cfg = DictConfig(
        {
            "root": "/home/grislain/SkillDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_1k.pkl",  # "/lustre/fsn1/projects/rech/fch/uxv44vw/TrajectoryDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_50k.pkl",
            "num_data": 10,  # 38225,
        },
    )

    training_cfg = DictConfig(
        {
            "eval_every": 1,
            "num_epochs": 10,
            "batch_size": 2,
            "lr": 1e-3,
            "save_every": 1,
            "num_valid": 10,
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

    train_set = ImageDataset(cfg.root, num_trajectories=cfg.num_data, use_state=False)

    # Split train and valid
    valid_inds = [
        i for i in range(0, len(train_set), len(train_set) // training_cfg.num_valid)
    ][: training_cfg.num_valid]
    valid_set = Subset(train_set, valid_inds)

    # Remove valide from train
    all_inds = set(range(len(train_set)))
    train_inds = list(all_inds - set(valid_inds))
    train_set = Subset(train_set, train_inds)

    print("Train data:", len(train_set))
    print("Valid data:", len(valid_set))

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Frozen encoder model
    encoder_model = DinoV2Encoder()

    # Decoder model
    decoder_model = TransposedConvDecoder(
        emb_dim=encoder_model.emb_dim,
        observation_shape=(3, target_size[0], target_size[0]),
        patch_size=16,
    )
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
        for image in tqdm(
            train_loader, desc=f"Epoch {epoch} / {training_cfg.num_epochs}"
        ):
            cls_emb, patch_emb = encoder_model(image)
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
            for image in tqdm(valid_loader, desc=f"Eval Epoch {epoch}"):
                cls_emb, patch_emb = encoder_model(image)
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
            past_model_path = (
                results_folder / f"decoder_model_{epoch - training_cfg.save_every}.pth"
            )
            if os.path.exists(past_model_path):
                os.remove(past_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    args = parser.parse_args()
    main(args)
