import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

from encoder import DinoV2Encoder  # noqa: E402

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import argparse

from lorel.expert_dataset import ExpertDataset, TrajDataset  # noqa: E402, F401


def main(args):
    cfg = DictConfig(
        {
            "root": "/home/grislain/SkillDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_1k.pkl",  # "/lustre/fsn1/projects/rech/fch/uxv44vw/TrajectoryDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_50k.pkl",
            "num_data": 100,  # 38225,
            "skip_frames": 1,
        },
    )

    data_filename = Path(cfg.root).stem
    folder_name = Path(cfg.root).parent
    os.makedirs(
        folder_name / data_filename / f"data_with_{args.features}_features",
        exist_ok=True,
    )

    print("Data folder:", folder_name / data_filename / "features")

    train_set = TrajDataset(cfg.root, num_trajectories=cfg.num_data, use_state=False)
    # train_set = ExpertDataset(
    #     cfg.root,
    #     num_trajectories=cfg.num_data,
    #     use_state=False,
    #     normalize_states=False,
    #     skip_frames=cfg.skip_frames,
    # )

    print("Train data (traj):", len(train_set))

    # Frozen encoder model
    if args.features == "dino":
        encoder_model = DinoV2Encoder(
            name="facebook/dinov2-base",  # "/lustre/fsn1/projects/rech/fch/uxv44vw/facebook/dinov2-base"
        )
    else:
        raise ValueError(f"Unknown feature type {args.features}")

    for ii, data in tqdm(
        enumerate(train_set),
        desc=f"Generate {args.features} features of training data",
        total=len(train_set),
    ):
        traj = data.copy()
        images = traj["states"]
        all_cls_emb = []
        all_patch_emb = []
        for image in images:
            cls_emb, patch_emb = encoder_model(image)
            all_cls_emb.append(cls_emb.cpu().numpy())
            all_patch_emb.append(patch_emb.cpu().numpy())
        traj["dino_cls_emb"] = np.concatenate(all_cls_emb)
        traj["dino_patch_emb"] = np.concatenate(all_patch_emb)

        # Save as npz
        np.savez(
            folder_name
            / data_filename
            / f"data_with_{args.features}_features/data_{ii}.npz",
            **traj,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    parser.add_argument("-f", "--features", type=str, default="dino")
    args = parser.parse_args()
    main(args)
