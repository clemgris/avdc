import argparse
import os
import sys
from pathlib import Path

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

import torch
from goal_diffusion import GoalGaussianDiffusion, Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset
from torchvision import utils
from transformers import CLIPTextModel, CLIPTokenizer
from unet import UnetMW as Unet

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from lorel.expert_dataset import ExpertDataset, ExpertTrainDataset  # noqa: E402, F401

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Total GPUs available: {torch.cuda.device_count()}")


def main(args):
    valid_n = 1
    sample_per_seq = 2  # 10

    results_folder = "../results_single/lorel"

    cfg = DictConfig(
        {
            "root": "/home/grislain/SkillDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_1k/data_with_dino_features",  # "/home/grislain/SkillDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_1k.pkl",  # "/lustre/fsn1/projects/rech/fch/uxv44vw/TrajectoryDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_50k.pkl",
            "skip_frames": 10,  # 2
            "diffuse_on": "pixel",
            "num_data": 100,  # 38225,
        },
    )

    if cfg.diffuse_on == "dino_feat":
        target_size = (16, 16)
    else:
        target_size = (64, 64)

    results_folder = Path(results_folder)

    if args.mode == "train":
        if os.path.exists(results_folder):
            if not args.override:
                raise ValueError(
                    f"Results folder {results_folder} already exists. Use --override to overwrite."
                )
        results_folder.mkdir(exist_ok=True, parents=True)

    with open(os.path.join(results_folder, "data_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(cfg))

    if args.mode == "inference":
        train_set = valid_set = [None]  # dummy
    else:
        # train_set = ExpertDataset(
        #     cfg.root,
        #     num_trajectories=cfg.num_data,
        #     use_state=False,
        #     normalize_states=False,
        #     skip_frames=cfg.skip_frames,
        # )
        train_set = ExpertTrainDataset(
            cfg.root, skip_frames=cfg.skip_frames, diffuse_on=cfg.diffuse_on
        )

        # Split train and valid
        valid_inds = [i for i in range(0, len(train_set), len(train_set) // valid_n)][
            :valid_n
        ]
        valid_set = Subset(train_set, valid_inds)

        # Remove valide from train
        all_inds = set(range(len(train_set)))
        train_inds = list(all_inds - set(valid_inds))
        train_set = Subset(train_set, train_inds)

        print("Train data:", len(train_set))
        print("Valid data:", len(valid_set))

        ## DEBUG
        # import torchvision

        # for idx in range(len(train_set)):
        #     x, x_cond, task = train_set[idx]
        #     torchvision.utils.save_image(
        #         x.reshape((7, 3, 96, 96)), f"train_img_{idx}_{task}.png"
        #     )
        #     if idx > 10: break

        # for idx in range(len(valid_set)):
        #     x, x_cond, task = valid_set[idx]
        #     torchvision.utils.save_image(
        #         x.reshape((7, 3, 96, 96)), f"valid_img_{idx}_{task}.png"
        #     )
        #     if idx > 10: break

    # breakpoint()
    unet = Unet()

    pretrained_model = (
        "openai/clip-vit-base-patch32"
        # "/lustre/fsmisc/dataset/HuggingFace_Models/openai/clip-vit-base-patch32"
    )
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusion(
        channels=3 * (sample_per_seq - 1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type="l2",
        objective="pred_v",
        beta_schedule="cosine",
        min_snr_loss_weight=True,
        auto_normalize=True,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=1e-4,
        train_num_steps=60000,
        save_and_sample_every=2500,
        ema_update_every=10,
        ema_decay=0.999,
        train_batch_size=16,
        valid_batch_size=32,
        gradient_accumulate_every=1,
        num_samples=valid_n,
        results_folder=results_folder,
        fp16=True,
        amp=True,
        calculate_fid=False,
    )

    if args.checkpoint_num is not None:
        trainer.load(args.checkpoint_num)

    if args.mode == "train":
        trainer.train()
    else:
        import imageio
        import torch
        from PIL import Image
        from torchvision import transforms

        text = args.text
        os.makedirs(
            str(results_folder / f"test_imgs / outputs / {text.replace(' ', '_')}"),
            exist_ok=True,
        )

        guidance_weight = args.guidance_weight
        image = Image.open(args.inference_path)
        image.save(str(results_folder / "test_imgs / test_img.png"))

        batch_size = 1
        transform = transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.ToTensor(),
            ]
        )
        image = transform(image)
        for i in range(args.num_samples):
            output = trainer.sample(
                image.unsqueeze(0), [text], batch_size, guidance_weight
            ).cpu()
            output = output[0].reshape(-1, 3, *target_size)
            output = torch.cat([image.unsqueeze(0), output], dim=0)
            utils.save_image(
                output,
                os.path.join(
                    str(
                        results_folder
                        / f"test_imgs / outputs / {text.replace(' ', '_')}"
                    ),
                    f"{text.replace(' ', '_')}_sample-{i}.png",
                ),
                nrow=sample_per_seq,
            )
            output_gif = os.path.join(
                str(results_folder / f"test_imgs / outputs / {text.replace(' ', '_')}"),
                f"{text.replace(' ', '_')}_sample-{i}.gif",
            )
            output = (
                output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255
            ).astype("uint8")
            imageio.mimsave(output_gif, output, duration=200, loop=1000)
            print(f"Generated {output_gif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    parser.add_argument(
        "-num", "--num_samples", type=int, default=1
    )  # set to number of samples to generate
    parser.add_argument(
        "-m", "--mode", type=str, default="train", choices=["train", "inference"]
    )  # set to 'inference' to generate samples
    parser.add_argument(
        "-c", "--checkpoint_num", type=int, default=None
    )  # set to checkpoint number to resume training or generate samples
    parser.add_argument(
        "-p", "--inference_path", type=str, default=None
    )  # set to path to generate samples
    parser.add_argument(
        "-t", "--text", type=str, default=None
    )  # set to text to generate samples
    parser.add_argument(
        "-n", "--sample_steps", type=int, default=100
    )  # set to number of steps to sample
    parser.add_argument(
        "-g", "--guidance_weight", type=int, default=0
    )  # set to positive to use guidance
    args = parser.parse_args()
    if args.mode == "inference":
        assert args.checkpoint_num is not None
        assert args.inference_path is not None
        assert args.text is not None
        assert args.sample_steps <= 100
    main(args)
