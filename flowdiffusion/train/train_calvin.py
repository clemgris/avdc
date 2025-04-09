import argparse
import os
import sys
from pathlib import Path

import torch
from einops import rearrange

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

from goal_diffusion import GoalGaussianDiffusion, Trainer
from omegaconf import DictConfig, OmegaConf
from torchvision import utils
from transformers import CLIPTextModel, CLIPTokenizer
from unet import UnetMW as Unet

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


def main(args):
    results_folder = args.results_folder
    data_path = args.data_path

    cfg = DictConfig(
        {
            "root": data_path,
            "datamodule": {
                "lang_dataset": {
                    "_target_": "calvin_agent.datasets.disk_dataset.DiskDiffusionDataset",
                    "key": "lang",
                    "save_format": "npz",
                    "batch_size": 16,
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
                    "num_subgoals": args.num_subgoals,
                    "pad": True,
                    "lang_folder": "lang_annotations",
                    "num_workers": 2,
                    "diffuse_on": args.diffuse_on,
                    "norm_dino_feat": args.diffuse_on == "dino_feat",
                },
            },
            "train_num_steps": args.train_num_steps,
            "save_and_sample_every": 2500,
        }
    )

    print("Config:", cfg)

    sample_per_seq = cfg.datamodule.lang_dataset.num_subgoals + 1

    if cfg.datamodule.lang_dataset.diffuse_on == "dino_feat":
        target_size = (16, 16)
        channel = 768
    elif cfg.datamodule.lang_dataset.diffuse_on == "pixel":
        target_size = (96, 96)
        channel = 3
    else:
        raise ValueError(
            f"Diffusion type {cfg.datamodule.lang_dataset.diffuse_on} not supported."
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
    if args.mode == "train":
        data_module.setup()
    results_folder = Path(results_folder)

    if args.mode == "train":
        if os.path.exists(results_folder):
            if not args.override:
                raise ValueError(
                    f"Results folder {results_folder} already exists. Use --override to overwrite."
                )
        results_folder.mkdir(exist_ok=True, parents=True)
        print("Results folder:", results_folder)

    with open(os.path.join(results_folder, "data_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(cfg))

    if args.mode == "inference":
        train_set = valid_set = [None]  # dummy
        valid_n = 0
    else:
        train_set = data_module.train_datasets["lang"]
        valid_set = data_module.val_datasets["lang"]
        valid_n = 1

        print("Train data:", len(train_set))
        print("Valid data:", len(valid_set))

        ## DEBUG
        # import torchvision

        # for idx in range(len(train_set)):
        #     x, x_cond, task = train_set[idx]
        #     torchvision.utils.save_image(
        #         x.reshape((1, 3, 96, 96)), f"train_img_{idx}_{task}.png"
        #     )
        #     if idx > 10:
        #         break

        # for idx in range(len(valid_set)):
        #     x, x_cond, task = valid_set[idx]
        #     torchvision.utils.save_image(
        #         x.reshape((7, 3, 96, 96)), f"valid_img_{idx}_{task}.png"
        #     )
        #     if idx > 10: break
        # breakpoint()

    unet = Unet(channel)

    if args.server == "jz":
        text_pretrained_model = (
            "/lustre/fsmisc/dataset/HuggingFace_Models/openai/clip-vit-base-patch32"
        )
    else:
        text_pretrained_model = "openai/clip-vit-base-patch32"

    tokenizer = CLIPTokenizer.from_pretrained(text_pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(text_pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    decoder_weigth_path = args.feature_decoder_checkpoint_path

    if cfg.datamodule.lang_dataset.diffuse_on == "dino_feat":
        import torch
        from decoder import TransposedConvDecoder

        decoder_model = TransposedConvDecoder(
            emb_dim=768,
            observation_shape=(3, 96, 96),
            patch_size=16,
        )
        state_dict = torch.load(decoder_weigth_path)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        decoder_model.load_state_dict(state_dict)
        decoder_model = decoder_model.to(device)
        decoder_model.eval()

    else:
        decoder_model = None

    diffusion = GoalGaussianDiffusion(
        channels=channel * (sample_per_seq - 1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type="l2",
        objective="pred_v",
        beta_schedule="cosine",
        min_snr_loss_weight=True,
        auto_normalize=False,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        feature_decoder=decoder_model,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=1e-4,
        train_num_steps=cfg.train_num_steps,
        save_and_sample_every=cfg.save_and_sample_every,
        ema_update_every=10,
        ema_decay=0.999,
        train_batch_size=cfg.datamodule.lang_dataset.batch_size,
        valid_batch_size=1,
        gradient_accumulate_every=1,
        num_samples=valid_n,
        results_folder=results_folder,
        fp16=True,
        amp=True,
        calculate_fid=False,
        dino_stats_path=os.path.join(cfg.root, "dino_stats.pt"),
        norm_feat=cfg.datamodule.lang_dataset.norm_dino_feat,
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

        val_annotations = OmegaConf.load(
            os.path.join(
                root_path,
                "calvin/calvin_models/conf/annotations/new_playtable_validation.yaml",
            )
        )

        text = text.relace(" ", "_")
        text_ann = val_annotations[text][0]

        # Save text annotation
        with open(
            str(results_folder / f"test_imgs / {text.replace(' ', '_')}.txt"), "w"
        ) as file:
            file.write(text_ann)
        print(f"Text annotation: {text_ann}")

        guidance_weight = args.guidance_weight
        image = Image.open(args.inference_path).convert("RGB")
        image.save(str(results_folder / "test_imgs / test_img.png"))

        batch_size = 1
        if cfg.datamodule.lang_dataset.diffuse_on == "pixel":
            transform = transforms.Compose(
                [
                    transforms.Resize(target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
            image = transform(image)
            for i in range(args.num_samples):
                output = trainer.sample(
                    image.unsqueeze(0), [text_ann], batch_size, guidance_weight
                ).cpu()

                # Unnormalize
                output = (output + 1) / 2
                output = output[0].reshape(-1, 3, *target_size)

                # Save output
                output = torch.cat([(image[None] + 1) / 2, output], dim=0)
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
                    str(
                        results_folder
                        / f"test_imgs / outputs / {text.replace(' ', '_')}"
                    ),
                    f"{text.replace(' ', '_')}_sample-{i}.gif",
                )
                output = (
                    output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255
                ).astype("uint8")
                imageio.mimsave(output_gif, output, duration=200, loop=1000)
                print(f"Generated {output_gif}")

        elif cfg.datamodule.lang_dataset.diffuse_on == "dino_feat":
            from decoder import TransposedConvDecoder
            from encoder import DinoV2Encoder

            transform = transforms.Compose(
                [
                    transforms.Resize(target_size),
                    transforms.ToTensor(),
                ]
            )

            image = transform(image)

            if args.server == "jz":
                encoder_name = (
                    "/lustre/fsn1/projects/rech/fch/uxv44vw/facebook/dinov2-base",
                )
            else:
                encoder_name = "facebook/dinov2-base"

            encoder_model = DinoV2Encoder(name=encoder_name)

            # Load statistics
            dino_stats_path = os.path.join(cfg.root, "dino_stats.pt")
            dino_stats = torch.load(dino_stats_path)["dino_features"]
            # Generate samples
            _, init_feat = encoder_model(image[None, ...].to(device))

            if cfg.datamodule.lang_dataset.norm_dino_feat:
                # Normalise init_feat
                init_feat = (init_feat - dino_stats["min"]) / (
                    dino_stats["max"] - dino_stats["min"]
                )
                init_feat = init_feat * 2 - 1
            init_feat = rearrange(init_feat, "b (h w) c -> b c h w ", w=16, h=16)
            for i in range(args.num_samples):
                output = trainer.sample(
                    init_feat, [text], batch_size, guidance_weight
                ).cpu()
                output = rearrange(
                    output,
                    "b (n c) h w -> b n (h w) c",
                    w=16,
                    h=16,
                    n=(sample_per_seq - 1),
                )
                if cfg.datamodule.lang_dataset.norm_dino_feat:
                    # Unnormalize
                    output = (output + 1) / 2
                    output = (
                        output * (dino_stats["max"] - dino_stats["min"])
                        + dino_stats["min"]
                    )

                output = trainer.feature_decoder(output.to(device)).detach().cpu()

                # Save output
                output = torch.cat([image[None], output], dim=0)
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
                    str(
                        results_folder
                        / f"test_imgs / outputs / {text.replace(' ', '_')}"
                    ),
                    f"{text.replace(' ', '_')}_sample-{i}.gif",
                )
                output = (
                    output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255
                ).astype("uint8")
                imageio.mimsave(output_gif, output, duration=200, loop=1000)
                print(f"Generated {output_gif}")
        else:
            raise ValueError(
                f"Diffusion type {cfg.datamodule.lang_dataset.diffuse_on} not supported."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--server", type=str, default="hacienda", choices=["jz", "local"]
    )  # set to 'jz' to use jean zay server
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
    )  # set to get initial image
    parser.add_argument(
        "-t", "--text", type=str, default=None
    )  # set to text to generate samples
    parser.add_argument(
        "-n", "--sample_steps", type=int, default=100
    )  # set to number of steps to sample
    parser.add_argument(
        "-g", "--guidance_weight", type=int, default=0
    )  # set to positive to use guidance
    parser.add_argument(
        "--diffuse_on", type=str, default="pixel", choices=["pixel", "dino_feat"]
    )  # set to 'pixel' or 'dino_feat' to diffuse on pixel or dino features
    parser.add_argument(
        "--num_subgoals", type=int, default=8
    )  # set to number of subgoals
    parser.add_argument(
        "-r", "--results_folder", type=str, default="../results_huit_ann/calvin"
    )  # set to results folder
    parser.add_argument(
        "--feature_decoder_checkpoint_path",
        type=str,
        default="/home/grislain/AVDC/calvin/models/decoder/decoder_model_48.pth",
    )  # set to decoder checkpoint path
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset",
    )  # set to data path
    parser.add_argument(
        "--train_num_steps", type=int, default=150000
    )  # set to number of training steps
    args = parser.parse_args()

    if args.diffuse_on == "diffuse_on":
        assert args.feature_decoder_checkpoint_path is not None
    if args.mode == "inference":
        assert args.checkpoint_num is not None
        assert args.inference_path is not None
        assert args.text is not None
        assert args.sample_steps <= 100
    main(args)
