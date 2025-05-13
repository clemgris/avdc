import glob
import os

import torch
import torchvision.utils as utils
from einops import rearrange


def get_paths(root="../berkeley"):
    f = []
    for dirpath, dirname, filename in os.walk(root):
        if "image" in dirpath:
            f.append(dirpath)
    print(f"Found {len(f)} sequences")
    return f


def get_paths_from_dir(dir_path):
    paths = glob.glob(os.path.join(dir_path, "im*.jpg"))
    try:
        paths = sorted(paths, key=lambda x: int((x.split("/")[-1].split(".")[0])[3:]))
    except:
        print(paths)
    return paths


def assert_configs_equal(cfg1, cfg2, list_exception, path=""):
    if isinstance(cfg1, dict) and isinstance(cfg2, dict):
        assert cfg1.keys() == cfg2.keys(), (
            f"\n Key mismatch at {path}: {cfg1.keys()} vs {cfg2.keys()}"
        )
        for k in cfg1:
            new_path = f"{path}.{k}" if path else k
            if new_path in list_exception:
                continue
            assert_configs_equal(cfg1[k], cfg2[k], list_exception, new_path)
    elif isinstance(cfg1, list) and isinstance(cfg2, list):
        assert len(cfg1) == len(cfg2), f"List length mismatch at {path}"
        for i, (v1, v2) in enumerate(zip(cfg1, cfg2)):
            new_path = f"{path}[{i}]"
            if new_path in list_exception:
                continue
            assert_configs_equal(v1, v2, list_exception, new_path)
    else:
        assert cfg1 == cfg2, f"Value mismatch at {path}: {cfg1} != {cfg2}"


def save_images(img, path: str, nrow: int = 1):
    if img.shape[1] == 3:  # RGB image
        utils.save_image(img, path, nrow=nrow)
    elif img.shape[1] == 4:  # RGB-D image
        rgb_image = img[:, :3, :, :]
        depth_image = img[:, 3:, :, :].expand(-1, 3, -1, -1)
        img = torch.cat((rgb_image, depth_image), dim=1)
        img = rearrange(img, "b (x c) h w -> (x b) c h w", x=2)
        utils.save_image(img, path, nrow=nrow)
    elif img.shape[1] == 6:  # RGB static + ego views
        img = rearrange(img, "b (x c) h w -> (x b) c h w", c=3)
        utils.save_image(img, path, nrow=nrow)
    elif img.shape[1] == 8:  # RGB-D static + ego views
        rgb_image = img[:, :6, :, :]
        static_depth_image = img[:, 6:7, :, :].expand(-1, 3, -1, -1)
        gripper_depth_image = img[:, 7:8, :, :].expand(-1, 3, -1, -1)
        img = torch.cat((rgb_image, static_depth_image, gripper_depth_image), dim=1)
        img = rearrange(img, "b (x c) h w -> (x b) c h w", c=3)
        utils.save_image(img, path, nrow=nrow)
