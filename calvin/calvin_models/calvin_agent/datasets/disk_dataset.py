import logging
import os
import pickle
import random
import re
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from calvin_agent.datasets.base_dataset import BaseDataset
from calvin_agent.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_features,
    process_language,
    process_rgb,
    process_state,
)
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def lookup_naming_pattern(
    dataset_dir: Path, save_format: str
) -> Tuple[Tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class DiskDiffusionDataset(BaseDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        num_subgoals: Number of subgoals per episodes.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        num_subgoals: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.num_subgoals = num_subgoals
        if self.with_lang:
            self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_task = (
                self._build_file_indices_lang(self.abs_datasets_dir)
            )
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )

    def _get_episode_name(self, file_idx: int) -> Path:
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _get_feat_name(self, file_idx: int) -> Path:
        return Path(
            self.abs_datasets_dir
            / f"features_{self.diffuse_on}/features_{file_idx}.npz"
        )

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load num_goals frames of the episodes (plus the initial frame) evenly spaced.

        Args:
            idx: Index of first frame.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx, end_idx = self.episode_lookup[idx]

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        frames_idx = np.linspace(
            start_idx,
            end_idx,
            min(self.max_window_size, self.num_subgoals + 1),
            dtype=int,
        )
        frames = [
            self.load_file(self._get_episode_name(file_idx)) for file_idx in frames_idx
        ]

        episode = {key: np.stack([ep[key] for ep in frames]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            episode["task"] = self.lang_task[self.lang_lookup[idx]]
        if self.with_feat:
            features = [
                self.load_file(self._get_feat_name(file_idx))["patch_emb"]
                for file_idx in frames_idx
            ]
            episode["features"] = features
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language tasks.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        # Load lang data from pickle
        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are <=64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]  # length total number of annotations
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            assert end_idx >= self.max_window_size
            lang_lookup.append(i)
            episode_lookup.append((start_idx, end_idx))

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _get_sequences(self, idx: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx)

        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        seq_feat = process_features(
            episode, self.with_feat, self.feat_stats, self.norm_feat
        )
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
            **seq_feat,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.num_subgoals + 1 - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            # repeat action for world coordinates action space
            seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            seq_acts = torch.cat(
                [
                    self._pad_with_zeros(seq["actions"][..., :-1], pad_size),
                    self._pad_with_repetition(seq["actions"][..., -1:], pad_size),
                ],
                dim=-1,
            )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    def _pad_with_repetition(
        self, input_tensor: torch.Tensor, pad_size: int
    ) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(
            torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    def _pad_with_zeros(
        self, input_tensor: torch.Tensor, pad_size: int
    ) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """

        sequence = self._get_sequences(idx)
        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size)

        views = []
        for key in sequence["rgb_obs"].keys():
            views.append(sequence["rgb_obs"][key])
        for key in sequence["depth_obs"].keys():
            views.append(sequence["depth_obs"][key])
        images = torch.cat(views, dim=1)
        features = sequence["features"]

        if self.with_feat:
            x_cond = features[0][None, ...]
            x_cond = rearrange(x_cond, "f wh c -> f c wh")
            x_cond = rearrange(
                x_cond,
                "f c (w h) -> f c w h",
                w=self.feat_patch_size,
                h=self.feat_patch_size,
            )
            x_cond = x_cond.squeeze(0)

            x = features[1:].squeeze(1)
            x = rearrange(x, "f wh c -> f c wh")
            x = rearrange(
                x,
                "f c (w h) -> f c w h",
                w=self.feat_patch_size,
                h=self.feat_patch_size,
            )
            x = rearrange(x, "f c h w -> (f c) h w")
        else:
            x_cond = images[0, ...]
            x = images[1:, ...]
            x_cond = x_cond.squeeze(0)
            x = rearrange(x, "f c h w -> (f c) h w")
        task = sequence["lang"]
        return x, x_cond, task


class DiskImageDataset(BaseDataset):
    """
    Dataset that loads individual images from episodes while ensuring each image is seen once per epoch.

    Args:
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        save_format: str = "npz",
        pretrain: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError

        self.pretrain = pretrain

        if self.with_lang:
            self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_task = (
                self._build_file_indices_lang(self.abs_datasets_dir)
            )
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )

        self.frame_indices = self._generate_frame_indices()

    def _get_episode_name(self, file_idx: int) -> Path:
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _get_feat_name(self, file_idx: int) -> Path:
        return Path(
            self.abs_datasets_dir
            / f"features_{self.diffuse_on}/features_{file_idx}.npz"
        )

    def _generate_frame_indices(self) -> List[Tuple[int, int]]:
        """
        Generates a list of (episode_idx, frame_idx) pairs to ensure every image is seen once per epoch.
        """
        frames_idx = []
        for _, (start_idx, end_idx) in enumerate(self.episode_lookup):
            for j in range(start_idx, end_idx + 1):
                frames_idx.append(j)

        return frames_idx

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """
        Returns a single image from a trajectory.

        Args:
            index: Index in the precomputed list of (episode, frame) pairs.
        Returns:
            sample: Dict containing a single image frame.
        """
        frame_idx = self.frame_indices[index]

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")

        frame = self.load_file(self._get_episode_name(frame_idx))
        if self.with_feat:
            features = self.load_file(self._get_feat_name(frame_idx))

        sample = {key: frame[key] for key in keys}
        if self.with_feat:
            sample["features"] = features["patch_emb"]

        rgb_obs = process_rgb(sample, self.observation_space, self.transforms)
        features = process_features(
            sample, self.with_feat, self.feat_stats, self.norm_feat
        )
        sample = {**rgb_obs, **features}

        image = sample["rgb_obs"]["rgb_static"].squeeze(0)
        features = sample["features"].squeeze(0)
        return frame_idx, image, features

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        Builds mapping from index to file_name for loading individual images.

        Args:
            abs_datasets_dir: Absolute path of the dataset directory.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example index to language instruction.
            lang_ann: Language tasks.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        try:
            print(
                "Trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]
        lang_ann = lang_data["language"]["ann"]
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            assert end_idx >= self.max_window_size
            lang_lookup.append(i)
            episode_lookup.append((start_idx, end_idx))

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        count = 0
        for start, end in ep_start_end_ids:
            count += end - start + 1
        logger.info(f"Found {count} frames")
        return ep_start_end_ids


class RandomApplyTransform:
    def __init__(self, transform_fn, p=0.5):
        self.transform_fn = transform_fn
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.transform_fn(img)
        return img


class ComposeTensor:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        x = (x + 1) / 2
        for t in self.transforms:
            x = t(x).clamp(0, 1)
        norm_x = (x - 0.5) * 2
        return norm_x


def get_stochastic_augmentation(p=0.5):
    return ComposeTensor(
        [
            RandomApplyTransform(
                lambda x: F.adjust_brightness(x, random.uniform(0.8, 1.2)), p=p
            ),
            RandomApplyTransform(
                lambda x: F.adjust_contrast(x, random.uniform(0.8, 1.2)), p=p
            ),
            RandomApplyTransform(
                lambda x: F.adjust_saturation(x, random.uniform(0.8, 1.2)), p=p
            ),
            RandomApplyTransform(
                lambda x: F.adjust_hue(x, random.uniform(-0.1, 0.1)), p=p
            ),
        ]
    )


class DiskActionDataset(BaseDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        num_subgoals: Number of subgoals per episodes.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        num_subgoals: int = 1,
        prob_aug: float = 0.5,
        save_format: str = "npz",
        pretrain: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.num_subgoals = num_subgoals
        if self.with_lang:
            self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_task = (
                self._build_file_indices_lang(self.abs_datasets_dir)
            )
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )

        self.prob_data_aug = prob_aug
        self.with_lang = False

    def _get_episode_name(self, file_idx: int) -> Path:
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _get_feat_name(self, file_idx: int) -> Path:
        return Path(
            self.abs_datasets_dir
            / f"features_{self.diffuse_on}/features_{file_idx}.npz"
        )

    def __len__(self) -> int:
        return len(self.episode_lookup)

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load entire_episode.

        Args:
            idx: Index of first frame.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx, end_idx, j = self.episode_lookup[idx]

        num_frames = end_idx - start_idx + 1
        chunk_size = random.randint(
            int(np.ceil(self.min_window_size / self.num_subgoals)),
            int(np.ceil(num_frames / self.num_subgoals)),
        )

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")

        episodes_idx = np.arange(start_idx, end_idx + 1 - chunk_size + 1)

        # Pick randm frame from the episode except the last one
        frame_idx = episodes_idx[j]

        # Action idx are from the frame_idx to the next frame
        assert frame_idx + chunk_size <= end_idx + 1
        actions_idx = np.arange(frame_idx, frame_idx + chunk_size)

        episodes = [
            self.load_file(self._get_episode_name(file_idx)) for file_idx in actions_idx
        ]

        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_feat:
            features = [
                self.load_file(self._get_feat_name(file_idx))["patch_emb"]
                for file_idx in actions_idx
            ]
            episode["features"] = features
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language tasks.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        # Load lang data from pickle
        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are <=64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]  # length total number of annotations
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            assert end_idx >= self.max_window_size
            lang_lookup.append(i)
            num_frames = end_idx - start_idx + 1
            chunk_size = int(np.ceil(num_frames / self.num_subgoals))
            for j in range(0, max(1, num_frames - chunk_size)):
                episode_lookup.append((start_idx, end_idx, j))
        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            assert end_idx >= self.max_window_size
            num_frames = end_idx - start_idx + 1
            chunk_size = int(np.ceil(self.max_window_size / self.num_subgoals))
            for j in range(0, max(1, num_frames - chunk_size)):
                episode_lookup.append(
                    (
                        start_idx
                        + (j // (self.max_window_size - chunk_size + 1))
                        * (self.max_window_size - chunk_size + 1),
                        start_idx
                        + (j // (self.max_window_size - chunk_size + 1))
                        * (self.max_window_size - chunk_size + 1)
                        + self.max_window_size,
                        j % (self.max_window_size - chunk_size + 1),
                    )
                )
        return np.array(episode_lookup)

    def _get_sequences(self, idx: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx)

        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        seq_feat = process_features(
            episode, self.with_feat, self.feat_stats, self.norm_feat
        )
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
            **seq_feat,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def _pad_sequence(self, seq: Dict, pad_size: int) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        if self.with_feat:
            seq.update(
                {"features": self._pad_with_repetition(seq["features"], pad_size)}
            )
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            # repeat action for world coordinates action space
            seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            seq_acts = torch.cat(
                [
                    self._pad_with_zeros(seq["actions"][..., :-1], pad_size),
                    self._pad_with_repetition(seq["actions"][..., -1:], pad_size),
                ],
                dim=-1,
            )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    def _pad_with_repetition(
        self, input_tensor: torch.Tensor, pad_size: int
    ) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(
            torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    def _pad_with_zeros(
        self, input_tensor: torch.Tensor, pad_size: int
    ) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return int(np.ceil(self.max_window_size / self.num_subgoals)) - len(
            sequence["actions"]
        )

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """

        sequence = self._get_sequences(idx)
        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size)

        if not self.with_feat:
            views_static = []
            views_gripper = []
            for key in sequence["rgb_obs"].keys():
                if key == "rgb_static":
                    views_static.append(sequence["rgb_obs"][key][0])
                elif key == "rgb_gripper":
                    views_gripper.append(sequence["rgb_obs"][key][0])
            for key in sequence["depth_obs"].keys():
                if key == "depth_static":
                    views_static.append(sequence["depth_obs"][key][0])
                elif key == "depth_gripper":
                    views_gripper.append(sequence["depth_obs"][key][0])

            assert len(views_static) > 0 or len(views_gripper) > 0

            start_image_static = (
                torch.cat(views_static, dim=0)[None] if len(views_static) > 0 else None
            )
            start_image_gripper = (
                torch.cat(views_gripper, dim=0)[None]
                if len(views_gripper) > 0
                else None
            )

            target_views_static = []
            target_views_gripper = []

            for key in sequence["rgb_obs"].keys():
                if key == "rgb_static":
                    target_views_static.append(
                        get_stochastic_augmentation(p=self.prob_data_aug)(
                            sequence["rgb_obs"][key][-1]
                        )
                    )
                elif key == "rgb_gripper":
                    target_views_gripper.append(
                        get_stochastic_augmentation(p=self.prob_data_aug)(
                            sequence["rgb_obs"][key][-1]
                        )
                    )
            for key in sequence["depth_obs"].keys():
                if key == "depth_static":
                    target_views_static.append(sequence["depth_obs"][key][-1])
                elif key == "depth_gripper":
                    target_views_gripper.append(sequence["depth_obs"][key][-1])

            end_image_static = (
                torch.cat(target_views_static, dim=0)[None]
                if len(target_views_static) > 0
                else None
            )
            end_image_gripper = (
                torch.cat(target_views_gripper, dim=0)[None]
                if len(target_views_gripper) > 0
                else None
            )
            start_end_images_static = (
                torch.cat([start_image_static, end_image_static], dim=0)
                if len(target_views_static) > 0
                else None
            )
            start_end_images_gripper = (
                torch.cat([start_image_gripper, end_image_gripper], dim=0)
                if len(target_views_gripper) > 0
                else None
            )

        elif self.with_feat:
            start_image = sequence["features"][0][None]
            start_image = rearrange(
                start_image,
                "f (w h) c -> f c w h",
                w=self.feat_patch_size,
                h=self.feat_patch_size,
            )
            end_image = sequence["features"][-1][None]
            end_image = rearrange(
                end_image,
                "f (w h) c -> f c w h",
                w=self.feat_patch_size,
                h=self.feat_patch_size,
            )

            start_end_images_static = torch.cat([start_image, end_image], dim=0)
            state = torch.zeros((start_end_images_static.shape[0], 0))

            views_static = [start_image]
            views_gripper = []

        state = (
            torch.zeros((start_end_images_static.shape[0], 0))
            if len(views_static) > 0
            else torch.zeros((start_end_images_gripper.shape[0], 0))
        )

        actions = sequence["actions"][:-1]
        action_is_pad = torch.zeros_like(actions)

        res = {
            "observation.state": state,
            "action": actions,
            "action_is_pad": action_is_pad,
        }
        if len(views_static) > 0:
            res["observation.image_static"] = start_end_images_static
        if len(views_gripper) > 0:
            res["observation.image_gripper"] = start_end_images_gripper
        return res


class DiskEvaluatorDataset(BaseDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        num_subgoals: Number of subgoals per episodes.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        num_subgoals: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.num_subgoals = num_subgoals
        self.with_lang = True
        self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_task = (
            self._build_file_indices_lang(self.abs_datasets_dir)
        )

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )

        task2ann_path = root_path + "/conf/annotations/new_playtable.yaml"
        print("Loading task2ann from: ", task2ann_path)
        self.task2ann = OmegaConf.load(task2ann_path)

    def _get_episode_name(self, file_idx: int) -> Path:
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _get_feat_name(self, file_idx: int) -> Path:
        return Path(
            self.abs_datasets_dir
            / f"features_{self.diffuse_on}/features_{file_idx}.npz"
        )

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load num_goals frames of the episodes (plus the initial frame) evenly spaced.

        Args:
            idx: Index of first frame.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx, end_idx, sucess = self.episode_lookup[idx]

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        # Positive example
        frames_idx = np.linspace(
            start_idx,
            end_idx,
            min(self.max_window_size, self.num_subgoals + 1),
            dtype=int,
        )

        frames = [
            self.load_file(self._get_episode_name(file_idx)) for file_idx in frames_idx
        ]

        episode = {key: np.stack([ep[key] for ep in frames]) for key in keys}
        if sucess == 1:
            episode["task"] = self.lang_task[self.lang_lookup[idx]].replace("_", " ")
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
        else:
            true_task = self.lang_task[self.lang_lookup[idx]].replace("_", " ")
            # Lifting
            if "lift" in true_task:
                filtered_tasks = set(
                    [
                        task
                        for task in self.lang_task
                        if "lift" in task and task != true_task
                    ]
                )
            # Rotating
            elif "rotate" in true_task:
                filtered_tasks = set(
                    [
                        task
                        for task in self.lang_task
                        if "rotate" in task and task != true_task
                    ]
                )
            # Pushing
            elif "push" in true_task:
                filtered_tasks = set(
                    [
                        task
                        for task in self.lang_task
                        if "push" in task and task != true_task
                    ]
                )
            # Open/close
            elif "move" in true_task or "close" in true_task or "open" in true_task:
                filtered_tasks = set(
                    [
                        task
                        for task in self.lang_task
                        if ("move" in task or "close" in task or "open" in task)
                        and task != true_task
                    ]
                )
            # Place
            elif "place" in true_task or "stack" in true_task or "unstack" in true_task:
                filtered_tasks = set(
                    [
                        task
                        for task in self.lang_task
                        if ("place" in task or "stack" in task or "unstack" in task)
                        and task != true_task
                    ]
                )
            # Lightning
            elif "led" in true_task or "lightbulb" in true_task:
                filtered_tasks = set(
                    [
                        task
                        for task in self.lang_task
                        if ("led" in task or "lightbulb" in task) and task != true_task
                    ]
                )
            else:
                raise ValueError(f"Unknown task '{true_task}'")

            # Select a random task from the filtered tasks
            assert len(filtered_tasks) > 0

            episode["task"] = random.choice(list(filtered_tasks))
            false_ann = self.task2ann[episode["task"]]
            episode["language"] = random.choice(false_ann)

        if self.with_feat:
            features = [
                self.load_file(self._get_feat_name(file_idx))["patch_emb"]
                for file_idx in frames_idx
            ]
            episode["features"] = features
        return episode, sucess

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language tasks.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        # Load lang data from pickle
        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are <=64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]  # length total number of annotations
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            assert end_idx >= self.max_window_size
            lang_lookup.append(i)
            lang_lookup.append(i)
            episode_lookup.append((start_idx, end_idx, 0))
            episode_lookup.append((start_idx, end_idx, 1))
        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _get_sequences(self, idx: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode, sucess = self._load_episode(idx)

        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_feat = process_features(
            episode, self.with_feat, self.feat_stats, self.norm_feat
        )
        seq_lang = process_language(episode, self.transforms, self.with_lang)

        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
            **seq_feat,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict, sucess

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """

        sequence, sucess = self._get_sequences(idx)
        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size)
        views = []
        for key in sequence["rgb_obs"].keys():
            views.append(sequence["rgb_obs"][key])
        for key in sequence["depth_obs"].keys():
            views.append(sequence["depth_obs"][key])
        images = torch.cat(views, dim=0)
        features = sequence["features"]
        if self.with_feat:
            images = features
        # task = sequence["lang"]
        task = sequence["task"]
        return images, task, sucess


class DiskDiffusionOracleDataset(BaseDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        num_subgoals: Number of subgoals per episodes.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        num_subgoals: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.num_subgoals = num_subgoals
        if self.with_lang:
            self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_task = (
                self._build_file_indices_lang(self.abs_datasets_dir)
            )
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )

    def _get_episode_name(self, file_idx: int) -> Path:
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _get_feat_name(self, file_idx: int) -> Path:
        return Path(
            self.abs_datasets_dir
            / f"features_{self.diffuse_on}/features_{file_idx}.npz"
        )

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load num_goals frames of the episodes (plus the initial frame) evenly spaced.

        Args:
            idx: Index of first frame.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx, end_idx = self.episode_lookup[idx]

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes_idx = np.linspace(
            start_idx,
            end_idx,
            min(self.max_window_size, self.num_subgoals + 1),
            dtype=int,
        )
        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in episodes_idx
        ]

        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["task"] = self.lang_task[self.lang_lookup[idx]]
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
        if self.with_feat:
            features = [
                self.load_file(self._get_feat_name(file_idx))["patch_emb"]
                for file_idx in episodes_idx
            ]
            episode["features"] = features
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language tasks.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        # Load lang data from pickle
        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are <=64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]  # length total number of annotations
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            assert end_idx >= self.max_window_size
            lang_lookup.append(i)
            episode_lookup.append((start_idx, end_idx))
        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """

        sequence = self._get_sequences(idx)
        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size)
        return sequence

    def _get_sequences(self, idx: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx)

        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        seq_feat = process_features(
            episode, self.with_feat, self.feat_stats, self.norm_feat
        )
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
            **seq_feat,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.num_subgoals + 1 - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            # repeat action for world coordinates action space
            seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            seq_acts = torch.cat(
                [
                    self._pad_with_zeros(seq["actions"][..., :-1], pad_size),
                    self._pad_with_repetition(seq["actions"][..., -1:], pad_size),
                ],
                dim=-1,
            )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(
            torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded
