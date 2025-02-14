import logging
import os
import pickle
import re
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from calvin_agent.datasets.base_dataset import BaseDataset
from calvin_agent.datasets.utils.episode_utils import process_rgb

logger = logging.getLogger(__name__)


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


class DiskDataset(BaseDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        skip_frames: int = 1,
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
        self.skip_frames = skip_frames
        if self.with_lang:
            self.episode_lookup, self.lang_lookup, self.lang_ann = (
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

    def _get_dino_feat_name(self, file_idx: int) -> Path:
        return Path(self.abs_datasets_dir / f"features/dino_features_{file_idx}.npz")

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load entire_episode which frames spaced by skip_frames.

        Args:
            idx: Index of first frame.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx, end_idx = self.episode_lookup[idx]

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        num_frames = end_idx - start_idx + 1
        episodes_idx = np.linspace(
            start_idx,
            end_idx,
            min(self.max_window_size, num_frames // self.skip_frames),
            dtype=int,
        )
        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in episodes_idx
        ]

        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
        if self.with_dino_feat:
            dino_features = [
                self.load_file(self._get_dino_feat_name(file_idx))["patch_emb"]
                for file_idx in episodes_idx
            ]
            episode["dino_features"] = dino_features
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
        lang_ann = lang_data["language"]["task"]  # length total number of annotations
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            assert end_idx >= self.max_window_size
            lang_lookup.append(i)
            episode_lookup.append((start_idx, end_idx))

        return np.array(episode_lookup), lang_lookup, lang_ann


class DiskImageDataset(BaseDataset):
    """
    Dataset that loads individual images from episodes while ensuring each image is seen once per epoch.

    Args:
        skip_frames: Step size for skipping frames in a trajectory.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        skip_frames: int = 1,
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
        self.skip_frames = skip_frames

        if self.with_lang:
            self.episode_lookup, self.lang_lookup, self.lang_ann = (
                self._build_file_indices_lang(self.abs_datasets_dir)
            )
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )

        # Precompute all (trajectory, frame) index pairs for exact coverage
        self.image_indices = self._generate_image_indices()

    def _get_episode_name(self, file_idx: int) -> Path:
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _generate_image_indices(self) -> List[Tuple[int, int]]:
        """
        Generates a list of (episode_idx, frame_idx) pairs to ensure every image is seen once per epoch.
        """
        image_indices = []
        for episode_idx, (start_idx, end_idx) in enumerate(self.episode_lookup):
            frames = np.arange(start_idx, end_idx + 1, self.skip_frames)
            image_indices.extend([(episode_idx, frame) for frame in frames])

        return image_indices

    def __len__(self) -> int:
        return len(self.image_indices)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """
        Returns a single image from a trajectory.

        Args:
            index: Index in the precomputed list of (episode, frame) pairs.
        Returns:
            sample: Dict containing a single image frame.
        """
        episode_idx, frame_idx = self.image_indices[index]

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")

        episode = self.load_file(self._get_episode_name(frame_idx))

        sample = {key: episode[key] for key in keys}
        rgb_obs = process_rgb(sample, self.observation_space, self.transforms)

        return frame_idx, rgb_obs["rgb_obs"]["rgb_static"].squeeze(0)

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
        lang_ann = lang_data["language"]["task"]
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            assert end_idx >= self.max_window_size
            lang_lookup.append(i)
            episode_lookup.append((start_idx, end_idx))

        return np.array(episode_lookup), lang_lookup, lang_ann
