
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
from typing import Callable
import numpy as np
import torch
from transformers import AutoProcessor
from transformers.image_utils import ChannelDimension
import types
import torch.nn.functional as F
from transformers.models.idefics3.image_processing_idefics3 import Idefics3ImageProcessor
from scipy.ndimage import zoom

class CoinFlipMaker:
    def __init__(self, output_dimensions=20, only_zero_flips=False):
        self.output_dimensions = output_dimensions
        self.only_zero_flips = only_zero_flips

    def __call__(self, seed):
        if self.only_zero_flips:
            return np.zeros((self.output_dimensions), dtype=np.float32)
        rng = np.random.RandomState(seed)
        return 2 * rng.binomial(1, 0.5, size=(self.output_dimensions)) - 1


class cfn_pifeature_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        feature_dir,
        step_list
    ):
        features_dict = torch.load(feature_dir, map_location="cpu")
        self.CoinFlipMaker = CoinFlipMaker()
        features = []
        for i in range(len(step_list)):
            features.append(features_dict[f"denoise_step{step_list[i]}"])
        features = torch.cat(features, dim=0)
        self.features = features
        self.len = len(features)

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> dict:
        item = {}
        item["feature"] = self.features[idx]
        item["CoinFlip_target"] = self.CoinFlipMaker(idx)

        return item

