
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


class dis_pifeature_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        all_dir,
        step_list
    ):
        # all_dir = "/gemini/platform/public/embodiedAI/users/ysy/data/dataset/feature_distance/container_place"
        feature_dict = torch.load(f"{all_dir}/feature_all.pt", map_location="cpu")
        distance_dict = torch.load(f"{all_dir}/distance_all.pt", map_location="cpu")

        features = []
        distances = []
        for i in range(len(step_list)):
            features.append(feature_dict[f"denoise_step{step_list[i]}"])
            distances.append(distance_dict[f"denoise_step{step_list[i]}"])
        features = torch.cat(features, dim=0)
        distances = torch.cat(distances, dim=0)
        num_infer, num_noise, feature_dim = features.shape
        features = features.reshape(num_infer * num_noise, feature_dim)
        distances = distances.reshape(num_infer * num_noise)

        self.features = features
        self.distances = distances

        self.len = len(features)
        # import ipdb;ipdb.set_trace()

        self.CoinFlipMaker = CoinFlipMaker()

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> dict:
        item = {}
        item["feature"] = self.features[idx]
        item["distance"] = self.distances[idx]
        # item["CoinFlip_target"] = self.CoinFlipMaker(idx)

        return item

