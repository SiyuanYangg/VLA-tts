
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

# @profile
def resize_numpy(
    self,
    image,  # np.ndarray
    size,
    data_format=None,
    resample=None,
    input_data_format=None,
    antialias: bool = True,
) -> np.ndarray:
    """
    Resize a numpy image using scipy.ndimage.zoom.

    Args:
        image (np.ndarray): Array of shape (C, H, W) or (H, W, C)
        size (dict): Must contain 'longest_edge' or both 'height' and 'width'
        data_format: Desired output format, 'channels_first' or 'channels_last'
        input_data_format: Input format, inferred if None
        antialias: Whether to apply anti-aliasing during resizing
    Returns:
        np.ndarray: Resized image in specified `data_format`
    """    

    # import ipdb; ipdb.set_trace()

    # Infer input format
    if input_data_format is None:
        input_data_format = (
            ChannelDimension.LAST if image.ndim == 3 and image.shape[-1] in [1, 3, 4]
            else ChannelDimension.FIRST
        )
    data_format = input_data_format

    # Convert to channels_first for resizing
    if input_data_format == ChannelDimension.LAST:
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW

    c, h, w = image.shape

    # Compute target size
    if "longest_edge" in size:
        scale = size["longest_edge"] / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
    elif "height" in size and "width" in size:
        new_h, new_w = size["height"], size["width"]
    else:
        raise ValueError("size must contain 'longest_edge' or both 'height' and 'width'")

    # Compute zoom factors per axis
    zoom_factors = (1, new_h / h, new_w / w)
    image = zoom(image, zoom_factors, order=1 if antialias else 0)  # order=1: bilinear

    # Convert back to desired format
    if data_format == ChannelDimension.LAST:
        image = np.transpose(image, (1, 2, 0))  # CHW → HWC

    # import ipdb; ipdb.set_trace()
    return image


def resize_torch(
    self, 
    image,  # np.ndarray
    size,
    data_format=None,
    resample=None,
    input_data_format=None,
    antialias: bool = True,
) -> np.ndarray:
    """
    Resize a numpy image using torch.nn.functional.interpolate.

    Args:
        image (np.ndarray): Array of shape (C, H, W) or (H, W, C)
        size (dict): Must contain 'longest_edge' or both 'height' and 'width'
        data_format: Desired output format, 'channels_first' or 'channels_last'
        input_data_format: Input format, inferred if None
        antialias: Whether to apply anti-aliasing during resizing
    Returns:
        np.ndarray: Resized image in specified `data_format`
    """
    # import ipdb; ipdb.set_trace()

    # Convert numpy to torch.Tensor
    image = torch.from_numpy(image)

    # Infer input format
    if input_data_format is None:
        input_data_format = (
            ChannelDimension.LAST if image.ndim == 3 and image.shape[-1] in [1, 3, 4]
            else ChannelDimension.FIRST
        )
    data_format = input_data_format

    # Convert to channels_first for interpolate
    if input_data_format == ChannelDimension.LAST:
        image = image.permute(2, 0, 1)  # HWC → CHW

    c, h, w = image.shape

    # Compute target size
    if "longest_edge" in size:
        scale = size["longest_edge"] / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
    elif "height" in size and "width" in size:
        new_h, new_w = size["height"], size["width"]
    else:
        raise ValueError("size must contain 'longest_edge' or both 'height' and 'width'")

    # import ipdb; ipdb.set_trace()
    image = image.unsqueeze(0).float()  # Add batch dim
    image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=antialias)
    image = image.squeeze(0)  # Remove batch dim

    # Convert back to channels_last if needed
    if data_format == ChannelDimension.LAST:
        image = image.permute(1, 2, 0)  # CHW → HWC

    # import ipdb; ipdb.set_trace()
    return image.numpy()

class cfn_lerobot_dataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        wrist_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        use_delta_action: bool = False,
        vision_model_name='HuggingFaceTB/SmolVLM-500M-Instruct',
    ):
        # breakpoint()
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            wrist_transforms=wrist_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            use_delta_action=use_delta_action,
        )

        self.CoinFlipMaker = CoinFlipMaker()


    # @profile
    def __getitem__(self, idx) -> dict:

        import time

        start_time = time.time()

        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        # breakpoint()
        
        # 映射到真实的ep_idx, 本来是数据集中的ep索引
        ep_idx = ep_idx - self.episodes[0]

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if self.use_delta_action:
            item = self.delta_action_transform(item)

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}
        
        time1 = time.time()
        delta1 = time1 - start_time

        images_list = []
        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                if 'depth' in cam:
                    continue

                if 'wrist' in cam or 'image_1' in cam or 'image_2' in cam:
                    item[cam] = self.wrist_transforms(item[cam])
                else:
                    item[cam] = self.image_transforms(item[cam])
                
                images_list.append(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        item["CoinFlip_target"] = self.CoinFlipMaker(self.episodes[0] + idx)

        return item

    def extract_smol_inputs(self, images_list, text):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": text}
                ]
            }        
        ]

        prompt = self.processor.apply_chat_template(message, add_generation_prompt=True)
        input_ = self.processor(text=prompt, images=images_list, return_tensors="pt")

        return input_
    
    def pad_input_ids(self, max_length, input_ids, attention_mask):
        """
        将 input_ids 和 attention_mask pad 到 max_length。
        
        参数:
            max_length (int): 目标长度
            input_ids (torch.Tensor): shape = (batch_size, seq_len)
            attention_mask (torch.Tensor): shape = (batch_size, seq_len)
        
        返回:
            padded_input_ids (torch.Tensor): shape = (batch_size, max_length)
            padded_attention_mask (torch.Tensor): shape = (batch_size, max_length)
        """
        pad_token_id = 2
        pad_attention = 0

        batch_size, seq_len = input_ids.shape
        if seq_len >= max_length:
            return input_ids[:, :max_length], attention_mask[:, :max_length]
        
        pad_len = max_length - seq_len

        # 构造 padding 区域
        pad_input = torch.full((batch_size, pad_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        pad_mask = torch.full((batch_size, pad_len), pad_attention, dtype=attention_mask.dtype, device=attention_mask.device)

        # 拼接
        padded_input_ids = torch.cat([input_ids, pad_input], dim=1)
        padded_attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        return padded_input_ids, padded_attention_mask

    # def extract_images_from_batch(self, batch):
    #     """
    #     从 batch 中提取 3 张图像并转为 PIL 格式
    #     返回: List[List[PIL.Image]]，每个样本含 3 张图
    #     """
    #     cam_high = batch['observation.images.cam_high']
    #     cam_left = batch['observation.images.cam_left_wrist']
    #     cam_right = batch['observation.images.cam_right_wrist']
    #     B = cam_high.shape[0]
    #     # import ipdb; ipdb.set_trace()
    #     images = []
    #     for i in range(B):
    #         imgs = [
    #             self.to_pil(cam_high[i].cpu()),
    #             self.to_pil(cam_left[i].cpu()),
    #             self.to_pil(cam_right[i].cpu())
    #             # cam_high[i],
    #             # cam_left[i],
    #             # cam_right[i]
    #         ]
    #         images.append(imgs)
        
    #     # import ipdb; ipdb.set_trace()
    #     return images


    # def encode_instruction(self, image_list, text_list):
    #     # 构建 chat prompt（每个样本含 3 张图）
    #     messages_batch = []
    #     for text in text_list:
    #         messages_batch.append({
    #             "role": "user",
    #             "content": [
    #                 {"type": "image"},
    #                 {"type": "image"},
    #                 {"type": "image"},
    #                 {"type": "text", "text": text}
    #             ]
    #         })

    #     # breakpoint()
    #     prompts = [self.processor.apply_chat_template([m], add_generation_prompt=True)
    #                for m in messages_batch]

    #     inputs = self.processor(
    #         text=prompts,
    #         images=image_list,  # List[List[PIL.Image]]
    #         return_tensors="pt",
    #         padding=True, 
    #         do_resize=False,
    #     ).to(next(self.parameters()).device)

        
    #     # from torchvision.transforms.functional import to_pil_image
    #     # img_tensor = inputs['pixel_values'][0, 0]  # shape: [3, 512, 512]
    #     # # 确保在 CPU 上
    #     # img_tensor = img_tensor.detach().cpu()
    #     # # 将张量转换为 PIL 图像
    #     # img = to_pil_image(img_tensor)
    #     # # 保存图像到磁盘
    #     # img.save("restored_image_2_0726.png")

    #     # import ipdb; ipdb.set_trace()

    #     # breakpoint()
    #     with torch.no_grad():
    #         outputs = self.vision_model(**inputs, output_hidden_states=True)
    #         # outputs = self.vision_model(**inputs)
    #         # print('good')
    #         # breakpoint()
    #         last_hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]

    #     cls_embedding = last_hidden[:, 0, :]  # or mean-pooling
    #     return self.language_proj(cls_embedding)  # [B, embed_dim]
