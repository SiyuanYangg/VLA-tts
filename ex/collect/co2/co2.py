from lerobot.common.policies.pretrained import T
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import set_seed
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import time

# from lerobot.common.datasets.factory import make_dataset
from lerobot.common.utils.logging_utils import AverageMeter
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from cfn.pi0_cfn.cfn_net_pi_prior_big import CFNWrapper_pi_prior_big

from lerobot.common.datasets.lerobot_dataset import (
    # LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)

from cfn.cfn_dataset import cfn_lerobot_dataset

from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.factory import resolve_delta_timestamps
import logging
from pprint import pformat
# from torch.utils.tensorboard import SummaryWriter
import json

def make_dataset(cfg: TrainPipelineConfig) -> cfn_lerobot_dataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    # image_transforms = (
    #     ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    # )
    
    if cfg.dataset.repo_id.startswith("["):
        cfg.dataset.repo_id = eval(cfg.dataset.repo_id)

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        # 获取observation.images开头的第一个key
        image_key = next(
            (key for key in ds_meta.features.keys() if key.startswith("observation.images.")),
            None
        )
        if image_key is None:
            raise ValueError("No image key found in the dataset")
        
        # 获取图像维度名称列表
        image_dim_names = ds_meta.features[image_key]['names']
        
        # 找到height和width在names中的索引位置
        height_idx = image_dim_names.index('height') if 'height' in image_dim_names else None
        width_idx = image_dim_names.index('width') if 'width' in image_dim_names else None
        
        if height_idx is None or width_idx is None:
            raise ValueError("Could not find 'height' or 'width' in image dimension names")
            
        # 根据索引从shape中获取实际的高度和宽度值
        image_shape = ds_meta.features[image_key]['shape']
        height = image_shape[height_idx]
        width = image_shape[width_idx]

        # image_transforms = ImageTransforms.create_piohfive_sequential_transform(
        #     (height, width)
        # ) if cfg.dataset.image_transforms.enable else None

        # 做图像增强: 
        # image_transforms = ImageTransforms.create_jax_pi0_main_camera_transform(img_size=(height, width)) if cfg.dataset.image_transforms.enable else None
        # wrist_transforms = ImageTransforms.create_jax_pi0_wrist_camera_transform(img_size=(height, width)) if cfg.dataset.wrist_transforms.enable else None

        # 不做图像增强:
        image_transforms = None
        wrist_transforms = None

        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        # breakpoint()

        # get total_episodes
        with open(f"{cfg.dataset.root}/meta/info.json", "r", encoding="utf-8") as f:
            data_cfg = json.load(f)
        print(f"total_episodes = {data_cfg['total_episodes']} !!!!")       
        total_episodes = data_cfg["total_episodes"] ###############################

        # 按比例划分 95% 训练，5% 测试
        train_rat = 1
        train_episodes = list(range(int(train_rat * total_episodes)))
        # test_episodes = list(range(int(train_rat * total_episodes), total_episodes))

        dataset = cfn_lerobot_dataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=train_episodes,
            # episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            wrist_transforms=wrist_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
            use_delta_action=cfg.policy.use_delta_action,
        )
        # val_dataset = cfn_lerobot_dataset(
        #     cfg.dataset.repo_id,
        #     root=cfg.dataset.root,
        #     episodes=test_episodes,
        #     delta_timestamps=delta_timestamps,
        #     image_transforms=image_transforms,
        #     wrist_transforms=wrist_transforms,
        #     revision=cfg.dataset.revision,
        #     video_backend=cfg.dataset.video_backend,
        #     use_delta_action=cfg.policy.use_delta_action,
        # )
        # breakpoint()
    else:
        raise ValueError(f"Invalid dataset repo_id: {cfg.dataset.repo_id}")

    return dataset

def evaluate(model, val_dataset, device, cfn_action_steps):
    # model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=device.type != "cpu"
    )
    total_val_loss = 0.0
    total_val_norm = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:

            loss, model_output_val = model.compute_loss(batch)
            total_val_loss += loss.item()
            
            # 计算 model_output_val 的 norm，并累计
            val_norm = model_output_val.norm(p=2, dim=1).mean()  # 每个batch的norm取均值
            total_val_norm += val_norm.item()

            num_batches += 1

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_norm = total_val_norm / num_batches  # 平均所有batch的norm
    return avg_val_loss, avg_val_norm


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    overall_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    cfg.validate()

    t0 = time.time()
    dataset = make_dataset(cfg)
    t1 = time.time()
    print(f"📦 数据集加载时间: {t1 - t0:.2f}s")

    # torch.set_printoptions(sci_mode=False)
    # import ipdb; ipdb.set_trace()

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    cfn_action_steps = 30
    assert len(dataset[0]['observation.state'].shape) == 1
    assert len(dataset[0]['action'].shape) == 2
    assert cfn_action_steps <= dataset[0]['action'].shape[0]

    t0 = time.time()
    model = CFNWrapper_pi_prior_big(
        cfn_output_dim=20,
        pretrained_checkpoint_path="/gemini/platform/public/embodiedAI/users/ysy/data/dataset/rt_pi0_ckpt/robotwin_new_transforms_all_tasks_50ep/25-08-06_00-31-57_pi0_gpu4_ck50_lr3e-5_bs12_s60K_seed42/checkpoints/030000/pretrained_model",
    ).to(device)
    t1 = time.time()
    print(f"🧠 模型初始化时间: {t1 - t0:.2f}s")

    num_batches_per_epoch = len(dataset) // cfg.batch_size
    total_epochs = 16  # cfg.steps // num_batches_per_epoch + 1

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(dataloader),
        epochs=total_epochs,
        anneal_strategy='cos',  # 余弦退火
    )   

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    model.train()

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    seed = 42
    torch.manual_seed(seed)
    print(f"noise seed is {seed} !!!")
    # noise = self.policy.model.sample_noise(actions_shape, device, dtype)
    noise_num = 50
    actions_shape = (noise_num, model.policy.model.config.n_action_steps, model.policy.model.config.max_action_dim)
    noise42 = torch.normal(
        mean=0.0,
        std=1.0,
        size=actions_shape,
        dtype=dtype,
    ).to(device)
    #######################################################################################
    # only get one noise
    # noise42 = noise42[[14]]
    # noise_num = 1
    #######################################################################################
    print(f"noise is\n{noise42}")

    # seed = 2345
    # torch.manual_seed(seed)
    # print(f"noise seed is {seed} !!!")
    # noise = self.policy.model.sample_noise(actions_shape, device, dtype)
    # noise_num = 50
    # actions_shape = (noise_num, model.policy.model.config.n_action_steps, model.policy.model.config.max_action_dim)
    # noise2345 = torch.normal(
    #     mean=0.0,
    #     std=1.0,
    #     size=actions_shape,
    #     dtype=dtype,
    # ).to(device)
    # print(f"noise is\n{noise}")

    step_list = [9]
    # step_list = [8, 9]
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir is {output_dir} !!!")

    total_epochs = 1
    for epoch in range(total_epochs):
        print(f"\n📘 Epoch {epoch + 1} 开始")
        model.train()
        model.policy.eval() ######################## note this
        optimizer.zero_grad()

        features_all = {}
        distance_all = {}
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            # 转到cuda
            batch['observation.state'] = batch['observation.state'].to(device)
            batch['observation.images.cam_high'] = batch['observation.images.cam_high'].to(device)
            batch['observation.images.cam_left_wrist'] = batch['observation.images.cam_left_wrist'].to(device)
            batch['observation.images.cam_right_wrist'] = batch['observation.images.cam_right_wrist'].to(device)
            # 推理action用不上
            batch['action'] = batch['action'].to(device)
            # noise_num = 5
            bs = batch['observation.state'].shape[0]
            # import ipdb; ipdb.set_trace()

            for i in tqdm(range(bs)):
                noise_batch = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        noise_batch[key] = value[[i]]
                    elif isinstance(value, list):
                        noise_batch[key] = [value[i]]
                    else:
                        assert 0, "this batch is not right"

                noise_batch['observation.state'] = noise_batch['observation.state'].repeat(noise_num, 1)
                noise_batch['observation.images.cam_high'] = noise_batch['observation.images.cam_high'].repeat(noise_num, 1, 1, 1)
                noise_batch['observation.images.cam_left_wrist'] = noise_batch['observation.images.cam_left_wrist'].repeat(noise_num, 1, 1, 1)
                noise_batch['observation.images.cam_right_wrist'] = noise_batch['observation.images.cam_right_wrist'].repeat(noise_num, 1, 1, 1)
                noise_batch['action'] = noise_batch['action'].repeat(noise_num, 1, 1)
                noise_batch['task'] = noise_batch['task']*noise_num
                # import ipdb;ipdb.set_trace()
                with torch.no_grad():
                    normalized_actions, features = model.policy.get_all(noise_batch, noise42.clone(), step_list)
                assert normalized_actions.shape[0] == len(step_list)
                assert normalized_actions.shape[1] == noise42.shape[0]
                assert features.shape[0] == len(step_list)
                assert features.shape[1] == noise42.shape[0]

                # import ipdb;ipdb.set_trace()
                noise_batch = model.policy.normalize_targets(noise_batch)
                gt_action = noise_batch['action']

                # 选出最优的
                with torch.no_grad():
                    for i in range(len(step_list)):
                        if f"denoise_step{step_list[i]}" not in features_all.keys():
                            features_all[f"denoise_step{step_list[i]}"] = []
                        if f"denoise_step{step_list[i]}" not in distance_all.keys():
                            distance_all[f"denoise_step{step_list[i]}"] = []
                        distance = torch.norm(normalized_actions[i] - gt_action, dim=(1, 2), p=2)
                        distance_all[f"denoise_step{step_list[i]}"].append(distance)
                        # min_index = torch.argmin(norm42)
                        # import ipdb;ipdb.set_trace()
                        features_all[f"denoise_step{step_list[i]}"].append(features[i])

                        # print(f"step is {step_list[i]}, norm_mean is {norm42.mean()}")

                # import ipdb;ipdb.set_trace()
            # aa +=1
            # if aa >2:
            #     break
        
        for i in range(len(step_list)):
            features_all[f"denoise_step{step_list[i]}"] = torch.stack(features_all[f"denoise_step{step_list[i]}"])
            distance_all[f"denoise_step{step_list[i]}"] = torch.stack(distance_all[f"denoise_step{step_list[i]}"])
        
        save_path = output_dir
        torch.save(features_all, save_path / "feature_all.pt")
        torch.save(distance_all, save_path / "distance_all.pt")
        print(f"features and distance have been saved at {save_path}")
        print()


if __name__ == "__main__":
    train()
