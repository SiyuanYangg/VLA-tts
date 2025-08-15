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
from cfn.pi0_cfn.cfn_net_pi import CFNWrapper_pi

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
from torch.utils.tensorboard import SummaryWriter
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
        image_transforms = ImageTransforms.create_jax_pi0_main_camera_transform(img_size=(height, width)) if cfg.dataset.image_transforms.enable else None
        wrist_transforms = ImageTransforms.create_jax_pi0_wrist_camera_transform(img_size=(height, width)) if cfg.dataset.wrist_transforms.enable else None

        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        # breakpoint()

        # get total_episodes
        with open(f"{cfg.dataset.root}/meta/info.json", "r", encoding="utf-8") as f:
            data_cfg = json.load(f)
        print(f"total_episodes = {data_cfg['total_episodes']} !!!!")       
        total_episodes = data_cfg["total_episodes"] ###############################

        # 按比例划分 95% 训练，5% 测试
        train_rat = 0.999
        train_episodes = list(range(int(train_rat * total_episodes)))
        test_episodes = list(range(int(train_rat * total_episodes), total_episodes))

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
        val_dataset = cfn_lerobot_dataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=test_episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            wrist_transforms=wrist_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
            use_delta_action=cfg.policy.use_delta_action,
        )
        # breakpoint()
    else:
        raise ValueError(f"Invalid dataset repo_id: {cfg.dataset.repo_id}")

    return dataset, val_dataset

def evaluate(model, val_dataset, device, cfn_action_steps):
    model.eval()
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
            # state = batch['observation.state'].to(device)
            # action = batch["action"].to(device)
            # instructions = batch["task"]
            # action = action[:, :cfn_action_steps, :]
            # batch_size = action.shape[0]
            # action = action.reshape(batch_size, -1)
            # target = batch["CoinFlip_target"].float().to(device)

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
    dataset, val_dataset = make_dataset(cfg)
    t1 = time.time()
    print(f"📦 数据集加载时间: {t1 - t0:.2f}s")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    cfn_action_steps = 30
    assert len(dataset[0]['observation.state'].shape) == 1
    assert len(dataset[0]['action'].shape) == 2
    assert cfn_action_steps <= dataset[0]['action'].shape[0]

    t0 = time.time()
    model = CFNWrapper_pi(
        cfn_output_dim=getattr(cfg.policy, "cfn_output_dim", 20),
        pretrained_checkpoint_path="/gemini/platform/public/embodiedAI/users/ysy/data/dataset/rt_pi0_ckpt/25-07-21_12-18-18_pi0_gpu2_ck50_lr3e-5_bs12_s120K_seed42/checkpoints/060000/pretrained_model",
    ).to(device)
    t1 = time.time()
    print(f"🧠 模型初始化时间: {t1 - t0:.2f}s")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    loss_meter = AverageMeter("loss", ":.4f")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    model.train()
    num_batches_per_epoch = len(dataset) // cfg.batch_size
    total_epochs = 2  # cfg.steps // num_batches_per_epoch + 1

    grad_accum_steps = 1  # 每多少个 batch 累积一次梯度
    log_interval = 1  # 每多少个batch做一次log

    for epoch in range(total_epochs):
        print(f"\n📘 Epoch {epoch + 1} 开始")
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        # 时间累加器
        accum_batch_time = 0.0
        accum_data_time = 0.0
        accum_forward_time = 0.0
        accum_backward_time = 0.0
        batch_counter = 0

        batch_start = time.time()
        data_start = time.time()

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            
            data_end = time.time()

            forward_start = time.time()
            loss, model_output_train = model.compute_loss(batch)

            forward_end = time.time()

            loss = loss / grad_accum_steps  # 梯度缩放
            backward_start = time.time()
            loss.backward()
            backward_end = time.time()

            batch_end = time.time()

            total_loss += loss.item() * grad_accum_steps
            batch_counter += 1

            # 计算 model_output_train 的 norm，并记录
            train_norm = model_output_train.norm(p=2, dim=1).mean()  # 每个batch的norm取均值
            writer.add_scalar("Norm/train", train_norm.item(), epoch * num_batches_per_epoch + batch_counter)

            accum_batch_time += batch_end - batch_start
            accum_data_time += data_end - data_start
            accum_forward_time += forward_end - forward_start
            accum_backward_time += backward_end - backward_start

            if batch_counter % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if batch_counter % log_interval == 0:
                step = epoch * num_batches_per_epoch + batch_counter
                avg_loss = total_loss / batch_counter
                writer.add_scalar("Loss/train", avg_loss, step)

                if batch_counter % (log_interval * 100) == 0:  # 每5次log再验证一次
                    avg_val_loss, avg_val_norm = evaluate(model, val_dataset, device, cfn_action_steps)
                    writer.add_scalar("Loss/val", avg_val_loss, step)
                    writer.add_scalar("Norm/val", avg_val_norm, step)

            if batch_counter % 200 == 0:
                print(f"⏱️ 平均每 200 batch 用时: "
                      f"batch={accum_batch_time:.2f}s | "
                      f"data={accum_data_time:.2f}s | "
                      f"forward={accum_forward_time:.2f}s | "
                      f"backward={accum_backward_time:.2f}s")
                accum_batch_time = 0.0
                accum_data_time = 0.0
                accum_forward_time = 0.0
                accum_backward_time = 0.0

            if batch_counter >= num_batches_per_epoch:
                break

            batch_start = time.time()
            data_start = time.time()

        # 如果最后没整除梯度步长，也要 step 一次
        if batch_counter % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / batch_counter
        loss_meter.update(avg_loss)
        print(f"✅ Epoch {epoch + 1} 平均 Loss: {avg_loss:.4f}")
        print(f"🕒 Epoch 总耗时: {time.time() - epoch_start:.2f}s")

        if (epoch + 1) % cfg.save_freq == 0:
            torch.save(model.cfn.state_dict(), output_dir / f"model_epoch{epoch + 1}.pt")

        writer.close()

    print(f"✅ 所有训练完成，总耗时: {time.time() - overall_start:.2f}s")
if __name__ == "__main__":
    train()
