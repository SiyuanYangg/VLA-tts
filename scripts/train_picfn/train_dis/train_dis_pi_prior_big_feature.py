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
from cfn.pi0_dis_feature.dis_net_pi_prior_big import DISWrapper_pi_prior_big


from cfn.dis_dataset_feature.dis_feature_dataset import dis_pifeature_dataset

from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.factory import resolve_delta_timestamps
import logging
from pprint import pformat
from torch.utils.tensorboard import SummaryWriter
import json

# def evaluate(model, val_dataset, device, cfn_action_steps):
#     # model.eval()
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=64,
#         shuffle=False,
#         num_workers=8,
#         pin_memory=device.type != "cpu"
#     )
#     total_val_loss = 0.0
#     total_val_norm = 0.0
#     num_batches = 0
#     with torch.no_grad():
#         for batch in val_loader:

#             loss, model_output_val = model.compute_loss(batch)
#             total_val_loss += loss.item()
            
#             # 计算 model_output_val 的 norm，并累计
#             val_norm = model_output_val.norm(p=2, dim=1).mean()  # 每个batch的norm取均值
#             total_val_norm += val_norm.item()

#             num_batches += 1

#     avg_val_loss = total_val_loss / len(val_loader)
#     avg_val_norm = total_val_norm / num_batches  # 平均所有batch的norm
#     return avg_val_loss, avg_val_norm


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    overall_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    cfg.validate()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_name = output_dir.name

    t0 = time.time()
    dataset = dis_pifeature_dataset(
        all_dir=f"/gemini/platform/public/embodiedAI/users/ysy/data/dataset/feature_distance/{task_name}", # 选择的noise 的feature
        # feature_dir=f"/gemini/platform/public/embodiedAI/users/ysy/data/dataset/feature_rt_good_noise/{task_name}/feature.pt", # seed42的第一个noise
        # feature_dir=f"/gemini/platform/public/embodiedAI/users/ysy/data/dataset/feature_rt_index14_noise/{task_name}/feature.pt",
        step_list=[9]
    )
    t1 = time.time()
    print(f"📦 数据集加载时间: {t1 - t0:.2f}s")

    # torch.set_printoptions(sci_mode=False)
    # import ipdb; ipdb.set_trace()

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    t0 = time.time()
    model = DISWrapper_pi_prior_big(
        pretrained_checkpoint_path="/gemini/platform/public/embodiedAI/users/ysy/data/dataset/rt_pi0_ckpt/robotwin_new_transforms_all_tasks_50ep/25-08-06_00-31-57_pi0_gpu4_ck50_lr3e-5_bs12_s60K_seed42/checkpoints/030000/pretrained_model",
    ).to(device)
    t1 = time.time()
    print(f"🧠 模型初始化时间: {t1 - t0:.2f}s")

    num_batches_per_epoch = len(dataset) // cfg.batch_size
    total_epochs = 50  # cfg.steps // num_batches_per_epoch + 1

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=len(dataloader),
        epochs=total_epochs,
        anneal_strategy='cos',  # 余弦退火
    )   

    loss_meter = AverageMeter("loss", ":.4f")

    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    model.train()

    grad_accum_steps = 2  # 每多少个 batch 累积一次梯度
    log_interval = 2  # 每多少个batch做一次log

    for epoch in range(total_epochs):
        print(f"\n📘 Epoch {epoch + 1} 开始")
        epoch_start = time.time()
        model.train()
        model.policy.eval() ######################## note this
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
            loss, dis_pred, var = model.compute_loss_feature(batch)

            forward_end = time.time()

            loss = loss / grad_accum_steps  # 梯度缩放
            backward_start = time.time()
            loss.backward()
            backward_end = time.time()

            batch_end = time.time()

            total_loss += loss.item() * grad_accum_steps
            batch_counter += 1

            # 计算 model_output_train 的 norm，并记录
            writer.add_scalar("Train/dis_pred", dis_pred.mean().item(), epoch * num_batches_per_epoch + batch_counter)
            writer.add_scalar("Train/var", var.mean().item(), epoch * num_batches_per_epoch + batch_counter)
            writer.add_scalar("Prior/prior_mean", model.dis.prior_mean.item(), epoch * num_batches_per_epoch + batch_counter)
            writer.add_scalar("Prior/prior_var", model.dis.prior_logvar.item(), epoch * num_batches_per_epoch + batch_counter)

            accum_batch_time += batch_end - batch_start
            accum_data_time += data_end - data_start
            accum_forward_time += forward_end - forward_start
            accum_backward_time += backward_end - backward_start

            if batch_counter % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # update learning rate
            scheduler.step()

            if batch_counter % log_interval == 0:
                step = epoch * num_batches_per_epoch + batch_counter
                avg_loss = total_loss / batch_counter
                writer.add_scalar("Train/loss", avg_loss, step)
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Train/lr", current_lr, step)
                writer.add_scalar("Train/epoch", epoch+1, step)
                
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
            scheduler.step()

        avg_loss = total_loss / batch_counter
        loss_meter.update(avg_loss)
        print(f"✅ Epoch {epoch + 1} 平均 Loss: {avg_loss:.4f}")
        print(f"🕒 Epoch 总耗时: {time.time() - epoch_start:.2f}s")

        if (epoch + 1) % cfg.save_freq == 0:
            torch.save(model.dis.state_dict(), output_dir / f"model_epoch{epoch + 1}.pt")

    writer.close()

    print(f"✅ 所有训练完成，总耗时: {time.time() - overall_start:.2f}s")
if __name__ == "__main__":
    train()
